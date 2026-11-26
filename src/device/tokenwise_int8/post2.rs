//! The post-FFN norm, residual add and layer scalar of the FFN fast path, one token per slice
//! (`post::post_ffn_pass`'s arithmetic, bit for bit).
//!
//! `post` feeds the gamma and the residual to the vector engine as 7.5 KiB VRF operands, one row
//! half at a time (the VRF holds 8 KiB), so each half waits for a Sub-context VRF store before its
//! Main pass: four stores serialized with five short passes at the very end of the stage. Here the
//! gamma and the residual are paired with the stream instead (`begin_interleaved` and the
//! vector engine's pair mode), so the phase is three Main passes and no large VRF operand:
//! 1. `n = bf16((y / rms) * g)`: pair (y, g), y divided by the row's rms, zipped by a multiply;
//! 2. `v = bf16(f32(n) + f32(x))`: pair (n, x), zipped by the clip stage's f32 add (`post`'s op);
//! 3. `bf16(f32(v) * s)` with the layer scalar.
//!
//! Each pair is read in unzip groups of 128 elements (`Kh / 128, Pi, Kh % 128 / 8`): one-flit
//! groups run a pair pass at about twice its time.
use furiosa_opt_std::prelude::*;

use super::gemm::Pi;
use super::rotate::{Kh, TCl, TSl};
use crate::axes::H;
use crate::{Chip, EPS};

type Row<D, T, const HALF: usize, E> = DmTensor<D, Chip, TCl<T, HALF>, TSl<T, HALF>, E>;

/// `rms = sqrt(fold(y^2) / 3840 + eps)` of one row in the VRF (`post::rms`).
fn rms<T: AxisName, const HALF: usize>(device: &mut Device, y: &Row<bf16, T, HALF, m![Kh]>) -> VrfTensor<f32, Chip, TCl<T, HALF>, TSl<T, HALF>, m![1 # 8]> {
    let ms: Row<f32, T, HALF, m![1 # 8]> = device
        .main
        .begin(y.view())
        .fetch::<m![Kh / 16], m![Kh % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Kh / 8], m![Kh % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Kh / 4], m![Kh % 4]>()
        .vector_stash()
        // Delay the stash read by two vector stages to keep the square pipelined.
        .vector_fp_binary(FpBinaryOp::AddF, -0f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<Kh, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(3840f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: Row<f32, T, HALF, m![1 # 8]> = device
        .main
        .begin(ms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    device.sub.begin(rms.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf()
}

/// `post_ffn_pass` with the residual rows from DM and the result returned in DM (the caller
/// stores it): the DM hand-off from the O stage and to the next QKV stage.
pub(crate) fn post_ffn_pass_dm<T: AxisName, const HALF: usize, const PS: usize, const P: usize, const PI: usize>(
    device: &mut Device,
    y: &HbmTensor<bf16, Chip, m![T, H]>,
    gamma: &Row<bf16, T, HALF, m![Kh]>,
    s: &VrfTensor<f32, Chip, TCl<T, HALF>, TSl<T, HALF>, m![1 # 8]>,
    xd: &Row<bf16, T, HALF, m![Kh]>,
) -> Row<bf16, T, HALF, m![Kh]> {
    let yd: Row<bf16, T, HALF, m![H]> = y
        .view()
        .tile::<m![T / { PS }], 1, m![T / { PS } = 1 # { P }, T % { PS }, H]>(PI)
        .to_dm(&mut device.tdma);
    let yd: Row<bf16, T, HALF, m![Kh]> = unsafe { yd.reshape() };
    let r = rms::<T, HALF>(device, &yd);
    let n: Row<bf16, T, HALF, m![Kh]> = device
        .main
        .begin_interleaved::<Pi, _, _, _, _, _>(yd.view(), gamma.view())
        .fetch::<m![Kh / 128, Pi, Kh % 128 / 8], m![Kh % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Kh / 128, Pi, Kh % 128 / 8], m![Kh % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Pi, m![Kh / 8]>()
        .vector_narrow_split::<m![Kh / 8, Kh / 4 % 2], m![Kh % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &r, ())
        .vector_fp_zip(FpBinaryOp::MulF(FpMulAlu::Mul0))
        .vector_widen_concat::<m![Kh / 8], m![Kh % 8]>()
        .vector_final()
        .cast::<bf16, m![Kh % 8 # 16]>()
        .commit_trim::<m![Kh % 8]>()
        .commit();
    let v: Row<bf16, T, HALF, m![Kh]> = device
        .main
        .begin_interleaved::<Pi, _, _, _, _, _>(n.view(), xd.view())
        .fetch::<m![Kh / 128, Pi, Kh % 128 / 8], m![Kh % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Kh / 128, Pi, Kh % 128 / 8], m![Kh % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Pi, m![Kh / 8]>()
        .vector_clip_zip(ClipBinaryOpF32::Add)
        .vector_final()
        .cast::<bf16, m![Kh % 8 # 16]>()
        .commit_trim::<m![Kh % 8]>()
        .commit();
    let out: Row<bf16, T, HALF, m![Kh]> = device
        .main
        .begin(v.view())
        .fetch::<m![Kh / 16], m![Kh % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Kh / 8], m![Kh % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Kh / 4], m![Kh % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), s)
        .vector_widen_concat::<m![Kh / 8], m![Kh % 8]>()
        .vector_final()
        .cast::<bf16, m![Kh % 8 # 16]>()
        .commit_trim::<m![Kh % 8]>()
        .commit();
    out
}

/// `post::post_attention_pass_dm` in this module's pair form: `bf16(f32(bf16(y / rms(y) * gamma))
/// + f32(x))` with `y` and `x` loaded from HBM, the gamma and the residual paired with the stream
/// (128 elements per unzip group) instead of `post`'s half-row VRF operands, each of which waits for
/// a Sub-context VRF store before its Main pass. The same operations in the same order (`DivF` by
/// the row's rms, then a multiply by the gamma; an f32 add), so the bits are `post`'s.
/// Used by the two-pass (Tok1024) v9 O stage: 14 us faster per segment there, 1.6 us slower than
/// `post`'s form at Tok512 (216757).
pub(crate) fn post_attention_pass_dm<T: AxisName, const HALF: usize, const PS: usize, const P: usize, const PI: usize>(
    device: &mut Device,
    y: &HbmTensor<bf16, Chip, m![T, H]>,
    gamma: &Row<bf16, T, HALF, m![Kh]>,
    x: &HbmTensor<bf16, Chip, m![T, H]>,
) -> Row<bf16, T, HALF, m![Kh]> {
    let yd: Row<bf16, T, HALF, m![H]> = y
        .view()
        .tile::<m![T / { PS }], 1, m![T / { PS } = 1 # { P }, T % { PS }, H]>(PI)
        .to_dm(&mut device.tdma);
    let xd: Row<bf16, T, HALF, m![H]> = x
        .view()
        .tile::<m![T / { PS }], 1, m![T / { PS } = 1 # { P }, T % { PS }, H]>(PI)
        .to_dm(&mut device.tdma);
    let yd: Row<bf16, T, HALF, m![Kh]> = unsafe { yd.reshape() };
    let xd: Row<bf16, T, HALF, m![Kh]> = unsafe { xd.reshape() };
    let r = rms::<T, HALF>(device, &yd);
    let n: Row<bf16, T, HALF, m![Kh]> = device
        .main
        .begin_interleaved::<Pi, _, _, _, _, _>(yd.view(), gamma.view())
        .fetch::<m![Kh / 128, Pi, Kh % 128 / 8], m![Kh % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Kh / 128, Pi, Kh % 128 / 8], m![Kh % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Pi, m![Kh / 8]>()
        .vector_narrow_split::<m![Kh / 8, Kh / 4 % 2], m![Kh % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &r, ())
        .vector_fp_zip(FpBinaryOp::MulF(FpMulAlu::Mul0))
        .vector_widen_concat::<m![Kh / 8], m![Kh % 8]>()
        .vector_final()
        .cast::<bf16, m![Kh % 8 # 16]>()
        .commit_trim::<m![Kh % 8]>()
        .commit();
    device
        .main
        .begin_interleaved::<Pi, _, _, _, _, _>(n.view(), xd.view())
        .fetch::<m![Kh / 128, Pi, Kh % 128 / 8], m![Kh % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Kh / 128, Pi, Kh % 128 / 8], m![Kh % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Pi, m![Kh / 8]>()
        .vector_clip_zip(ClipBinaryOpF32::Add)
        .vector_final()
        .cast::<bf16, m![Kh % 8 # 16]>()
        .commit_trim::<m![Kh % 8]>()
        .commit()
}

/// Same arithmetic as post_attention_pass_dm, receiving its O rows directly from DM.
pub(crate) fn post_attention_from_dm<T: AxisName, const HALF: usize, const PS: usize, const P: usize, const PI: usize>(
    device: &mut Device,
    yd: &Row<bf16, T, HALF, m![Kh]>,
    gamma: &Row<bf16, T, HALF, m![Kh]>,
    x: &HbmTensor<bf16, Chip, m![T, H]>,
) -> Row<bf16, T, HALF, m![Kh]> {
    let xd: Row<bf16, T, HALF, m![H]> = x
        .view()
        .tile::<m![T / { PS }], 1, m![T / { PS } = 1 # { P }, T % { PS }, H]>(PI)
        .to_dm(&mut device.tdma);
    let xd: Row<bf16, T, HALF, m![Kh]> = unsafe { xd.reshape() };
    let r = rms::<T, HALF>(device, yd);
    let n: Row<bf16, T, HALF, m![Kh]> = device
        .main
        .begin_interleaved::<Pi, _, _, _, _, _>(yd.view(), gamma.view())
        .fetch::<m![Kh / 128, Pi, Kh % 128 / 8], m![Kh % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Kh / 128, Pi, Kh % 128 / 8], m![Kh % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Pi, m![Kh / 8]>()
        .vector_narrow_split::<m![Kh / 8, Kh / 4 % 2], m![Kh % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &r, ())
        .vector_fp_zip(FpBinaryOp::MulF(FpMulAlu::Mul0))
        .vector_widen_concat::<m![Kh / 8], m![Kh % 8]>()
        .vector_final()
        .cast::<bf16, m![Kh % 8 # 16]>()
        .commit_trim::<m![Kh % 8]>()
        .commit();
    device
        .main
        .begin_interleaved::<Pi, _, _, _, _, _>(n.view(), xd.view())
        .fetch::<m![Kh / 128, Pi, Kh % 128 / 8], m![Kh % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Kh / 128, Pi, Kh % 128 / 8], m![Kh % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Pi, m![Kh / 8]>()
        .vector_clip_zip(ClipBinaryOpF32::Add)
        .vector_final()
        .cast::<bf16, m![Kh % 8 # 16]>()
        .commit_trim::<m![Kh % 8]>()
        .commit()
}

/// Identical post-FFN arithmetic with an already-resident BF16 input.
pub(crate) fn post_ffn_from_dm<T: AxisName, const HALF: usize>(
    device: &mut Device,
    yd: &Row<bf16, T, HALF, m![Kh]>,
    gamma: &Row<bf16, T, HALF, m![Kh]>,
    s: &VrfTensor<f32, Chip, TCl<T, HALF>, TSl<T, HALF>, m![1 # 8]>,
    xd: &Row<bf16, T, HALF, m![Kh]>,
) -> Row<bf16, T, HALF, m![Kh]> {
    let r = rms::<T, HALF>(device, yd);
    let n: Row<bf16, T, HALF, m![Kh]> = device
        .main
        .begin_interleaved::<Pi, _, _, _, _, _>(yd.view(), gamma.view())
        .fetch::<m![Kh / 128, Pi, Kh % 128 / 8], m![Kh % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Kh / 128, Pi, Kh % 128 / 8], m![Kh % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Pi, m![Kh / 8]>()
        .vector_narrow_split::<m![Kh / 8, Kh / 4 % 2], m![Kh % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &r, ())
        .vector_fp_zip(FpBinaryOp::MulF(FpMulAlu::Mul0))
        .vector_widen_concat::<m![Kh / 8], m![Kh % 8]>()
        .vector_final()
        .cast::<bf16, m![Kh % 8 # 16]>()
        .commit_trim::<m![Kh % 8]>()
        .commit();
    let v: Row<bf16, T, HALF, m![Kh]> = device
        .main
        .begin_interleaved::<Pi, _, _, _, _, _>(n.view(), xd.view())
        .fetch::<m![Kh / 128, Pi, Kh % 128 / 8], m![Kh % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Kh / 128, Pi, Kh % 128 / 8], m![Kh % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Pi, m![Kh / 8]>()
        .vector_clip_zip(ClipBinaryOpF32::Add)
        .vector_final()
        .cast::<bf16, m![Kh % 8 # 16]>()
        .commit_trim::<m![Kh % 8]>()
        .commit();
    let out: Row<bf16, T, HALF, m![Kh]> = device
        .main
        .begin(v.view())
        .fetch::<m![Kh / 16], m![Kh % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Kh / 8], m![Kh % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Kh / 4], m![Kh % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), s)
        .vector_widen_concat::<m![Kh / 8], m![Kh % 8]>()
        .vector_final()
        .cast::<bf16, m![Kh % 8 # 16]>()
        .commit_trim::<m![Kh % 8]>()
        .commit();
    out
}
