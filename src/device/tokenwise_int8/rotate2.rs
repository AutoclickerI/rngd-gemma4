//! The FFN fast path's per-token input phase: the pre-FFN norm, `rotate`'s H60 (x) H64 rotation
//! (bit for bit: `rotate::rotate`, the two-term split) and its per-row INT8.
//!
//! Differences from `rotate::norm_rotate_quantize_pass` (none in the arithmetic):
//! - the Hadamard constants and the signed gamma arrive as DM copies built by an all-gather (each
//!   slice reads a sixteenth of the factor, a sixteen-slice `Broadcast1` completes it), loaded once
//!   per stage for both rotations; the replicated 4 MB DMA per factor (about 5.5k cycles each of
//!   the single DMA queue) sat on the stage's DMA-bound start;
//! - the gamma's DM element order is the all-gather's (`DG`), read by this module's normalize.
use furiosa_opt_std::prelude::*;

use super::rotate::{self, H60Hbm, H60Trf, H64Trf, Ka, Kb, Kh, Oa, Ob, Rc, RotPad, Sgi, Sl8, TCl, TSl};
use crate::Chip;
use crate::axes::{Dummy2, H};
use super::gemm::Pi;

type Row<D, T, const HALF: usize, E> = DmTensor<D, Chip, TCl<T, HALF>, TSl<T, HALF>, E>;

const MAGIC: f32 = 12_582_912.0;
const MAGIC_NEG_I32: i32 = -12_582_912;

axes![Kq = 4096];

axes![X16 = 16];

/// `h64` in DM, element order `[Ob % 4, Kb / 16, Ob / 4, Kb % 16]` (the all-gather's order).
pub(crate) type D64 = m![Ob % 4, Kb / 16, Ob / 4, Kb % 16];
/// `h60` in DM, element order `[Ka / 4, Oa / 32, Oa % 8, Ka % 4, Oa / 8 % 4]`.
pub(crate) type D60 = m![Ka / 4, Oa / 32, Oa % 8, Ka % 4, Oa / 8 % 4];

/// The Hadamard factors in every slice's DM, `(h64, h60)`. A tuple, not a struct: a struct field
/// typed on `TSl` hits an internal compiler error in the device translator.
pub(crate) type DmConsts<T, const HALF: usize> = (Row<bf16, T, HALF, D64>, Row<bf16, T, HALF, D60>);

/// The TRF form of the factors (`rotate::consts`'s result).
pub(crate) type TrfConsts<T, const HALF: usize> = (H64Trf<T, HALF>, H60Trf<T, HALF>, H60Trf<T, HALF>);

/// `h64` and `h60` into every slice's DM, as an all-gather: each slice reads one sixteenth of a
/// factor (512 or 480 bytes, slice `16 g + j` reads piece `j` of HBM copy `g`), then a
/// `Broadcast1` over sixteen-slice rings hands every slice all sixteen pieces (about 256 cycles
/// of the Main context per factor). A replicated DMA of a whole factor to all 512 slices costs
/// about 5.5k cycles of the single DMA queue; these two loads sit on the stage's first
/// critical path, where the DMA queue is the bottleneck and Main is idle.
pub(crate) fn dm_consts<T: AxisName, const HALF: usize>(
    device: &mut Device,
    h64: &HbmTensor<bf16, Chip, m![Rc, Ob, Kb]>,
    h60: &HbmTensor<bf16, Chip, m![Rc, { H60Hbm }]>,
) -> DmConsts<T, HALF> {
    let p64: DmTensor<bf16, Chip, m![Dummy2], m![Rc, Ob / 4], m![Ob % 4, Kb]> = h64.to_dm(&mut device.tdma);
    let p64: DmTensor<bf16, Chip, TCl<T, HALF>, m![Rc, Ob / 4], m![Ob % 4, Kb]> = unsafe { p64.reshape() };
    let d64: DmTensor<bf16, Chip, TCl<T, HALF>, m![Rc, X16], D64> = device
        .main
        .begin(p64.view())
        .fetch::<m![Ob % 4, Kb / 16], m![Kb % 16]>()
        .switch::<m![Rc, X16], m![Ob % 4, Kb / 16, Ob / 4]>(SwitchConfig::Broadcast1 { slice1: 16, slice0: 1 })
        .collect::<m![Ob % 4, Kb / 16, Ob / 4], m![Kb % 16]>()
        .commit_trim::<m![Kb % 16]>()
        .commit();
    let p60: DmTensor<bf16, Chip, m![Dummy2], m![Rc, Oa / 32, Oa % 8], m![Ka, Oa / 8 % 4]> = h60.to_dm(&mut device.tdma);
    let p60: DmTensor<bf16, Chip, TCl<T, HALF>, m![Rc, Oa / 32, Oa % 8], m![Ka, Oa / 8 % 4]> = unsafe { p60.reshape() };
    let d60: DmTensor<bf16, Chip, TCl<T, HALF>, m![Rc, X16], D60> = device
        .main
        .begin(p60.view())
        .fetch::<m![Ka / 4], m![Ka % 4, Oa / 8 % 4]>()
        .switch::<m![Rc, X16], m![Ka / 4, Oa / 32, Oa % 8]>(SwitchConfig::Broadcast1 { slice1: 16, slice0: 1 })
        .collect::<m![Ka / 4, Oa / 32, Oa % 8], m![Ka % 4, Oa / 8 % 4]>()
        .commit_trim::<m![Ka % 4, Oa / 8 % 4]>()
        .commit();
    (unsafe { d64.reshape() }, unsafe { d60.reshape() })
}

/// The factors from DM into the TRF (`rotate::consts` without its DMA).
pub(crate) fn trf_consts<T: AxisName, const HALF: usize>(device: &mut Device, c: &DmConsts<T, HALF>) -> TrfConsts<T, HALF> {
    let t64 = device
        .sub
        .begin(c.0.view())
        .fetch::<m![Ob % 8, Ob / 8, Kb / 16], m![Kb % 16]>()
        .collect::<m![Ob % 8, Ob / 8, Kb / 16], m![Kb % 16]>()
        .to_trf();
    let h0 = h60_half::<T, HALF>(device, &c.1, 0);
    let h1 = h60_half::<T, HALF>(device, &c.1, 1);
    (t64, h0, h1)
}

fn h60_half<T: AxisName, const HALF: usize>(device: &mut Device, d60: &Row<bf16, T, HALF, D60>, i: usize) -> H60Trf<T, HALF> {
    device
        .sub
        .begin(d60.view().tile::<m![Oa / 32], 1, m![Ka / 4, Oa / 32 = 1 # 2, Oa % 8, Ka % 4, Oa / 8 % 4]>(i))
        .fetch::<m![Oa % 8, Ka / 4], m![Ka % 4, Oa / 8 % 4]>()
        .collect::<m![Oa % 8, Ka / 4], m![Ka % 4, Oa / 8 % 4]>()
        .to_trf()
}

/// `normalize` with the gamma paired with the stream (`post2`'s form, `begin_interleaved` and the vector
/// engine's pair mode) instead of a 7.5 KiB VRF operand per row half, each of which waited for a Sub-context VRF
/// store before its Main pass: `h = bf16(x / rms * g)`, the same operations in the same order (`DivF` by the
/// row's rms, then a multiply by the gamma), so the bits are `normalize`'s. `g` in the natural `Kh` order
/// (`gamma_nat_ag2`).
pub(crate) fn normalize_pair<T: AxisName, const HALF: usize>(device: &mut Device, x: &Row<bf16, T, HALF, m![Kh]>, g: &Row<bf16, T, HALF, m![Kh]>) -> Row<bf16, T, HALF, m![Kh]> {
    let ms: Row<f32, T, HALF, m![1 # 8]> = device
        .main
        .begin(x.view())
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
        .vector_clip(ClipBinaryOpF32::Add, crate::EPS)
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
    let rms: VrfTensor<f32, Chip, TCl<T, HALF>, TSl<T, HALF>, m![1 # 8]> =
        device.sub.begin(rms.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    device
        .main
        .begin_interleaved::<Pi, _, _, _, _, _>(x.view(), g.view())
        .fetch::<m![Kh / 128, Pi, Kh % 128 / 8], m![Kh % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Kh / 128, Pi, Kh % 128 / 8], m![Kh % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Pi, m![Kh / 8]>()
        .vector_narrow_split::<m![Kh / 8, Kh / 4 % 2], m![Kh % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms, ())
        .vector_fp_zip(FpBinaryOp::MulF(FpMulAlu::Mul0))
        .vector_widen_concat::<m![Kh / 8], m![Kh % 8]>()
        .vector_final()
        .cast::<bf16, m![Kh % 8 # 16]>()
        .commit_trim::<m![Kh % 8]>()
        .commit()
}

// ------------------------------------------------------------- one-term rotation (tolerance)

/// Stage A (`u = H64 x` per 64-block) rounded once to bf16 (the cast's round to nearest even), the
/// whole of `rotate`'s three-term split replaced by its leading term. The rounding error is at most
/// 2^-9 of each element, about a tenth of the per-row INT8 step that follows (amax / 127); the
/// three-term split kept stage B exact for bit equality with the f32 reference, which tolerance
/// does not require.
fn stage_a1<T: AxisName, const HALF: usize>(device: &mut Device, x: &Row<bf16, T, HALF, m![Ka, Kb]>, h64: &H64Trf<T, HALF>) -> Row<bf16, T, HALF, m![Ka, Ob]> {
    device
        .main
        .begin(x.view())
        .fetch::<m![Ka, Kb / 16], m![Kb % 16]>()
        .collect::<m![Ka, Kb / 16], m![Kb % 16]>()
        .contract_outer::<m![Ka, Kb / 32, Ob / 8], m![Kb % 32], _, _, _>(h64)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ka, Ob / 8]>()
        .contract_lane::<m![Ka, Ob / 8], m![Ob % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_final()
        .cast::<bf16, m![Ob % 8 # 16]>()
        .commit_trim::<m![Ob % 8]>()
        .commit()
}

/// Stage B (`H60` over `a`, scaled) on the one-term stage A, output rows `a'` in half `i`.
fn stage_b1<T: AxisName, const HALF: usize>(
    device: &mut Device,
    u: &Row<bf16, T, HALF, m![Ka, Ob]>,
    h60: &H60Trf<T, HALF>,
    scale: f32,
    i: usize,
    y: &mut Row<bf16, T, HALF, m![Ob / 32, Oa, Ob % 32]>,
) {
    device
        .main
        .begin(u.view())
        .fetch::<m![Ob / 32, Ka, Ob % 32 / 16], m![Ob % 16]>()
        .collect::<m![Ob / 32, Ka, Ob % 32 / 16], m![Ob % 16]>()
        .contract_outer::<m![Ob / 32, Ka, Oa / 8 % 4], m![Ob % 32], _, _, _>(h60)
        .contract_packet::<m![Ob % 32]>()
        .contract_time::<m![Ob / 32, Oa / 8 % 4]>()
        .contract_lane::<m![Ob / 32, Oa / 8 % 4, Oa % 8, Ob % 32 / 8], m![Ob % 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ob / 32, Oa / 8 % 4, Oa % 8, Ob % 32 / 4], m![Ob % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scale)
        .vector_widen_concat::<m![Ob / 32, Oa / 8 % 4, Oa % 8, Ob % 32 / 8], m![Ob % 8]>()
        .vector_final()
        .cast::<bf16, m![Ob % 8 # 16]>()
        .commit_trim::<m![Ob % 8]>()
        .commit_view(y.view_mut().tile::<m![Oa / 32], 1, m![Ob / 32, Oa / 32 = 1 #{!} 2, Oa % 32, Ob % 32]>(i));
}

/// `rotate::rotate` with the one-term stage A: three Main passes instead of seven, a third of
/// stage B's contraction.
pub(crate) fn rotate_j1<T: AxisName, const HALF: usize>(device: &mut Device, x: &Row<bf16, T, HALF, m![Ka, Kb]>, c: &TrfConsts<T, HALF>, scale: f32) -> Row<bf16, T, HALF, RotPad> {
    let u = stage_a1::<T, HALF>(device, x, &c.0);
    let mut y: Row<bf16, T, HALF, m![Ob / 32, Oa, Ob % 32]> = DmTensor::new();
    stage_b1::<T, HALF>(device, &u, &c.1, scale, 0, &mut y);
    stage_b1::<T, HALF>(device, &u, &c.2, scale, 1, &mut y);
    unsafe { y.reshape() }
}

/// A replicated `[Rc, H]` row constant (the norm gammas) as an all-gather with contiguous
/// per-slice reads (slice `16 g + j` reads the 480-byte run `H / 240 = j` of copy `g`, as
/// `gamma_ag`), the switch's `[flit, source]` stream committed into the natural `Kh` order.
pub(crate) fn gamma_nat_ag2<T: AxisName, const HALF: usize>(device: &mut Device, g: &HbmTensor<bf16, Chip, m![Rc, H]>) -> Row<bf16, T, HALF, m![Kh]> {
    let p: DmTensor<bf16, Chip, m![Dummy2], m![Rc, H / 240], m![H % 240]> = g.to_dm(&mut device.tdma);
    let p: DmTensor<bf16, Chip, TCl<T, HALF>, m![Rc, Kh / 240], m![Kh % 240]> = unsafe { p.reshape() };
    let d: DmTensor<bf16, Chip, TCl<T, HALF>, m![Rc, X16], m![Kh]> = device
        .main
        .begin(p.view())
        .fetch::<m![Kh % 240 / 16], m![Kh % 16]>()
        .switch::<m![Rc, X16], m![Kh % 240 / 16, Kh / 240]>(SwitchConfig::Broadcast1 { slice1: 16, slice0: 1 })
        .collect::<m![Kh % 240 / 16, Kh / 240], m![Kh % 16]>()
        .commit_trim::<m![Kh % 16]>()
        .commit();
    unsafe { d.reshape() }
}

/// Quantize natural-order rows to INT8 using the startup smoothing factors.
/// `s = max(amax / 127, 1e-30)` and `q = clamp(rne(y / s), -127, 127)` (`row_scale` and `quantize` on a natural row
/// instead of the rotated one). One token per slice, the row `y [Q]` in DM.
pub(crate) fn row_quantize_i8<T: AxisName, const HALF: usize, Q: AxisName>(
    device: &mut Device,
    y: &Row<bf16, T, HALF, m![Q]>,
) -> (Row<f32, T, HALF, m![1 # 8]>, Row<i8, T, HALF, m![Q]>) {
    let s: Row<f32, T, HALF, m![1 # 8]> = device
        .main
        .begin(y.view())
        .fetch::<m![Q / 16], m![Q % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Q / 8], m![Q % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, i32::MAX)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![Q / 4], m![Q % 4]>()
        .vector_intra_slice_reduce::<Q, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_fp_div(127f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1e-30f32)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let sv: VrfTensor<f32, Chip, TCl<T, HALF>, TSl<T, HALF>, m![1 # 8]> =
        device.sub.begin(s.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let q: Row<i8, T, HALF, m![Q]> = device
        .main
        .begin(y.view())
        .fetch::<m![Q / 16], m![Q % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Q / 8], m![Q % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Q / 4], m![Q % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &sv)
        .vector_fp_binary(FpBinaryOp::AddF, MAGIC)
        .vector_widen_concat::<m![Q / 8], m![Q % 8]>()
        .vector_fp_to_fxp(31)
        .vector_clip(ClipBinaryOpI32::AddFxp, MAGIC_NEG_I32)
        .vector_clip(ClipBinaryOpI32::Min, 127)
        .vector_clip(ClipBinaryOpI32::Max, -127)
        .vector_final()
        .cast::<i8, m![Q % 8 # 32]>()
        .commit_trim::<m![Q % 8]>()
        .commit();
    (s, q)
}

/// Normalize and quantize one pass of natural-order rows to INT8.
/// `normalize_pair` with the plain gamma (`gamma_nat_ag2`), the codes in the natural `[T, H]` order, the scales by
/// `rotate::store_scales` (`norm_quantize_pass_dm_f8` with INT8 codes).
#[allow(clippy::too_many_arguments)]
pub(crate) fn norm_quantize_pass_dm_i8<T: AxisName, const HALF: usize, const PS: usize, const P: usize, const PI: usize>(
    device: &mut Device,
    xd: &Row<bf16, T, HALF, m![Kh]>,
    g: &Row<bf16, T, HALF, m![Kh]>,
    onehot: &VrfTensor<f32, Chip, TCl<T, HALF>, m![1 # 256], m![Sgi, Sl8]>,
    codes: &mut HbmTensor<i8, Chip, m![T, H]>,
    scales: &mut HbmTensor<f32, Chip, m![T]>,
) {
    let h = normalize_pair::<T, HALF>(device, xd, g);
    let (s, q) = row_quantize_i8::<T, HALF, Kh>(device, &h);
    let q: Row<i8, T, HALF, m![H]> = unsafe { q.reshape() };
    q.view().to_hbm_view(
        &mut device.tdma,
        codes.view_mut().tile::<m![T / { PS }], 1, m![T / { PS } = 1 #{!} { P }, T % { PS }, H]>(PI),
    );
    rotate::store_scales::<T, HALF, m![T / { PS } = 1 #{!} { P }, T % { PS }]>(
        device,
        s,
        onehot,
        scales.view_mut().tile::<m![T / { PS }], 1, m![T / { PS } = 1 #{!} { P }, T % { PS }]>(PI),
    );
}

/// Scale the attention output by the exact inverse power of two and round to BF16.
pub(crate) fn scale_row_pow2<T: AxisName, const HALF: usize, Q: AxisName>(
    device: &mut Device, x: &Row<bf16, T, HALF, m![Q]>, inverse: &Row<bf16, T, HALF, m![Q]>,
) -> Row<bf16, T, HALF, m![Q]> {
    device.main.begin_interleaved::<Pi, _, _, _, _, _>(x.view(), inverse.view())
        .fetch::<m![Q / 8, Pi], m![Q % 8]>().fetch_cast::<f32>()
        .collect::<m![Q / 8, Pi], m![Q % 8]>()
        .vector_init().vector_intra_slice_unzip::<Pi, m![Q / 8]>()
        .vector_narrow_split::<m![Q / 4], m![Q % 4]>()
        .vector_fp_zip(FpBinaryOp::MulF(FpMulAlu::Mul0))
        .vector_widen_concat::<m![Q / 8], m![Q % 8]>().vector_final()
        .cast::<bf16, m![Q % 8 # 16]>().commit_trim::<m![Q % 8]>().commit()
}

// Plain static FP8: BF16 producer boundary, f32 division, saturating E4M3 RNE.
pub(crate) fn row_quantize_static_f8<T: AxisName, const HALF: usize, Q: AxisName>(
    device: &mut Device,
    sigma: &HbmTensor<f32, Chip, m![1 # 8]>,
    y: &Row<bf16, T, HALF, m![Q]>,
) -> (Row<f32, T, HALF, m![1 # 8]>, Row<f8e4m3, T, HALF, m![Q]>) {
    let s: Row<f32, T, HALF, m![1 # 8]> = sigma.to_dm(&mut device.tdma);
    let sv: VrfTensor<f32, Chip, TCl<T, HALF>, TSl<T, HALF>, m![1 # 8]> =
        device.sub.begin(s.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let q: Row<f8e4m3, T, HALF, m![Q]> = device
        .main
        .begin(y.view())
        .fetch::<m![Q / 16], m![Q % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Q / 8], m![Q % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Q / 4], m![Q % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &sv)
        .vector_widen_concat::<m![Q / 8], m![Q % 8]>()
        .vector_clip(ClipBinaryOpF32::Min, 448f32)
        .vector_clip(ClipBinaryOpF32::Max, -448f32)
        .vector_final()
        .cast::<f8e4m3, m![Q % 8 # 32]>()
        .commit_trim::<m![Q % 8]>()
        .commit();
    (s, q)
}

/// `row_quantize_static_f8` with the static sigma row already in DM (the paired RSS O stage loads
/// it once for both passes and its epilogue). Same operations on the same values, so the codes are identical.
pub(crate) fn row_quantize_static_f8_dm<T: AxisName, const HALF: usize, Q: AxisName>(
    device: &mut Device,
    s: &Row<f32, T, HALF, m![1 # 8]>,
    y: &Row<bf16, T, HALF, m![Q]>,
) -> Row<f8e4m3, T, HALF, m![Q]> {
    let sv: VrfTensor<f32, Chip, TCl<T, HALF>, TSl<T, HALF>, m![1 # 8]> =
        device.sub.begin(s.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    device
        .main
        .begin(y.view())
        .fetch::<m![Q / 16], m![Q % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Q / 8], m![Q % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Q / 4], m![Q % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &sv)
        .vector_widen_concat::<m![Q / 8], m![Q % 8]>()
        .vector_clip(ClipBinaryOpF32::Min, 448f32)
        .vector_clip(ClipBinaryOpF32::Max, -448f32)
        .vector_final()
        .cast::<f8e4m3, m![Q % 8 # 32]>()
        .commit_trim::<m![Q % 8]>()
        .commit()
}

/// Normalize and quantize one pass to E4M3 using the static activation scale,
/// no rotation: `normalize_pair` with the plain gamma `g` (natural order, `gamma_nat_ag2`), the codes stored in the
/// natural `[T, H]` order and the scales by `rotate::store_scales`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn norm_quantize_static_pass_dm_f8<T: AxisName, const HALF: usize, const PS: usize, const P: usize, const PI: usize>(
    device: &mut Device,
    sigma: &HbmTensor<f32, Chip, m![1 # 8]>,
    xd: &Row<bf16, T, HALF, m![Kh]>,
    g: &Row<bf16, T, HALF, m![Kh]>,
    onehot: &VrfTensor<f32, Chip, TCl<T, HALF>, m![1 # 256], m![Sgi, Sl8]>,
    codes: &mut HbmTensor<f8e4m3, Chip, m![T, H]>,
    scales: &mut HbmTensor<f32, Chip, m![T]>,
) {
    let h = normalize_pair::<T, HALF>(device, xd, g);
    let (s, q) = row_quantize_static_f8::<T, HALF, Kh>(device, sigma, &h);
    let q: Row<f8e4m3, T, HALF, m![H]> = unsafe { q.reshape() };
    q.view().to_hbm_view(
        &mut device.tdma,
        codes.view_mut().tile::<m![T / { PS }], 1, m![T / { PS } = 1 #{!} { P }, T % { PS }, H]>(PI),
    );
    rotate::store_scales::<T, HALF, m![T / { PS } = 1 #{!} { P }, T % { PS }]>(
        device,
        s,
        onehot,
        scales.view_mut().tile::<m![T / { PS }], 1, m![T / { PS } = 1 #{!} { P }, T % { PS }]>(PI),
    );
}
