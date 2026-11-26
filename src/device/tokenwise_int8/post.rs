//! Per-token phases around the INT8 linears, one token per slice (`T <= 512`; the rotation
//! module's `TCl`/`TSl` layout): the post-attention / post-FFN norm with the residual add (and
//! the layer scalar), and the plain per-row INT8 quantization of an already rotated input (the
//! Sliding O input, whose per-head rotation is folded into V and O).
use furiosa_opt_std::prelude::*;

use super::rotate::{Kh, Rc, TCl, TSl};
use crate::axes::H;
use crate::{Chip, EPS};

axes![Kp = 4096];

type Row<D, T, const HALF: usize, E> = DmTensor<D, Chip, TCl<T, HALF>, TSl<T, HALF>, E>;

const KH_F32: f32 = 3840.0;

/// A `[H]` bf16 vector from its `Rc` HBM copies in each slice.
pub(crate) fn vector<T: AxisName, const HALF: usize, PP: super::passes::Passes>(device: &mut Device, g: &HbmTensor<bf16, Chip, m![Rc, H]>) -> Row<bf16, T, HALF, m![Kh]> {
    super::rotate::gamma::<T, HALF, PP>(device, g)
}

/// `1 / rms` pieces of one row: `rms = sqrt(fold(y^2) / 3840 + eps)` in the VRF.
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
        .vector_fp_div(KH_F32)
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

/// One half (`Kh / 1920 = i`) of `out = bf16(f32(bf16(y / rms * gamma)) + f32(x))`.
fn norm_add_half<T: AxisName, const HALF: usize>(
    device: &mut Device,
    yd: &Row<bf16, T, HALF, m![Kh]>,
    xd: &Row<bf16, T, HALF, m![Kh]>,
    gamma: &Row<bf16, T, HALF, m![Kh]>,
    r: &VrfTensor<f32, Chip, TCl<T, HALF>, TSl<T, HALF>, m![1 # 8]>,
    i: usize,
    out: &mut Row<bf16, T, HALF, m![Kh]>,
) {
        let gv: VrfTensor<f32, Chip, TCl<T, HALF>, TSl<T, HALF>, m![Kh % 1920]> = device
            .sub
            .begin(gamma.view().tile::<m![Kh / 1920], 1, m![Kh / 1920 = 1 # 2, Kh % 1920]>(i))
            .fetch::<m![Kh % 1920 / 16], m![Kh % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![Kh % 1920 / 8], m![Kh % 8]>()
            .to_vrf();
        let n: Row<bf16, T, HALF, m![Kh % 1920]> = device
            .main
            .begin(yd.view().tile::<m![Kh / 1920], 1, m![Kh / 1920 = 1 # 2, Kh % 1920]>(i))
            .fetch::<m![Kh % 1920 / 16], m![Kh % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![Kh % 1920 / 8], m![Kh % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Kh % 1920 / 4], m![Kh % 4]>()
            .vector_fp_binary(FpBinaryOp::DivF, r)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gv)
            .vector_widen_concat::<m![Kh % 1920 / 8], m![Kh % 8]>()
            .vector_final()
            .cast::<bf16, m![Kh % 8 # 16]>()
            .commit_trim::<m![Kh % 8]>()
            .commit();
        let xv: VrfTensor<f32, Chip, TCl<T, HALF>, TSl<T, HALF>, m![Kh % 1920]> = device
            .sub
            .begin(xd.view().tile::<m![Kh / 1920], 1, m![Kh / 1920 = 1 # 2, Kh % 1920]>(i))
            .fetch::<m![Kh % 1920 / 16], m![Kh % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![Kh % 1920 / 8], m![Kh % 8]>()
            .to_vrf();
        device
            .main
            .begin(n.view())
            .fetch::<m![Kh % 1920 / 16], m![Kh % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![Kh % 1920 / 8], m![Kh % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Add, &xv)
            .vector_final()
            .cast::<bf16, m![Kh % 8 # 16]>()
            .commit_trim::<m![Kh % 8]>()
            .commit_view(out.view_mut().tile::<m![Kh / 1920], 1, m![Kh / 1920 = 1 #{!} 2, Kh % 1920]>(i));
    }

/// `bf16(f32(bf16(y / rms(y) * gamma)) + f32(x))` of one row per slice (the post-attention or
/// post-FFN norm and the residual add: `shared::rmsnorm::normalize`, `shared::residual::add`).
fn norm_add_row<T: AxisName, const HALF: usize>(
    device: &mut Device,
    yd: &Row<bf16, T, HALF, m![Kh]>,
    xd: &Row<bf16, T, HALF, m![Kh]>,
    gamma: &Row<bf16, T, HALF, m![Kh]>,
) -> Row<bf16, T, HALF, m![Kh]> {
    let r = rms::<T, HALF>(device, yd);
    let mut out: Row<bf16, T, HALF, m![Kh]> = DmTensor::new();
    norm_add_half::<T, HALF>(device, yd, xd, gamma, &r, 0, &mut out);
    norm_add_half::<T, HALF>(device, yd, xd, gamma, &r, 1, &mut out);
    out
}

/// One pass (`PI`) of `P` passes of `PS = 2 HALF` rows (see `passes::Passes`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn post_attention_pass<T: AxisName, const HALF: usize, const PS: usize, const P: usize, const PI: usize>(
    device: &mut Device,
    y: &HbmTensor<bf16, Chip, m![T, H]>,
    gamma: &Row<bf16, T, HALF, m![Kh]>,
    x: &mut HbmTensor<bf16, Chip, m![T, H]>,
) {
    let (yd, xd) = load_pair::<T, HALF, PS, P, PI>(device, y, x);
    let out = norm_add_row::<T, HALF>(device, &yd, &xd, gamma);
    store_row::<T, HALF, PS, P, PI>(device, out, x);
}

/// `post_attention_pass` with the result kept in DM for the next stage (the DM hand-off): the
/// same loads and arithmetic, no store (the FFN that consumes the rows stores the final x).
pub(crate) fn post_attention_pass_dm<T: AxisName, const HALF: usize, const PS: usize, const P: usize, const PI: usize>(
    device: &mut Device,
    y: &HbmTensor<bf16, Chip, m![T, H]>,
    gamma: &Row<bf16, T, HALF, m![Kh]>,
    x: &HbmTensor<bf16, Chip, m![T, H]>,
) -> Row<bf16, T, HALF, m![Kh]> {
    let (yd, xd) = load_pair::<T, HALF, PS, P, PI>(device, y, x);
    norm_add_row::<T, HALF>(device, &yd, &xd, gamma)
}

/// One pass (`PI`) of `P` passes of `PS = 2 HALF` rows (see `passes::Passes`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn post_ffn_pass<T: AxisName, const HALF: usize, const PS: usize, const P: usize, const PI: usize>(
    device: &mut Device,
    y: &HbmTensor<bf16, Chip, m![T, H]>,
    gamma: &Row<bf16, T, HALF, m![Kh]>,
    s: &VrfTensor<f32, Chip, TCl<T, HALF>, TSl<T, HALF>, m![1 # 8]>,
    x: &mut HbmTensor<bf16, Chip, m![T, H]>,
) {
    let (yd, xd) = load_pair::<T, HALF, PS, P, PI>(device, y, x);
    let added = norm_add_row::<T, HALF>(device, &yd, &xd, gamma);
    let out = scale_row::<T, HALF>(device, &added, &s);
    store_row::<T, HALF, PS, P, PI>(device, out, x);
}

/// `bf16(f32(x) * s)` of one row per slice (`scale_by_layer_gate`).
fn scale_row<T: AxisName, const HALF: usize>(
    device: &mut Device,
    added: &Row<bf16, T, HALF, m![Kh]>,
    s: &VrfTensor<f32, Chip, TCl<T, HALF>, TSl<T, HALF>, m![1 # 8]>,
) -> Row<bf16, T, HALF, m![Kh]> {
    device
        .main
        .begin(added.view())
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
        .commit()
}

/// Pass `p` of rows `y` and `x` (`[T, H]`), one per slice.
#[allow(clippy::type_complexity)]
fn load_pair<T: AxisName, const HALF: usize, const PS: usize, const P: usize, const PI: usize>(
    device: &mut Device,
    y: &HbmTensor<bf16, Chip, m![T, H]>,
    x: &HbmTensor<bf16, Chip, m![T, H]>,
) -> (Row<bf16, T, HALF, m![Kh]>, Row<bf16, T, HALF, m![Kh]>) {
    let yd: Row<bf16, T, HALF, m![H]> = y
        .view()
        .tile::<m![T / { PS }], 1, m![T / { PS } = 1 # { P }, T % { PS }, H]>(PI)
        .to_dm(&mut device.tdma);
    let xd: Row<bf16, T, HALF, m![H]> = x
        .view()
        .tile::<m![T / { PS }], 1, m![T / { PS } = 1 # { P }, T % { PS }, H]>(PI)
        .to_dm(&mut device.tdma);
    (unsafe { yd.reshape() }, unsafe { xd.reshape() })
}

/// Store pass `p` of the residual rows.
pub(crate) fn store_row<T: AxisName, const HALF: usize, const PS: usize, const P: usize, const PI: usize>(
    device: &mut Device,
    out: Row<bf16, T, HALF, m![Kh]>,
    x: &mut HbmTensor<bf16, Chip, m![T, H]>,
) {
    let out: Row<bf16, T, HALF, m![H]> = unsafe { out.reshape() };
    out.view().to_hbm_view(
        &mut device.tdma,
        x.view_mut().tile::<m![T / { PS }], 1, m![T / { PS } = 1 #{!} { P }, T % { PS }, H]>(PI),
    );
}

/// The layer scalar `[1 # 8]` as an f32 VRF operand in every slice.
pub(crate) fn scalar_vrf<T: AxisName, const HALF: usize>(
    device: &mut Device,
    s: &HbmTensor<bf16, Chip, m![1 # 8]>,
) -> VrfTensor<f32, Chip, TCl<T, HALF>, TSl<T, HALF>, m![1 # 8]> {
    let s: Row<bf16, T, HALF, m![1 # 8]> = s.to_dm(&mut device.tdma);
    device.sub.begin(s.view()).fetch::<m![1], m![1 # 8]>().fetch_cast::<f32>().collect::<m![1], m![1 # 8]>().to_vrf()
}
