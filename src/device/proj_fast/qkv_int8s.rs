//! Natural-order INT8 Sliding QKV. The startup weights and input gamma include the
//! power-of-two smoothing factors. The row quantizer produces natural-order codes
//! for the projection, which applies the per-output-row weight scales.

use crate::device::tokenwise_int8::rotate;
use crate::device::tokenwise_int8::rotate2;
use furiosa_opt_std::prelude::*;

use super::*;

// One 512-row pass.

/// `project_v5_dm1` without the rotation: the weight first, the plain-gamma norm and per-row INT8 of the pass, then
/// `project_codes_ag`.
#[allow(clippy::type_complexity)]
pub(super) fn project_dm1_i8s<T: AxisName, const HALF: usize, const PS2: usize, RING: M, XS: M, YS: M, TT: M, TR: M, const TS: usize>(
    device: &mut Device,
    xd: &DmTensor<bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![Kh]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![Rc, H]>,
    eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>,
    wq: &HbmTensor<i8, Chip, m![Nq, H]>,
    ws: &HbmTensor<f32, Chip, m![Nq]>,
) -> HbmTensor<bf16, Chip, m![T, Nq]> {
    let wd: DmTensor<i8, Chip, QkvCl, m![{ RING }, { XS }, { YS }], m![Nq % 16, H]> = wq.to_dm(&mut device.tdma);
    let mut codes = HbmTensor::<i8, Chip, m![T, H]>::new();
    let mut scales = HbmTensor::<f32, Chip, m![T]>::new();
    let g = rotate2::gamma_nat_ag2::<T, HALF>(device, input_rms_weight);
    let onehot = rotate::onehot::<T, HALF>(device, eye);
    rotate2::norm_quantize_pass_dm_i8::<T, HALF, PS2, 1, 0>(device, xd, &g, &onehot, &mut codes, &mut scales);
    // SAFETY: as `project_dm2_i8s`.
    let codes: HbmTensor<i8, Chip, m![T, Ob / 32, Ra, Ob % 32]> = unsafe { codes.reshape() };
    project_codes_ag::<T, RING, XS, YS, TT, TR, TS>(device, codes, &scales, &wd, ws)
}

/// The no-rotation INT8 QKV stage on DM input rows, one pass (Tok512), the K/V scatters into the FP8 Sliding pools
/// (`sliding_qkv_stage_rows_v5_dm1_8` with `project_dm1_i8s`).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn sliding_qkv_stage_rows_dm1_i8s<
    T: AxisName,
    const HALF: usize,
    const PS2: usize,
    PP: Passes + HeadsRowsEarlyPasses8,
    RING: M,
    XS: M,
    YS: M,
    TT: M,
    TR: M,
    const TS: usize,
>(
    device: &mut Device,
    xd: &DmTensor<bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![Kh]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![Rc, H]>,
    eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>,
    wq: &HbmTensor<i8, Chip, m![Nq, H]>,
    ws: &HbmTensor<f32, Chip, m![Nq]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    rope_offsets: &HbmTensor<i32, Chip, m![T]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
    rows_s: &HbmTensor<i32, Chip, m![T, Hh]>,
    stage_s: &HbmTensor<i32, Chip, m![T]>,
    qin_s: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
    k_n: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
    v_n: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
) {
    let raw = project_dm1_i8s::<T, HALF, PS2, RING, XS, YS, TT, TR, TS>(device, xd, input_rms_weight, eye, wq, ws);
    <PP as HeadsRowsEarlyPasses8>::heads_rows_early8::<T, HALF, PS2>(device, &raw, q_rms_weight, k_rms_weight, rope_offsets, cos, sin, rows_s, stage_s, qin_s, k_n, v_n);
}
