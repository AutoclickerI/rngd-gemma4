//! Sliding QKV with one-pass heads and shared rotation constants. The full-weight
//! INT8 projection keeps each DMA row aligned to 256 bytes. Splitting a 3,840-byte
//! row at byte 1,920 would require an unaligned transfer.

use furiosa_opt_std::prelude::*;

use super::heads::{HeadsPasses, HeadsRowsEarlyPasses8};
use super::rotq::RotPasses;
use crate::Chip;
use crate::axes::{Ds, E, H, Hh, Hqs, Ns, Ps, Qs, Rq, Rs};
use crate::device::tokenwise_int8::gemm;
use crate::device::tokenwise_int8::passes::{Int8sBucket, Passes};
use crate::device::tokenwise_int8::rotate2::DmConsts;
use crate::device::tokenwise_int8::qkv::{Nq, QkvCl};
use crate::device::tokenwise_int8::rotate::{H60Hbm, Kb, Kh, Ob, Ra, Rc, Sgi, Sl8, TCl, TSl};

/// Sliding QKV with the head phase split into `PP` passes of `PS2 = 2 HALF` rows.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sliding_qkv_stage<
    T: AxisName + Int8sBucket,
    const G: usize,
    const HALF: usize,
    const PS2: usize,
    PP: Passes + HeadsPasses + RotPasses,
    RING: M,
    XS: M,
    YS: M,
    TT: M,
    TR: M,
    const TS: usize,
    const PH: usize,
    const PS: usize,
    const P: usize,
>(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![T, H]>,
    input_rms_weight_signed: &HbmTensor<bf16, Chip, m![Rc, H]>,
    h64: &HbmTensor<bf16, Chip, m![Rc, Ob, Kb]>,
    h60: &HbmTensor<bf16, Chip, m![Rc, { H60Hbm }]>,
    eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>,
    wq: &HbmTensor<i8, Chip, m![Nq, H]>,
    ws: &HbmTensor<f32, Chip, m![Nq]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    rope_offsets: &HbmTensor<i32, Chip, m![T]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
    q_out: &mut HbmTensor<bf16, Chip, m![T, Qs]>,
    k_out: &mut HbmTensor<bf16, Chip, m![T, Ps]>,
    v_out: &mut HbmTensor<bf16, Chip, m![T, Ps]>,
) {
    let raw = project_v1::<T, HALF, PS2, PP, RING, XS, YS, TT, TR, TS>(device, x, input_rms_weight_signed, h64, h60, eye, wq, ws);
    PP::heads::<T, HALF, PS2>(device, &raw, q_rms_weight, k_rms_weight, rope_offsets, cos, sin, q_out, k_out, v_out);
}

/// Full-weight contraction with a fused
/// epilogue (per-token phase with constants loaded once), stored as `raw [T, Nq]`.
#[allow(clippy::too_many_arguments)]
fn project_v1<T: AxisName + Int8sBucket, const HALF: usize, const PS2: usize, PP: RotPasses + Passes, RING: M, XS: M, YS: M, TT: M, TR: M, const TS: usize>(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![T, H]>,
    input_rms_weight_signed: &HbmTensor<bf16, Chip, m![Rc, H]>,
    h64: &HbmTensor<bf16, Chip, m![Rc, Ob, Kb]>,
    h60: &HbmTensor<bf16, Chip, m![Rc, { H60Hbm }]>,
    eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>,
    wq: &HbmTensor<i8, Chip, m![Nq, H]>,
    ws: &HbmTensor<f32, Chip, m![Nq]>,
) -> HbmTensor<bf16, Chip, m![T, Nq]> {
    // int8s: natural codes from 512 rows, rotated below (`Int8sBucket`, one impl per token axis).
    // Keep the scale allocation through the projection for stable device layout.
    let (codes, _scales) = <T as Int8sBucket>::sliding_qkv_codes::<T, HALF, PS2, PP>(device, x, input_rms_weight_signed, h64, h60, eye);
    let xr = gemm::load_codes::<QkvCl, RING, TT, TR, H, 960, m![T, H]>(device, codes.view());
    let wd: DmTensor<i8, Chip, QkvCl, m![{ RING }, { XS }, { YS }], m![Nq % 16, H]> = wq.to_dm(&mut device.tdma);
    let w = gemm::to_trf::<QkvCl, m![{ RING }, { XS }, { YS }], Nq, H, 30, 960, 1920, m![Nq % 16, H]>(device, &wd);
    let wsv = gemm::col_scales::<QkvCl, m![{ RING }, { XS }, { YS }], Nq, m![Nq]>(device, ws.view());
    let y = gemm::contract_ws::<QkvCl, RING, XS, YS, TT, TR, Nq, H, TS, 30, 960, 1920>(device, &xr, &w, &wsv);
    let y: DmTensor<bf16, Chip, QkvCl, m![Nq / 128 % 32, Nq / 16 % 8], m![T, Nq % 16]> = unsafe { y.reshape() };
    let y: DmTensor<bf16, Chip, QkvCl, m![Nq / 128 % 32, T % 8], m![T / 8, Nq / 16 % 8, Nq % 16]> = device
        .main
        .begin(y.view())
        .fetch::<m![T / 8, T % 8], m![Nq % 16]>()
        .switch::<m![Nq / 128 % 32, T % 8], m![T / 8, Nq / 16 % 8]>(SwitchConfig::InterTranspose {
            slice1: 8,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![T / 8, Nq / 16 % 8], m![Nq % 16]>()
        .commit_trim::<m![Nq % 16]>()
        .commit();
    let mut raw = HbmTensor::<bf16, Chip, m![T, Nq]>::new();
    y.view().to_hbm_view(&mut device.tdma, raw.view_mut());
    raw
}

// All-gather the Hadamard factors, gamma and row scales from one copy per group.
// Issue the weight transfer first so it overlaps the per-token arithmetic.

/// The projection on codes and row scales with the weight already in DM (`project_codes` with the
/// all-gathered row scales).
#[cfg(feature = "compile-carried")]
fn project_codes_ag<T: AxisName, RING: M, XS: M, YS: M, TT: M, TR: M, const TS: usize>(
    device: &mut Device,
    codes: HbmTensor<i8, Chip, m![T, Ob / 32, Ra, Ob % 32]>,
    scales: &HbmTensor<f32, Chip, m![T]>,
    wd: &DmTensor<i8, Chip, QkvCl, m![{ RING }, { XS }, { YS }], m![Nq % 16, H]>,
    ws: &HbmTensor<f32, Chip, m![Nq]>,
) -> HbmTensor<bf16, Chip, m![T, Nq]> {
    // Main fills the TRF without a separate Sub transfer.
    let w = gemm::to_trf_main::<QkvCl, m![{ RING }, { XS }, { YS }], Nq, H, 30, 960, 1920, m![Nq % 16, H]>(device, wd);
    let codes: HbmTensor<i8, Chip, m![T, H]> = unsafe { codes.reshape() };
    let xr = gemm::load_codes::<QkvCl, RING, TT, TR, H, 960, m![T, H]>(device, codes.view());
    let _ = scales;
    let wsv = gemm::col_scales::<QkvCl, m![{ RING }, { XS }, { YS }], Nq, m![Nq]>(device, ws.view());
    let y = gemm::contract_ws::<QkvCl, RING, XS, YS, TT, TR, Nq, H, TS, 30, 960, 1920>(device, &xr, &w, &wsv);
    let y: DmTensor<bf16, Chip, QkvCl, m![Nq / 128 % 32, Nq / 16 % 8], m![T, Nq % 16]> = unsafe { y.reshape() };
    let y: DmTensor<bf16, Chip, QkvCl, m![Nq / 128 % 32, T % 8], m![T / 8, Nq / 16 % 8, Nq % 16]> = device
        .main
        .begin(y.view())
        .fetch::<m![T / 8, T % 8], m![Nq % 16]>()
        .switch::<m![Nq / 128 % 32, T % 8], m![T / 8, Nq / 16 % 8]>(SwitchConfig::InterTranspose {
            slice1: 8,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![T / 8, Nq / 16 % 8], m![Nq % 16]>()
        .commit_trim::<m![Nq % 16]>()
        .commit();
    let mut raw = HbmTensor::<bf16, Chip, m![T, Nq]>::new();
    y.view().to_hbm_view(&mut device.tdma, raw.view_mut());
    raw
}

// Natural-order INT8 QKV.
#[cfg(feature = "compile-carried")]
#[path = "qkv_int8s.rs"]
pub(crate) mod int8s;

#[path = "qkv_int8s_xchg.rs"]
pub(crate) mod int8s_xchg;
#[path = "qkv_fp8_xchg.rs"]
pub(crate) mod fp8_xchg;

// Same K reduction, 64-token by four K-quarter input slices for decode-only M128.

// Small geometry, explicit Tok512 quantization class; host routing remains unchanged.
pub(crate) fn sliding_qkv_stage_class512<
    T: AxisName + Int8sBucket,
    const G: usize,
    const HALF: usize,
    const PS2: usize,
    PP: Passes + HeadsPasses + RotPasses,
    RING: M,
    XS: M,
    YS: M,
    TT: M,
    TR: M,
    const TS: usize,
    const PH: usize,
    const PS: usize,
    const P: usize,
>(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![T, H]>,
    input_rms_weight_signed: &HbmTensor<bf16, Chip, m![Rc, H]>,
    h64: &HbmTensor<bf16, Chip, m![Rc, Ob, Kb]>,
    h60: &HbmTensor<bf16, Chip, m![Rc, { H60Hbm }]>,
    eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>,
    wq: &HbmTensor<i8, Chip, m![Nq, H]>,
    ws: &HbmTensor<f32, Chip, m![Nq]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    rope_offsets: &HbmTensor<i32, Chip, m![T]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
    q_out: &mut HbmTensor<bf16, Chip, m![T, Qs]>,
    k_out: &mut HbmTensor<bf16, Chip, m![T, Ps]>,
    v_out: &mut HbmTensor<bf16, Chip, m![T, Ps]>,
) {
    let raw = project_v1_class512::<T, HALF, PS2, PP, RING, XS, YS, TT, TR, TS>(device, x, input_rms_weight_signed, h64, h60, eye, wq, ws);
    PP::heads::<T, HALF, PS2>(device, &raw, q_rms_weight, k_rms_weight, rope_offsets, cos, sin, q_out, k_out, v_out);
}

/// Full-weight contraction with a fused
/// epilogue (per-token phase with constants loaded once), stored as `raw [T, Nq]`.
#[allow(clippy::too_many_arguments)]
fn project_v1_class512<T: AxisName + Int8sBucket, const HALF: usize, const PS2: usize, PP: RotPasses + Passes, RING: M, XS: M, YS: M, TT: M, TR: M, const TS: usize>(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![T, H]>,
    input_rms_weight_signed: &HbmTensor<bf16, Chip, m![Rc, H]>,
    h64: &HbmTensor<bf16, Chip, m![Rc, Ob, Kb]>,
    h60: &HbmTensor<bf16, Chip, m![Rc, { H60Hbm }]>,
    eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>,
    wq: &HbmTensor<i8, Chip, m![Nq, H]>,
    ws: &HbmTensor<f32, Chip, m![Nq]>,
) -> HbmTensor<bf16, Chip, m![T, Nq]> {
    // int8s: natural codes from 512 rows, rotated below (`Int8sBucket`, one impl per token axis).
    // Keep the scale allocation through the projection for stable device layout.
    let (codes, _scales) = <crate::axes::Tok512 as Int8sBucket>::sliding_qkv_codes::<T, HALF, PS2, PP>(device, x, input_rms_weight_signed, h64, h60, eye);
    let xr = gemm::load_codes::<QkvCl, RING, TT, TR, H, 960, m![T, H]>(device, codes.view());
    let wd: DmTensor<i8, Chip, QkvCl, m![{ RING }, { XS }, { YS }], m![Nq % 16, H]> = wq.to_dm(&mut device.tdma);
    let w = gemm::to_trf::<QkvCl, m![{ RING }, { XS }, { YS }], Nq, H, 30, 960, 1920, m![Nq % 16, H]>(device, &wd);
    let wsv = gemm::col_scales::<QkvCl, m![{ RING }, { XS }, { YS }], Nq, m![Nq]>(device, ws.view());
    let y = gemm::contract_ws::<QkvCl, RING, XS, YS, TT, TR, Nq, H, TS, 30, 960, 1920>(device, &xr, &w, &wsv);
    let y: DmTensor<bf16, Chip, QkvCl, m![Nq / 128 % 32, Nq / 16 % 8], m![T, Nq % 16]> = unsafe { y.reshape() };
    let y: DmTensor<bf16, Chip, QkvCl, m![Nq / 128 % 32, T % 8], m![T / 8, Nq / 16 % 8, Nq % 16]> = device
        .main
        .begin(y.view())
        .fetch::<m![T / 8, T % 8], m![Nq % 16]>()
        .switch::<m![Nq / 128 % 32, T % 8], m![T / 8, Nq / 16 % 8]>(SwitchConfig::InterTranspose {
            slice1: 8,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![T / 8, Nq / 16 % 8], m![Nq % 16]>()
        .commit_trim::<m![Nq % 16]>()
        .commit();
    let mut raw = HbmTensor::<bf16, Chip, m![T, Nq]>::new();
    y.view().to_hbm_view(&mut device.tdma, raw.view_mut());
    raw
}

#[cfg(feature = "compile-carried")]
#[path = "qkv_fp8_xchg_drain.rs"]
pub(crate) mod fp8_xchg_drain;
