//! Feed-forward stages with E4M3 activations and weights.
//!
//! Every linear is E4M3 x E4M3 with f32 accumulation and no rotation:
//! - the input row: the plain pre-FFN norm, then `rotate2::row_quantize_f8` (`s = max(amax / 256, 1e-30)`,
//!   `q = e4m3_rne(y / s)`);
//! - Up/Gate: the host's per-row E4M3 codes of the expanded NVFP4 weight at amax / 256, the epilogue
//!   `bf16((acc * sx) * ws)` with `ws = f32(s_w) * f32(g)` (`contract8`), GeGLU as `gl_geglu` (bf16 act);
//! - the Down input: one scale per token over the whole 15,360-wide act row (both pass pairs, both clusters: an
//!   inter-slice max per cluster, then a cluster swap and a pair max), the codes `e4m3_rne(act / s)` and GL's ring
//!   transpose into the Down pass's HBM codes;
//! - Down: E4M3 rows (per-row amax / 256 of the expanded weight), f32 partials scaled by the token's scale in the
//!   contraction epilogue (GL's place), summed over passes and clusters by GL's tail, times `ws = f32(s_w) * f32(g)`.
//!
//! Pad the 30,720 Up/Gate rows with 2,048 zero rows. This fills the entire weight
//! layout in DM on each pass. An unwritten E4M3 lane could hold NaN, survive multiplication
//! by a zero scale, and enter the cross-slice maximum that computes the Down scale.
use super::*;
axes![Lp = 32768];

/// Activation codes in each cluster as E4M3 (`gemm::XRing`'s layout).
pub(crate) type XRing8<CL, RING, TT, TR, K, const CH: usize> =
    DmTensor<f8e4m3, Chip, CL, m![{ RING }, { TR }, K / { CH } % 4], m![{ TT }, K % { CH }]>;
/// 16 weight rows per slice in the TRF as E4M3 (`gemm::WTrf`'s layout and stream order).
pub(crate) type WTrf8<CL, SL, N, K, const F: usize, const CH: usize, const CH2: usize> =
    TrfTensor<f8e4m3, Chip, CL, SL, m![N % 8], m![N / 8 % 2, K / 32 % { F }, K / { CH2 } % 2, K / { CH } % 2, K % 32]>;
/// Both Down passes' E4M3 codes in HBM (GL's `dcodes` layout).
pub(crate) type DCodes8<T> = HbmTensor<f8e4m3, Chip, m![Kq4 % 2, Kq4 / 2, T, Kd]>;

/// `gemm::to_trf` for E4M3 rows (the Sub fill, layout and stream order of the INT8 rows).
pub(crate) fn to_trf8<CL: M, SL: M, N: AxisName, K: AxisName, const F: usize, const CH: usize, const CH2: usize, WE: M>(
    device: &mut Device,
    w: &DmTensor<f8e4m3, Chip, CL, SL, WE>,
) -> WTrf8<CL, SL, N, K, F, CH, CH2> {
    device
        .sub
        .begin(w.view())
        .fetch::<m![N % 8, N / 8 % 2, K / 32 % { F }, K / { CH2 } % 2, K / { CH } % 2], m![K % 32]>()
        .collect::<m![N % 8, N / 8 % 2, K / 32 % { F }, K / { CH2 } % 2, K / { CH } % 2], m![K % 32]>()
        .to_trf()
}

/// `gemm::to_trf_main` for E4M3 rows (the Main fill of the O, Down and QKV weights).
pub(crate) fn to_trf8_main<CL: M, SL: M, N: AxisName, K: AxisName, const F: usize, const CH: usize, const CH2: usize, WE: M>(
    device: &mut Device,
    w: &DmTensor<f8e4m3, Chip, CL, SL, WE>,
) -> WTrf8<CL, SL, N, K, F, CH, CH2> {
    device
        .main
        .begin(w.view())
        .fetch::<m![N % 8, N / 8 % 2, K / 32 % { F }, K / { CH2 } % 2, K / { CH } % 2], m![K % 32]>()
        .collect::<m![N % 8, N / 8 % 2, K / 32 % { F }, K / { CH2 } % 2, K / { CH } % 2], m![K % 32]>()
        .to_trf()
}

/// `gemm::contract` over E4M3 operands: `y[t, n] = bf16((sum_k x[t, k] w[n, k] in f32) * sx[t] * ws[n])`.
#[allow(clippy::type_complexity)]
pub(crate) fn contract8<
    T: AxisName,
    CL: M,
    RING: M,
    XS: M,
    YS: M,
    TT: M,
    TR: M,
    N: AxisName,
    K: AxisName,
    const TS: usize,
    const F: usize,
    const CH: usize,
    const CH2: usize,
>(
    device: &mut Device,
    x: &XRing8<CL, RING, TT, TR, K, CH>,
    w: &WTrf8<CL, m![{ RING }, { XS }, { YS }], N, K, F, CH, CH2>,
    sx: &VrfTensor<f32, Chip, CL, m![{ RING }, { XS }, { YS }], m![T]>,
    ws: &VrfTensor<f32, Chip, CL, m![{ RING }, { XS }, { YS }], m![N / 8 % 2, N % 8]>,
) -> DmTensor<bf16, Chip, CL, m![{ RING }, { XS }, { YS }], m![{ TT }, { TR }, N / 8 % 2, N % 8]> {
    device
        .main
        .begin(x.view())
        .fetch::<m![{ TT }, K / 32 % { F }], m![K % 32]>()
        .switch::<m![{ RING }, { XS }, { YS }], m![{ TT }, { TR }, K / 32 % { F }, K / { CH } % 4]>(SwitchConfig::Broadcast01 {
            slice1: TS,
            slice0: 4,
            time0: F,
        })
        .collect::<m![{ TT }, { TR }, K / 32 % { F }, K / { CH } % 4], m![K % 32]>()
        .contract_outer::<m![{ TT }, { TR }, K / 32 % { F }, K / { CH2 } % 2, N / 8 % 2], m![K / { CH } % 2, K % 32], _, _, _>(w)
        .contract_packet::<m![1]>()
        .contract_time::<m![{ TT }, { TR }, N / 8 % 2]>()
        .contract_lane::<m![{ TT }, { TR }, N / 8 % 2], m![N % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{ TT }, { TR }, N / 8 % 2, N / 4 % 2], m![N % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), sx)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), ws)
        .vector_widen_concat::<m![{ TT }, { TR }, N / 8 % 2], m![N % 8]>()
        .vector_final()
        .cast::<bf16, m![N % 8 # 16]>()
        .commit_trim::<m![N % 8]>()
        .commit()
}

/// `gemm::contract_i32` over E4M3 operands: the f32 partial sums committed as they leave the lanes (no epilogue).
#[allow(clippy::type_complexity)]
pub(crate) fn contract8_f32<
    CL: M,
    RING: M,
    XS: M,
    YS: M,
    TT: M,
    TR: M,
    N: AxisName,
    K: AxisName,
    const TS: usize,
    const F: usize,
    const CH: usize,
    const CH2: usize,
>(
    device: &mut Device,
    x: &XRing8<CL, RING, TT, TR, K, CH>,
    w: &WTrf8<CL, m![{ RING }, { XS }, { YS }], N, K, F, CH, CH2>,
) -> DmTensor<f32, Chip, CL, m![{ RING }, { XS }, { YS }], m![{ TT }, { TR }, N / 8 % 2, N % 8]> {
    device
        .main
        .begin(x.view())
        .fetch::<m![{ TT }, K / 32 % { F }], m![K % 32]>()
        .switch::<m![{ RING }, { XS }, { YS }], m![{ TT }, { TR }, K / 32 % { F }, K / { CH } % 4]>(SwitchConfig::Broadcast01 {
            slice1: TS,
            slice0: 4,
            time0: F,
        })
        .collect::<m![{ TT }, { TR }, K / 32 % { F }, K / { CH } % 4], m![K % 32]>()
        .contract_outer::<m![{ TT }, { TR }, K / 32 % { F }, K / { CH2 } % 2, N / 8 % 2], m![K / { CH } % 2, K % 32], _, _, _>(w)
        .contract_packet::<m![1]>()
        .contract_time::<m![{ TT }, { TR }, N / 8 % 2]>()
        .contract_lane::<m![{ TT }, { TR }, N / 8 % 2], m![N % 8]>(LaneMode::Interleaved)
        .commit_trim::<m![N % 8]>()
        .commit()
}

/// One Up/Gate pass: the pass's rows of all 32 rings (the padded pair's zero rows included) into the weight DM,
/// the Sub TRF fill, `contract8`, GeGLU into the act tensor (`gl_geglu`).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn up_gate_pass8<T: AxisName, URING: M, UXS: M, UYS: M, TT: M, TR: M, const TS: usize>(
    device: &mut Device,
    xr: &XRing8<UCl, URING, TT, TR, H, 960>,
    sx: &VrfTensor<f32, Chip, UCl, USl<URING, UXS, UYS>, m![T]>,
    wug: &HbmTensor<f8e4m3, Chip, m![Lp, H]>,
    sd: &DmTensor<f32, Chip, UCl, USl<URING, UXS, UYS>, m![Lu / 16 % 4, Lu / 8 % 2, Lu % 8]>,
    p: usize,
    act: &mut DmTensor<bf16, Chip, UCl, USl<URING, UXS, UYS>, m![Lu / 32 % 2, { TT }, { TR }, Lu / 16 % 2, Lu % 8]>,
) {
    let wt = wug.view().tile::<m![Lp / 16 % 4], 1, m![Lp / 64, Lp / 16 % 4 = 1 # 4, Lp % 16, H]>(p);
    let wdm: DmTensor<f8e4m3, Chip, m![Lp / 512 % 2], m![Lp / 1024, Lp / 256 % 2, Lp / 64 % 4], m![Lp % 16, H]> = wt.to_dm(&mut device.tdma);
    // `Lp / 1024 < 30` is `Lu / 1024`; rings 30 and 31 are `Lu`'s padding (`gl_up_scales`' GSl -> USl relabel).
    let wdm: DmTensor<f8e4m3, Chip, UCl, USl<URING, UXS, UYS>, m![Lu % 16, H]> = unsafe { wdm.reshape() };
    let w = to_trf8::<UCl, USl<URING, UXS, UYS>, Lu, H, 30, 960, 1920, m![Lu % 16, H]>(device, &wdm);
    let ws: VrfTensor<f32, Chip, UCl, USl<URING, UXS, UYS>, m![Lu / 8 % 2, Lu % 8]> = device
        .sub
        .begin(sd.view().tile::<m![Lu / 16 % 4], 1, m![Lu / 16 % 4 = 1 # 4, Lu / 8 % 2, Lu % 8]>(p))
        .fetch::<m![Lu / 8 % 2], m![Lu % 8]>()
        .collect::<m![Lu / 8 % 2], m![Lu % 8]>()
        .to_vrf();
    let y = contract8::<T, UCl, URING, UXS, UYS, TT, TR, Lu, H, TS, 30, 960, 1920>(device, xr, &w, sx, &ws);
    gl_geglu::<TT, TR, URING, UXS, UYS>(device, &y, p, act);
}

/// Down input, step 6: pass pair `p`'s codes through GL's sixteen-slice ring transpose into Down pass `p`'s tile of
/// `dcodes`.
pub(crate) fn codes_pair8<T: AxisName>(device: &mut Device, q: &DmTensor<f8e4m3, Chip, UCl, GSl, m![Gpp, T, Kb % 16]>, dcodes: &mut DCodes8<T>, p: usize) {
    let t: DmTensor<f8e4m3, Chip, UCl, GItSl<T>, m![T / 16, Lu / 1024 % 2, Lu / 256 % 2, Kb]> = device
        .main
        .begin(q.view().tile::<m![Gpp], 1, m![Gpp = 1 # 2, T, Kb % 16]>(p))
        .fetch::<m![T / 16, T % 16], m![Kb % 16]>()
        .switch::<GItSl<T>, m![T / 16, Lu / 1024 % 2, Lu / 256 % 2, Kb / 16]>(SwitchConfig::InterTranspose { slice1: 16, slice0: 1, time0: 1 })
        .collect::<m![T / 16, Lu / 1024 % 2, Lu / 256 % 2, Kb / 16], m![Kb % 16 # 32]>()
        .commit_trim::<m![Kb % 16]>()
        .commit();
    let t: DmTensor<f8e4m3, Chip, m![Kq4 / 2], m![Kd / 256 # 16, T % 16], m![T / 16, Kd % 256]> = unsafe { t.reshape() };
    t.view().to_hbm_view(&mut device.tdma, dcodes.view_mut().tile::<m![Kq4 % 2], 1, m![Kq4 % 2 = 1 #{!} 2, Kq4 / 2, T, Kd]>(p));
}

/// The four Up/Gate passes from the input codes and scales into the act tensor (GSl view).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn up_gate8<T: AxisName, URING: M, UXS: M, UYS: M, TT: M, TR: M, const TS: usize>(
    device: &mut Device,
    codes: &HbmTensor<f8e4m3, Chip, m![T, H]>,
    scales: &HbmTensor<f32, Chip, m![T]>,
    wug: &HbmTensor<f8e4m3, Chip, m![Lp, H]>,
    wug_scale: &HbmTensor<f32, Chip, m![Lu]>,
) -> DmTensor<bf16, Chip, UCl, GSl, m![Gpp, T, Kb % 16]> {
    let mut act: DmTensor<bf16, Chip, UCl, USl<URING, UXS, UYS>, m![Lu / 32 % 2, { TT }, { TR }, Lu / 16 % 2, Lu % 8]> = DmTensor::new();
    let sx = row_scales_ag::<T, UCl, USl<URING, UXS, UYS>>(device, scales);
    let xr: XRing8<UCl, URING, TT, TR, H, 960> = codes.view().to_dm(&mut device.tdma);
    let sd = gl_up_scales::<URING, UXS, UYS>(device, wug_scale);
    up_gate_pass8::<T, URING, UXS, UYS, TT, TR, TS>(device, &xr, &sx, wug, &sd, 0, &mut act);
    up_gate_pass8::<T, URING, UXS, UYS, TT, TR, TS>(device, &xr, &sx, wug, &sd, 1, &mut act);
    up_gate_pass8::<T, URING, UXS, UYS, TT, TR, TS>(device, &xr, &sx, wug, &sd, 2, &mut act);
    up_gate_pass8::<T, URING, UXS, UYS, TT, TR, TS>(device, &xr, &sx, wug, &sd, 3, &mut act);
    unsafe { act.reshape() }
}

// ---- F1 (static scales): the FFN of `precision_policy`'s `fp8-s-st` carrier -----
// Prep folds every scale into the producers and the column scales. The pre-FFN gamma carries `1 / (s sx)`, so the norm
// emits the Up/Gate codes (`rotate2::norm_cast_pass_dm_f8`). Up/Gate's `wug_scale` carries `sx` on every row and, on
// the Up rows, `1 / (s_d sx_d)` of the Down input. So the GeGLU's product is already the Down input in code units and
// its epilogue emits E4M3 (`gl_geglu8`). Down's `wd_scale` carries `sx_d`. Compared with `core8` this deletes the row
// scales (store, all-gather, `MulF`), the Down input's amax chain (slice, cluster, cluster swap, lane max), its
// all-slice scale copy and code pass, and Down's per-token epilogue multiply. The ring transpose (`codes_pair8`) and
// the HBM hand-off stay.

pub(crate) fn down_input_static8<T: AxisName>(device: &mut Device, sigma: &HbmTensor<f32, Chip, m![1 # 8]>, act: &DmTensor<bf16, Chip, UCl, GSl, m![Gpp, T, Kb % 16]>, dcodes: &mut DCodes8<T>) {
    let q = act_codes_static::<T>(device, act, sigma);
    codes_pair8::<T>(device, &q, dcodes, 0);
    codes_pair8::<T>(device, &q, dcodes, 1);
}
fn core_static8<const OWN: usize, K: StaticGlTail, T: AxisName, const HALF: usize, const PS: usize, URING: M, UXS: M, UYS: M, DRING: M, DXS: M, DYS: M, TT: M, TR: M, const TS: usize>(
    device: &mut Device,
    sigma_down: &HbmTensor<f32, Chip, m![1 # 8]>,
    codes: &HbmTensor<f8e4m3, Chip, m![T, H]>,
    scales: &HbmTensor<f32, Chip, m![T]>,
    wug: &HbmTensor<f8e4m3, Chip, m![Lp, H]>,
    wug_scale: &HbmTensor<f32, Chip, m![Lu]>,
    wd: &HbmTensor<f8e4m3, Chip, m![Kq4, H, Kd]>,
    wd_scale: &HbmTensor<f32, Chip, m![H]>,
    dcodes: &mut DCodes8<T>,
) -> HbmTensor<bf16, Chip, m![T, H]> {
    let act = up_gate8::<T, URING, UXS, UYS, TT, TR, TS>(device, codes, scales, wug, wug_scale);
    let _ = down_input_static8::<T>(device, sigma_down, &act, dcodes);
    super::rowdown_f8::down::<T,OWN>(device,dcodes,wd,wd_scale,sigma_down)
}
pub(crate) fn ffn_stage_static_dm1_f8<
    T: AxisName,
    const HALF: usize,
    const PS: usize,
    URING: M,
    UXS: M,
    UYS: M,
    DRING: M,
    DXS: M,
    DYS: M,
    TT: M,
    TR: M,
    const TS: usize,
>(
    device: &mut Device,
    sigma: &HbmTensor<f32, Chip, m![1 # 8]>,
    sigma_down: &HbmTensor<f32, Chip, m![1 # 8]>,
    xd: &Row<bf16, T, HALF, m![Kh]>,
    x: &mut HbmTensor<bf16, Chip, m![T, H]>,
    eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>,
    pre_ff_rms_weight: &HbmTensor<bf16, Chip, m![Rc, H]>,
    wug: &HbmTensor<f8e4m3, Chip, m![Lp, H]>,
    wug_scale: &HbmTensor<f32, Chip, m![Lu]>,
    wd: &HbmTensor<f8e4m3, Chip, m![Kq4, H, Kd]>,
    wd_scale: &HbmTensor<f32, Chip, m![H]>,
    post_ff_rms_weight: &HbmTensor<bf16, Chip, m![Rc, H]>,
    layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
) -> Row<bf16, T, HALF, m![Kh]> {
    let g = rotate2::gamma_nat_ag2::<T, HALF>(device, pre_ff_rms_weight);
    let onehot = rotate::onehot::<T, HALF>(device, eye);
    let mut codes = HbmTensor::<f8e4m3, Chip, m![T, H]>::new();
    let mut scales = HbmTensor::<f32, Chip, m![T]>::new();
    rotate2::norm_quantize_static_pass_dm_f8::<T, HALF, PS, 1, 0>(device, sigma, xd, &g, &onehot, &mut codes, &mut scales);
    let mut dcodes = DCodes8::<T>::new();
    let d = core_static8::<HALF, AllReduce, T, HALF, PS, URING, UXS, UYS, DRING, DXS, DYS, TT, TR, TS>(device, sigma_down, &codes, &scales, wug, wug_scale, wd, wd_scale, &mut dcodes);
    let gp = rotate2::gamma_nat_ag2::<T, HALF>(device, post_ff_rms_weight);
    let s = post::scalar_vrf::<T, HALF>(device, layer_scalar);
    let out0 = post2::post_ffn_pass_dm::<T, HALF, PS, 1, 0>(device, &d, &gp, &s, xd);
    let v0: DmTensorView<'_, bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![H]> = unsafe { out0.view().reshape() };
    v0.to_hbm_view(&mut device.tdma, x.view_mut().tile::<m![T / { PS }], 1, m![T / { PS } = 1 #{!} 1, T % { PS }, H]>(0));
    out0
}

trait StaticGlTail {
}

impl StaticGlTail for AllReduce {
}

fn act_codes_static<T: AxisName>(
    device: &mut Device,
    act: &DmTensor<bf16, Chip, UCl, GSl, m![Gpp, T, Kb % 16]>,
    sigma: &HbmTensor<f32, Chip, m![1 # 8]>,
) -> DmTensor<f8e4m3, Chip, UCl, GSl, m![Gpp, T, Kb % 16]> {
    let sd: DmTensor<f32, Chip, UCl, GSl, m![1 # 8]> = sigma.to_dm(&mut device.tdma);
    let sv: VrfTensor<f32, Chip, UCl, GSl, m![1 # 8]> = device.sub.begin(sd.view())
        .fetch::<m![1],m![1 # 8]>().collect::<m![1],m![1 # 8]>().to_vrf();
    device
        .main
        .begin(act.view())
        .fetch::<m![Gpp, T], m![Kb % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gpp, T, Kb / 8 % 2], m![Kb % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gpp, T, Kb / 8 % 2, Kb / 4 % 2], m![Kb % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &sv)
        .vector_widen_concat::<m![Gpp, T, Kb / 8 % 2], m![Kb % 8]>()
        .vector_clip(ClipBinaryOpF32::Min, 448f32)
        .vector_clip(ClipBinaryOpF32::Max, -448f32)
        .vector_final()
        .cast::<f8e4m3, m![Kb % 8 # 32]>()
        .commit_trim::<m![Kb % 8]>()
        .commit()
}

// Keep the paired FFN signatures fixed for the kernel images.
#[path = "ffn_pair.rs"]
pub(crate) mod pair;

#[path = "ffn_cohort.rs"]
mod cohort;

#[path = "up_xchg_f8.rs"]
pub(crate) mod up_xchg;
