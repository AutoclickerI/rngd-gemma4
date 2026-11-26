//! Fixed FP8 scalar epilogues with fully occupied paired vector passes.
use super::*;

/// slices256c3: `stage2_up_dm_pp` with the all-256-slice Down (rowdown_slices256c2_f8) in place of rowdown_dm8_f8:
/// pair 0's codes (made beside Up/Gate passes 2-3 by `up_xchg::up_pp`) are loaded for both Down groups inside the
/// Up/Gate window, pair 1's per group in the Down. Same codes, K order and (q0+q1)+(q2+q3) f32 tree as rowdown_dm8_f8
/// (exact). The Tok1024 S8N Full resident transition uses this layout.
#[allow(clippy::too_many_arguments)]
pub(crate) fn stage2_up_dm_pp_s256<
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
    xd: (&Row<bf16, T, HALF, m![Kh]>, &Row<bf16, T, HALF, m![Kh]>),
    x: &mut HbmTensor<bf16, Chip, m![T, H]>,
    _eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>,
    pre_ff_rms_weight: &HbmTensor<bf16, Chip, m![Rc, H]>,
    wug: &HbmTensor<f8e4m3, Chip, m![Lp, H]>,
    wug_scale: &HbmTensor<f32, Chip, m![Lu]>,
    wd: &HbmTensor<f8e4m3, Chip, m![Kq4, H, Kd]>,
    wd_scale: &HbmTensor<f32, Chip, m![H]>,
    post_ff_rms_weight: &HbmTensor<bf16, Chip, m![Rc, H]>,
    layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
) -> (Row<bf16, T, HALF, m![Kh]>, Row<bf16, T, HALF, m![Kh]>) {
    let g = rotate2::gamma_nat_ag2::<T, HALF>(device, pre_ff_rms_weight);
    let (dcodes0, dcodes1) = super::up_xchg::up_pp::<super::up_xchg::TwoInputs,T,HALF,PS,2,URING,UXS,UYS,TT,TR>(device,sigma,sigma_down,[xd.0,xd.1],&g,wug,wug_scale);
    // slices256c3: both groups' half-0 activations depend only on pair 0's codes, so they load inside the Up/Gate window.
    let x00 = super::super::rowdown_slices256c2_f8::load_x_half::<T,HALF,PS>(device, &dcodes0, 0);
    let x01 = super::super::rowdown_slices256c2_f8::load_x_half::<T,HALF,PS>(device, &dcodes0, 1);
    let (d0,d1) = super::super::rowdown_slices256c2_f8::down_split::<T,HALF,PS>(device, x00, x01, &dcodes1, wd, wd_scale, sigma_down);
    let gp = rotate2::gamma_nat_ag2::<T, HALF>(device, post_ff_rms_weight);
    let s = post::scalar_vrf::<T, HALF>(device, layer_scalar);
    let out0 = post2::post_ffn_from_dm::<T, HALF>(device, &d0, &gp, &s, xd.0);
    let v0: DmTensorView<'_, bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![H]> = unsafe { out0.view().reshape() };
    v0.to_hbm_view(&mut device.tdma, x.view_mut().tile::<m![T / { PS }], 1, m![T / { PS } = 1 #{!} 2, T % { PS }, H]>(0));
    let out1 = post2::post_ffn_from_dm::<T, HALF>(device, &d1, &gp, &s, xd.1);
    let v1: DmTensorView<'_, bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![H]> = unsafe { out1.view().reshape() };
    v1.to_hbm_view(&mut device.tdma, x.view_mut().tile::<m![T / { PS }], 1, m![T / { PS } = 1 #{!} 2, T % { PS }, H]>(1));
    (out0, out1)
}

// Share the packed Up/Gate weights across token passes.

#[cfg(feature = "compile-carried")]
/// Three-pass pair producer with full-quarter weight-stationary Down.
pub(crate) fn stage3_up_dm_pp_weight_once<
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
    xd: (&Row<bf16, T, HALF, m![Kh]>, &Row<bf16, T, HALF, m![Kh]>, &Row<bf16, T, HALF, m![Kh]>),
    x: &mut HbmTensor<bf16, Chip, m![T, H]>,
    _eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>,
    pre_ff_rms_weight: &HbmTensor<bf16, Chip, m![Rc, H]>,
    wug: &HbmTensor<f8e4m3, Chip, m![Lp, H]>,
    wug_scale: &HbmTensor<f32, Chip, m![Lu]>,
    wd: &HbmTensor<f8e4m3, Chip, m![Kq4, H, Kd]>,
    wd_scale: &HbmTensor<f32, Chip, m![H]>,
    post_ff_rms_weight: &HbmTensor<bf16, Chip, m![Rc, H]>,
    layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
) -> (Row<bf16, T, HALF, m![Kh]>, Row<bf16, T, HALF, m![Kh]>, Row<bf16, T, HALF, m![Kh]>) {
    let g = rotate2::gamma_nat_ag2::<T, HALF>(device, pre_ff_rms_weight);
    let (dcodes0, dcodes1) = super::up_xchg::up_pp3::<T,HALF,PS,3,URING,UXS,UYS,TT,TR>(device,sigma,sigma_down,[xd.0,xd.1,xd.2],&g,wug,wug_scale);
    let (d0,d1,d2)=super::super::rowdown_dm8_f8::down_split3::<T,HALF,PS>(device,&dcodes0,&dcodes1,wd,wd_scale,sigma_down);
    let gp = rotate2::gamma_nat_ag2::<T, HALF>(device, post_ff_rms_weight);
    let s = post::scalar_vrf::<T, HALF>(device, layer_scalar);
    let out0 = post2::post_ffn_from_dm::<T,HALF>(device,&d0, &gp, &s, xd.0);
    let v0: DmTensorView<'_, bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![H]> = unsafe { out0.view().reshape() };
    v0.to_hbm_view(&mut device.tdma, x.view_mut().tile::<m![T / { PS }], 1, m![T / { PS } = 1 #{!} 3, T % { PS }, H]>(0));
    let out1 = post2::post_ffn_from_dm::<T,HALF>(device,&d1, &gp, &s, xd.1);
    let v1: DmTensorView<'_, bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![H]> = unsafe { out1.view().reshape() };
    v1.to_hbm_view(&mut device.tdma, x.view_mut().tile::<m![T / { PS }], 1, m![T / { PS } = 1 #{!} 3, T % { PS }, H]>(1));
    let out2 = post2::post_ffn_from_dm::<T,HALF>(device,&d2, &gp, &s, xd.2);
    let v2: DmTensorView<'_, bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![H]> = unsafe { out2.view().reshape() };
    v2.to_hbm_view(&mut device.tdma, x.view_mut().tile::<m![T / { PS }], 1, m![T / { PS } = 1 #{!} 3, T % { PS }, H]>(2));
    (out0, out1, out2)
}
