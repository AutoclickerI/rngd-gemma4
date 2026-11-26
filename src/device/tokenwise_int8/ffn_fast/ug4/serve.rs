//! Q4 Gate/Up with static-FP8 Down for the serving segments.
//! The straight-line two-flit body took 908 µs per Tok1024 launch, versus 1,018 µs
//! for FP8. Pass the Q4 operands explicitly instead of the tile-route
//! struct: the pre-FFN norm and rotation, the odd 4-bit activation codes (`row_scale_mid4`: clip 0.8, per row),
//! the packed weight nibbles and their row scales, GeGLU, the static-FP8 Down input and the FP8 Down + post-FFN.
//! Every operand the FP8 stage takes stays in the signature, so a counterpart's ABI is the FP8 program's with the three
//! Q4 tensors appended to the FFN struct (`ops_core::q4serve::Q4FfnWeights`).
use super::*;
use crate::device::tokenwise_int8::ffn_fast::fp8c as f8;
use f8::Lp;

/// Tok1024 (two row passes): drop-in for `fp8c::pair::stage2*`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn stage_dm2<
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
    const QS: usize,
    const NQ: usize,
>(
    device: &mut Device,
    _sigma: &HbmTensor<f32, Chip, m![1 # 8]>,
    sigma_down: &HbmTensor<f32, Chip, m![1 # 8]>,
    xd: (&Row<bf16, T, HALF, m![Kh]>, &Row<bf16, T, HALF, m![Kh]>),
    x: &mut HbmTensor<bf16, Chip, m![T, H]>,
    eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>,
    _pre_ff_rms_weight: &HbmTensor<bf16, Chip, m![Rc, H]>,
    _wug: &HbmTensor<f8e4m3, Chip, m![Lp, H]>,
    _wug_scale: &HbmTensor<f32, Chip, m![Lu]>,
    wd: &HbmTensor<f8e4m3, Chip, m![Kq4, H, Kd]>,
    wd_scale: &HbmTensor<f32, Chip, m![H]>,
    post_ff_rms_weight: &HbmTensor<bf16, Chip, m![Rc, H]>,
    layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
    pre_signed: &HbmTensor<bf16, Chip, m![Rc, H]>,
    codes: &HbmTensor<i8, Chip, m![Lu, Hb]>,
    scales: &HbmTensor<f32, Chip, m![Xc2, Lu]>,
    h64: &HbmTensor<bf16, Chip, m![Rc, Ob, Kb]>,
    h60: &HbmTensor<bf16, Chip, m![Rc, { H60Hbm }]>,
) -> (Row<bf16, T, HALF, m![Kh]>, Row<bf16, T, HALF, m![Kh]>) {
    let mut sx4 = HbmTensor::<f32, Chip, m![T]>::new();
    let mut r2 = HbmTensor::<f32, Chip, m![T]>::new();
    let mut pk: Row<i8,T,HALF,m![T/{PS},G240,J8]>=DmTensor::new();
    {
        let onehot = rotate::onehot::<T, HALF>(device, eye);
        let dc = rotate2::dm_consts::<T, HALF>(device, h64, h60);
        let g4 = rotate2::gamma_nat_ag2::<T, HALF>(device, pre_signed);
        pre_pass_pk::<T, HALF, PS, 2, 0>(device, xd.0, &g4, &dc, &onehot, &mut pk, &mut sx4, &mut r2);
        pre_pass_pk::<T, HALF, PS, 2, 1>(device, xd.1, &g4, &dc, &onehot, &mut pk, &mut sx4, &mut r2);
    }
    let pk:Row<i8,T,HALF,m![T/{PS},Hb]>=unsafe{pk.reshape()};
    let (dcodes0,dcodes1)=super::xg::up_pp::<T,HALF,PS,URING,UXS,UYS,m![T/64],m![T%64]>(device,&pk,&sx4,&r2,sigma_down,codes,scales);
    let x00=super::super::rowdown_slices256c2_f8::load_x_half::<T,HALF,PS>(device,&dcodes0,0);
    let x01=super::super::rowdown_slices256c2_f8::load_x_half::<T,HALF,PS>(device,&dcodes0,1);
    let (d0,d1)=super::super::rowdown_slices256c2_f8::down_split::<T,HALF,PS>(device,x00,x01,&dcodes1,wd,wd_scale,sigma_down);
    let gp = rotate2::gamma_nat_ag2::<T, HALF>(device, post_ff_rms_weight);
    let s = post::scalar_vrf::<T, HALF>(device, layer_scalar);
    let out0 = post2::post_ffn_from_dm::<T,HALF>(device,&d0, &gp, &s, xd.0);
    let v0: DmTensorView<'_, bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![H]> = unsafe { out0.view().reshape() };
    v0.to_hbm_view(&mut device.tdma, x.view_mut().tile::<m![T / { PS }], 1, m![T / { PS } = 1 #{!} 2, T % { PS }, H]>(0));
    let out1 = post2::post_ffn_from_dm::<T,HALF>(device,&d1, &gp, &s, xd.1);
    let v1: DmTensorView<'_, bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![H]> = unsafe { out1.view().reshape() };
    v1.to_hbm_view(&mut device.tdma, x.view_mut().tile::<m![T / { PS }], 1, m![T / { PS } = 1 #{!} 2, T % { PS }, H]>(1));
    (out0, out1)
}

/// One pass for a two-group (Tok128) bucket: the same pre pass, Q4 Gate/Up arithmetic per 64-row group
/// (`up_gate_pure_sl2`, the `up_gate_pure_sl2` body over groups 0 and 1), static FP8 Down and post-FFN.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "compile-carried")]
pub(crate) fn stage_dm1_sl2<
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
    const QS: usize,
    const NQ: usize,
>(
    device: &mut Device,
    _sigma: &HbmTensor<f32, Chip, m![1 # 8]>,
    sigma_down: &HbmTensor<f32, Chip, m![1 # 8]>,
    xd: &Row<bf16, T, HALF, m![Kh]>,
    x: &mut HbmTensor<bf16, Chip, m![T, H]>,
    eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>,
    _pre_ff_rms_weight: &HbmTensor<bf16, Chip, m![Rc, H]>,
    _wug: &HbmTensor<f8e4m3, Chip, m![Lp, H]>,
    _wug_scale: &HbmTensor<f32, Chip, m![Lu]>,
    wd: &HbmTensor<f8e4m3, Chip, m![Kq4, H, Kd]>,
    wd_scale: &HbmTensor<f32, Chip, m![H]>,
    post_ff_rms_weight: &HbmTensor<bf16, Chip, m![Rc, H]>,
    layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
    pre_signed: &HbmTensor<bf16, Chip, m![Rc, H]>,
    codes: &HbmTensor<i8, Chip, m![Lu, Hb]>,
    scales: &HbmTensor<f32, Chip, m![Xc2, Lu]>,
    h64: &HbmTensor<bf16, Chip, m![Rc, Ob, Kb]>,
    h60: &HbmTensor<bf16, Chip, m![Rc, { H60Hbm }]>,
) -> Row<bf16, T, HALF, m![Kh]> {
    let mut sx4 = HbmTensor::<f32, Chip, m![T]>::new();
    let mut r2 = HbmTensor::<f32, Chip, m![T]>::new();
    let mut codes4 = HbmTensor::<i8, Chip, m![T, Hb]>::new();
    {
        let onehot = rotate::onehot::<T, HALF>(device, eye);
        let dc = rotate2::dm_consts::<T, HALF>(device, h64, h60);
        let g4 = rotate2::gamma_nat_ag2::<T, HALF>(device, pre_signed);
        let mut pk: Row<i8, T, HALF, m![T / { PS }, G240, J8]> = DmTensor::new();
        pre_pass_pk::<T, HALF, PS, 1, 0>(device, xd, &g4, &dc, &onehot, &mut pk, &mut sx4, &mut r2);
        let view: DmTensorView<'_, i8, Chip, TCl<T, HALF>, TSl<T, HALF>, m![T / { PS }, Hb]> = unsafe { pk.view().reshape() };
        view.to_hbm_view(&mut device.tdma, codes4.view_mut());
    }
    let act = super::q4f8::up_gate_pure_sl2::<T, URING, UXS, UYS, m![T / 64], m![T % 64], Hb, 15, 480, 960, QS, NQ, m![T, Hb]>(device, codes4.view(), &sx4, &r2, codes, scales);
    let mut dcodes = f8::DCodes8::<T>::new();
    f8::down_input_static8::<T>(device, sigma_down, &act, &mut dcodes);
    let d = super::super::rowdown_f8::down::<T, HALF>(device, &dcodes, wd, wd_scale, sigma_down);
    let gp = rotate2::gamma_nat_ag2::<T, HALF>(device, post_ff_rms_weight);
    let s = post::scalar_vrf::<T, HALF>(device, layer_scalar);
    let out0 = post2::post_ffn_pass_dm::<T, HALF, PS, 1, 0>(device, &d, &gp, &s, xd);
    let v0: DmTensorView<'_, bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![H]> = unsafe { out0.view().reshape() };
    v0.to_hbm_view(&mut device.tdma, x.view_mut().tile::<m![T / { PS }], 1, m![T / { PS } = 1 #{!} 1, T % { PS }, H]>(0));
    out0
}

#[cfg(feature = "compile-carried")]
/// Tok1536 island stage: three complete row passes, unchanged Q4 codes, weight-once FP8 Down.
#[allow(clippy::too_many_arguments)]
pub(crate) fn stage_dm3<
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
    const QS: usize,
    const NQ: usize,
>(
    device: &mut Device,
    _sigma: &HbmTensor<f32, Chip, m![1 # 8]>,
    sigma_down: &HbmTensor<f32, Chip, m![1 # 8]>,
    xd: (&Row<bf16, T, HALF, m![Kh]>, &Row<bf16, T, HALF, m![Kh]>, &Row<bf16, T, HALF, m![Kh]>),
    x: &mut HbmTensor<bf16, Chip, m![T, H]>,
    eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>,
    _pre_ff_rms_weight: &HbmTensor<bf16, Chip, m![Rc, H]>,
    _wug: &HbmTensor<f8e4m3, Chip, m![Lp, H]>,
    _wug_scale: &HbmTensor<f32, Chip, m![Lu]>,
    wd: &HbmTensor<f8e4m3, Chip, m![Kq4, H, Kd]>,
    wd_scale: &HbmTensor<f32, Chip, m![H]>,
    post_ff_rms_weight: &HbmTensor<bf16, Chip, m![Rc, H]>,
    layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
    pre_signed: &HbmTensor<bf16, Chip, m![Rc, H]>,
    codes: &HbmTensor<i8, Chip, m![Lu, Hb]>,
    scales: &HbmTensor<f32, Chip, m![Xc2, Lu]>,
    h64: &HbmTensor<bf16, Chip, m![Rc, Ob, Kb]>,
    h60: &HbmTensor<bf16, Chip, m![Rc, { H60Hbm }]>,
) -> (Row<bf16, T, HALF, m![Kh]>, Row<bf16, T, HALF, m![Kh]>, Row<bf16, T, HALF, m![Kh]>) {
    const { assert!(T::SIZE == 1536 && HALF == 256 && PS == 512); }
    let mut sx4 = HbmTensor::<f32, Chip, m![T]>::new();
    let mut r2 = HbmTensor::<f32, Chip, m![T]>::new();
    let mut pk: Row<i8,T,HALF,m![T/{PS},G240,J8]>=DmTensor::new();
    {
        let onehot = rotate::onehot::<T, HALF>(device, eye);
        let dc = rotate2::dm_consts::<T, HALF>(device, h64, h60);
        let g4 = rotate2::gamma_nat_ag2::<T, HALF>(device, pre_signed);
        pre_pass_pk::<T, HALF, PS, 3, 0>(device, xd.0, &g4, &dc, &onehot, &mut pk, &mut sx4, &mut r2);
        pre_pass_pk::<T, HALF, PS, 3, 1>(device, xd.1, &g4, &dc, &onehot, &mut pk, &mut sx4, &mut r2);
        pre_pass_pk::<T, HALF, PS, 3, 2>(device, xd.2, &g4, &dc, &onehot, &mut pk, &mut sx4, &mut r2);
    }
    let pk:Row<i8,T,HALF,m![T/{PS},Hb]>=unsafe{pk.reshape()};
    let (dcodes0,dcodes1)=super::xg::up_pp::<T,HALF,PS,URING,UXS,UYS,m![T/64],m![T%64]>(device,&pk,&sx4,&r2,sigma_down,codes,scales);
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
