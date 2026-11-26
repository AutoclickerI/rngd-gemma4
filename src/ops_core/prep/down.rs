use furiosa_opt_std::prelude::*;

use super::hidden::Row;
use crate::Chip;
use crate::axes::{H, L};
use crate::device::tokenwise_int8::rotate::{Sgi, Sgo, Sl8, TCl, TSl};

/// One Down row decoded exactly to bf16 (`ffn_ug::fp4_decode` over 15,360 elements).
pub(super) fn fp4_decode<T: AxisName, const HALF: usize>(
    device: &mut Device,
    w: &Row<f4e2m1, T, HALF, m![L]>,
    s: &Row<f8e4m3, T, HALF, m![L / 16]>,
) -> Row<bf16, T, HALF, m![L]> {
    let sv: VrfTensor<f32, Chip, TCl<T, HALF>, TSl<T, HALF>, m![L / 16]> = device
        .sub
        .begin(s.view())
        .fetch::<m![L / 128], m![L / 16 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 128], m![L / 16 % 8]>()
        .to_vrf();
    device
        .main
        .begin(w.view())
        .fetch::<m![L / 1920], m![L % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &sv)
        .vector_widen_concat::<m![L / 8], m![L % 8]>()
        .vector_final()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit()
}

/// `rotate::store_scales` with `ws = f32(s * g)` applied on the compacted scales (slice 0 of each
/// cluster), in the block form `hidden::store_scale_blocks` stores.
pub(super) fn compact_ws<T: AxisName, const HALF: usize>(
    device: &mut Device,
    s: Row<f32, T, HALF, m![1 # 8]>,
    onehot: &VrfTensor<f32, Chip, TCl<T, HALF>, m![1 # 256], m![Sgi, Sl8]>,
    g: &VrfTensor<f32, Chip, TCl<T, HALF>, m![1 # 256], m![1 # 8]>,
) -> DmTensor<f32, Chip, m![H / 256 = 2], m![1 # 256], m![H % 256]> {
    let sv: VrfTensor<f32, Chip, TCl<T, HALF>, TSl<T, HALF>, m![1 # 8]> =
        device.sub.begin(s.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let s: Row<f32, T, HALF, m![Sl8]> = unsafe { s.reshape() };
    let s8: Row<f32, T, HALF, m![Sl8]> = device
        .main
        .begin(s.view())
        .fetch::<m![1], m![Sl8]>()
        .collect::<m![1], m![Sl8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip_with_mode(ClipBinaryOpF32::Max, BinaryArgMode::Mode11, &sv)
        .vector_final()
        .commit_trim::<m![Sl8]>()
        .commit();
    let s8: DmTensor<f32, Chip, TCl<T, HALF>, m![Sgo, Sgi], m![Sl8]> = unsafe { s8.reshape() };
    let c: DmTensor<f32, Chip, TCl<T, HALF>, m![1 # 256], m![Sgo, Sl8]> = device
        .main
        .begin(s8.view())
        .fetch::<m![1], m![Sl8]>()
        .switch::<m![1 # 256], m![Sgo, Sgi]>(SwitchConfig::Broadcast1 { slice1: 256, slice0: 1 })
        .collect::<m![Sgo, Sgi], m![Sl8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Sgo, Sgi, Sl8 / 4], m![Sl8 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), onehot)
        .vector_intra_slice_reduce::<Sgi, m![Sgo, Sl8 / 4], m![Sl8 % 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_concat::<m![Sgo], m![Sl8]>()
        .vector_final()
        .commit_trim::<m![Sl8]>()
        .commit();
    // `f32(s * g)` on the compacted scales (a multiply cannot follow the reduce in one chain).
    let w: DmTensor<f32, Chip, TCl<T, HALF>, m![1 # 256], m![Sgo, Sl8]> = device
        .main
        .begin(c.view())
        .fetch::<m![Sgo], m![Sl8]>()
        .collect::<m![Sgo], m![Sl8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Sgo, Sl8 / 4], m![Sl8 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), g)
        .vector_widen_concat::<m![Sgo], m![Sl8]>()
        .vector_final()
        .commit_trim::<m![Sl8]>()
        .commit();
    unsafe { w.reshape() }
}

/// The global in slice 0 of each cluster's VRF.
pub(crate) fn scalar0<T: AxisName, const HALF: usize>(device: &mut Device, g: &HbmTensor<f32, Chip, m![1]>) -> VrfTensor<f32, Chip, TCl<T, HALF>, m![1 # 256], m![1 # 8]> {
    let g: DmTensor<f32, Chip, TCl<T, HALF>, m![1 # 256], m![1 # 8]> = g.to_dm(&mut device.tdma);
    device.sub.begin(g.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf()
}
