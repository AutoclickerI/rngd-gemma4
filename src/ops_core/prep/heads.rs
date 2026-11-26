use furiosa_opt_std::prelude::*;

use super::hidden::{self, Hw, Row};
use crate::Chip;
use crate::axes::H;
use crate::device::tokenwise_int8::rotate::{Sgi, Sgo, Sl8, TCl, TSl};

/// The pass's scales compacted into slice 0 of each cluster (`rotate::store_scales`) and stored to
/// `ws` (the FP8 row scale is folded in after the passes, `hidden::fold_row_scale_h`).
pub(super) fn store_ws(
    device: &mut Device,
    s: Row<f32, Hw, 256, m![1 # 8]>,
    onehot: &VrfTensor<f32, Chip, TCl<Hw, 256>, m![1 # 256], m![Sgi, Sl8]>,
    b: usize,
    ws: &mut HbmTensor<f32, Chip, m![H]>,
) {
    let sv: VrfTensor<f32, Chip, TCl<Hw, 256>, TSl<Hw, 256>, m![1 # 8]> =
        device.sub.begin(s.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let s: Row<f32, Hw, 256, m![Sl8]> = unsafe { s.reshape() };
    let s8: Row<f32, Hw, 256, m![Sl8]> = device
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
    let s8: DmTensor<f32, Chip, TCl<Hw, 256>, m![Sgo, Sgi], m![Sl8]> = unsafe { s8.reshape() };
    let c: DmTensor<f32, Chip, TCl<Hw, 256>, m![1 # 256], m![Sgo, Sl8]> = device
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
    hidden::store_scale_blocks(device, unsafe { c.reshape() }, ws, b);
}
