use furiosa_opt_std::prelude::*;

use super::hidden::Row;
use crate::Chip;
use crate::axes::L;
use crate::device::tokenwise_int8::gemm::Pi;
use crate::device::tokenwise_int8::rotate::{Kh, Sgi, Sgo, Sl8, TCl, TSl};

axes![Gt = 2];

/// `wug` typed by the source digits: `[l / 512, l / 256 % 2, l / 16 % 8, l / 128 % 2, l / 8 % 2,
/// gate, l % 8]` rows in physical `Lu` order.
pub(crate) type WugRows = m![L / 512, L / 256 % 2, L / 16 % 8, L / 128 % 2, L / 8 % 2, Gt, L % 8];

/// One NVFP4 row decoded exactly to bf16: the fetch table unpacks the E2M1 codes (as E4M3), the
/// vector engine multiplies by the row's E4M3 block scales (`w * s` is bf16-exact).
pub(super) fn fp4_decode<T: AxisName, const HALF: usize>(
    device: &mut Device,
    w: &Row<f4e2m1, T, HALF, m![Kh]>,
    s: &Row<f8e4m3, T, HALF, m![Kh / 16]>,
) -> Row<bf16, T, HALF, m![Kh]> {
    let sv: VrfTensor<f32, Chip, TCl<T, HALF>, TSl<T, HALF>, m![Kh / 16]> = device
        .sub
        .begin(s.view())
        .fetch::<m![Kh / 128], m![Kh / 16 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Kh / 128], m![Kh / 16 % 8]>()
        .to_vrf();
    device
        .main
        .begin(w.view())
        .fetch::<m![Kh / 1920], m![Kh % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![Kh / 8], m![Kh % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Kh / 4], m![Kh % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &sv)
        .vector_widen_concat::<m![Kh / 8], m![Kh % 8]>()
        .vector_final()
        .cast::<bf16, m![Kh % 8 # 16]>()
        .commit_trim::<m![Kh % 8]>()
        .commit()
}

/// The global in slice 0 of each cluster's VRF (where `store_ws` compacts the scales).
pub(crate) fn scalar0(device: &mut Device, g: &HbmTensor<f32, Chip, m![1]>) -> VrfTensor<f32, Chip, TCl<L, 256>, m![1 # 256], m![1 # 8]> {
    let g: DmTensor<f32, Chip, TCl<L, 256>, m![1 # 256], m![1 # 8]> = g.to_dm(&mut device.tdma);
    device.sub.begin(g.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf()
}

/// `rotate::store_scales` with the scales multiplied by the rows' signs and the global before the
/// store: `ws = f32(s * d) * g`, equal to `f32(s * g) * d` since `d = +-1` (Up rows: `D_down[l]`;
/// Gate rows: the all-ones vector). The pass's per-row values are compacted into slice 0 of each
/// cluster (256 rows), where the cluster's 256 signs load as one contiguous run (a per-slice
/// 4-byte load of the signs fails the DMA verifier).
#[allow(clippy::too_many_arguments)]
pub(super) fn store_ws<SE: M>(
    device: &mut Device,
    s: Row<f32, L, 256, m![1 # 8]>,
    onehot: &VrfTensor<f32, Chip, TCl<L, 256>, m![1 # 256], m![Sgi, Sl8]>,
    g: &VrfTensor<f32, Chip, TCl<L, 256>, m![1 # 256], m![1 # 8]>,
    row_sign: &HbmTensor<f32, Chip, m![L]>,
    p: usize,
    scales: HbmTensorViewMut<'_, f32, Chip, SE>,
) {
    let sv: VrfTensor<f32, Chip, TCl<L, 256>, TSl<L, 256>, m![1 # 8]> =
        device.sub.begin(s.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let s: Row<f32, L, 256, m![Sl8]> = unsafe { s.reshape() };
    let s8: Row<f32, L, 256, m![Sl8]> = device
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
    let s8: DmTensor<f32, Chip, TCl<L, 256>, m![Sgo, Sgi], m![Sl8]> = unsafe { s8.reshape() };
    let c: DmTensor<f32, Chip, TCl<L, 256>, m![1 # 256], m![Sgo, Sl8]> = device
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
    // The cluster's 256 row signs, one run into slice 0.
    let d: DmTensor<f32, Chip, TCl<L, 256>, m![1 # 256], m![L % 256]> =
        row_sign.view().tile::<m![L / 512], 1, m![L / 512 = 1 # 30, L % 512]>(p).to_dm(&mut device.tdma);
    let d: DmTensor<f32, Chip, TCl<L, 256>, m![1 # 256], m![Sgo, Sl8]> = unsafe { d.reshape() };
    let w: DmTensor<f32, Chip, TCl<L, 256>, m![1 # 256], m![Sgo, Sl8]> = device
        .main
        .begin_interleaved::<Pi, _, _, _, _, _>(c.view(), d.view())
        .fetch::<m![Sgo, Pi], m![Sl8]>()
        .collect::<m![Sgo, Pi], m![Sl8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Pi, m![Sgo]>()
        .vector_narrow_split::<m![Sgo, Sl8 / 4], m![Sl8 % 4]>()
        .vector_fp_zip(FpBinaryOp::MulF(FpMulAlu::Mul0))
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), g)
        .vector_widen_concat::<m![Sgo], m![Sl8]>()
        .vector_final()
        .commit_trim::<m![Sl8]>()
        .commit();
    let w: DmTensor<f32, Chip, TCl<L, 256>, m![1 # 256], m![L % 256]> = unsafe { w.reshape() };
    w.view().to_hbm_view(&mut device.tdma, scales);
}
