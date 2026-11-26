//! Two consecutive Sliding decode groups share the 384-position gather layout.
pub mod pair384 {
use furiosa_opt_std::prelude::*;
use crate::Chip;
pub use crate::axes::{Ds, Gs, Ns, Rs, Rq, Hh, Hqs};
pub use crate::ops_dsl_attention::decode_groups::w384_fp8::{Rw, Po, Pi, Px};
pub use crate::axes::Dummy2 as R2;
pub type C = m![Rw / 4];
pub type S = m![Rw % 4, Po];
pub type S1 = m![Rw % 4, 1 # 64];
pub type SX = m![Rw % 4, Px];
pub type Off = HbmTensor<i32, Chip, m![R2, Rw, Pi, Po % 32, Po / 32]>;
pub type Cap = HbmTensor<f32, Chip, m![R2, Rw, Po, Pi]>;
pub type Rows = HbmTensor<i32, Chip, m![R2, Rw, Hh]>;
pub type Pool = HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>;
pub type KvDm = DmTensor<f8e4m3, Chip, C, S, m![R2, Pi, Ns, Ds]>;
pub type CapDm = DmTensor<f32, Chip, C, S, m![R2, Pi]>;
pub type QueryDm = DmTensor<bf16, Chip, C, m![Rw % 4, Gs, Ns, Ds / 64], m![R2, Ds % 64]>;
pub const WINDOW: usize = 384;
#[derive(DeviceSend)]
pub struct Group<'a> { pub kv_off: &'a Off, pub cap: &'a Cap, pub rows: &'a Rows }
pub(crate) fn gather_group(device: &mut Device, off: &Off, cap: &Cap, k: &Pool, v: &Pool) -> (KvDm, KvDm, CapDm) {
    (k.gather::<m![Rs], m![Ns, Ds]>().by_byte_offsets(off).to_dm(&mut device.tdma),
     v.gather::<m![Rs], m![Ns, Ds]>().by_byte_offsets(off).to_dm(&mut device.tdma),
     cap.to_dm(&mut device.tdma))
}
/// The list index (`Pi`) of the tile holding slot `W - 1`: both halves' newest entries (right-aligned lists).
pub const NEWEST_TILE: usize = Pi::SIZE - 1;
type TileDm = DmTensor<f8e4m3, Chip, C, S, m![R2, Pi = 1, Ns, Ds]>;
/// After the body: both halves' newest tiles (`kv_off`'s tile `NEWEST_TILE`) re-gathered from `k_n`/`v_n` and written
/// over the history gathered before it (`resident::wtop::refresh_newest`, per half), so a hoisted group computes on
/// the K/V a post-body gather reads.
pub(crate) fn refresh_newest(device: &mut Device, kv_off: &Off, k_n: &Pool, v_n: &Pool, kv: &mut (KvDm, KvDm, CapDm)) {
    let mut last = HbmTensor::<i32, Chip, m![R2, Rw, Pi = 1, Po % 32, Po / 32]>::new();
    kv_off.view().tile::<m![Pi], 1, m![R2, Rw, Pi = 1 # 6, Po % 32, Po / 32]>(NEWEST_TILE).to_hbm_view(&mut device.tdma, last.view_mut());
    let k: TileDm = k_n.gather::<m![Rs], m![Ns, Ds]>().by_byte_offsets(&last).to_dm(&mut device.tdma);
    let v: TileDm = v_n.gather::<m![Rs], m![Ns, Ds]>().by_byte_offsets(&last).to_dm(&mut device.tdma);
    splice(device, &k, &mut kv.0);
    splice(device, &v, &mut kv.1);
}
/// Writes `tile` over `dst`'s tile `NEWEST_TILE` in both halves (a plain copy through the main context).
fn splice(device: &mut Device, tile: &TileDm, dst: &mut KvDm) {
    device
        .main
        .begin(tile.view())
        .fetch::<m![R2, Pi = 1, Ns, Ds / 32], m![Ds % 32]>()
        .collect::<m![R2, Pi = 1, Ns, Ds / 32], m![Ds % 32]>()
        .commit_trim::<m![Ds % 32]>()
        .commit_view(dst.view_mut().tile::<m![Pi], 1, m![R2, Pi = 1 #{!} 6, Ns, Ds]>(NEWEST_TILE));
}
        pub(crate) fn group_from_query_dm(device: &mut Device, q_dm: &QueryDm, cap_dm: &CapDm, k: &KvDm, v: &KvDm) -> DmTensor<bf16, Chip, C, S1, m![R2, Ns, Gs, Ds]> {
            let q_trf: TrfTensor<bf16, Chip, C, S, m![Gs], m![Ns, Ds / 64, R2, Ds % 64]> = device
                .sub
                .begin(q_dm.view())
                .fetch::<m![R2, Ds % 64 / 16], m![Ds % 16]>()
                .switch::<m![Rw % 4, Po], m![Gs, Ns, Ds / 64, R2, Ds % 64 / 16]>(SwitchConfig::Broadcast01 { slice1: 64, slice0: 1, time0: 8 })
                .collect::<m![Gs, Ns, Ds / 64, R2, Ds % 64 / 16], m![Ds % 16]>()
                .to_trf();
            let cap_vrf: VrfTensor<f32, Chip, C, S, m![R2, Pi / 2, Pi % 2 # 8]> = device
                .sub
                .begin(cap_dm.view())
                .fetch::<m![R2, Pi / 2], m![Pi % 2]>()
                .collect::<m![R2, Pi / 2], m![Pi % 2 # 8]>()
                .to_vrf();

            let s: DmTensor<f32, Chip, C, S, m![R2, Pi, Ns, Gs]> = device
                .main
                .begin(k.view())
                .fetch::<m![R2, Pi, Ns, Ds / 16], m![Ds % 16]>()
                .fetch_table_lookup::<bf16>()
                .collect::<m![R2, Pi, Ns, Ds / 16], m![Ds % 16]>()
                .contract_outer::<m![R2, Pi, Ns, Ds / 32], m![Ds % 32], _, _, _>(&q_trf)
                .contract_packet::<m![1]>()
                .contract_time::<m![R2, Pi, Ns]>()
                .contract_lane::<m![R2, Pi, Ns], m![Gs # 8]>(LaneMode::Interleaved)
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_clip(ClipBinaryOpF32::Min, &cap_vrf)
                .vector_final()
                .commit_trim::<m![Gs]>()
                .commit();

            let slice_max: DmTensor<f32, Chip, C, S, m![R2, Ns, Gs]> = device
                .main
                .begin(s.view())
                .fetch::<m![R2, Ns / 4, Pi], m![Ns % 4, Gs]>()
                .collect::<m![R2, Ns / 4, Pi], m![Ns % 4, Gs]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![R2, Ns / 4, Pi, Ns % 4 / 2], m![Ns % 2, Gs]>()
                .vector_intra_slice_reduce::<Pi, m![R2, Ns / 4, Ns % 4 / 2], m![Ns % 2, Gs]>(IntraSliceReduceOpF32::Max)
                .vector_widen_concat::<m![R2, Ns / 4], m![Ns % 4, Gs]>()
                .vector_final()
                .commit_trim::<m![Ns % 4, Gs]>()
                .commit();
            let max: DmTensor<f32, Chip, C, SX, m![R2, Ns, Gs]> = device
                .main
                .begin(slice_max.view())
                .fetch::<m![R2, Ns / 4], m![Ns % 4, Gs]>()
                .collect::<m![R2, Ns / 4], m![Ns % 4, Gs]>()
                .vector_init()
                .vector_inter_slice_reduce::<SX, m![R2, Ns / 4]>(InterSliceReduceOpF32::Max)
                .vector_final()
                .commit_trim::<m![Ns % 4, Gs]>()
                .commit();
            let max: DmTensor<f32, Chip, C, S, m![R2, Ns, Gs]> = unsafe { max.reshape() };
            let max_vrf: VrfTensor<f32, Chip, C, S, m![R2, Ns, Gs]> = device
                .sub
                .begin(max.view())
                .fetch::<m![R2, Ns / 4], m![Ns % 4, Gs]>()
                .collect::<m![R2, Ns / 4], m![Ns % 4, Gs]>()
                .to_vrf();

            let p: DmTensor<bf16, Chip, C, S, m![R2, Gs, Ns, Pi / 3, Pi % 3 # 4]> = device
                .main
                .begin(s.view())
                .fetch::<m![R2, Pi / 3, Ns / 4, Pi % 3], m![Ns % 4, Gs]>()
                .collect::<m![R2, Pi / 3, Ns / 4, Pi % 3], m![Ns % 4, Gs]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![R2, Pi / 3, Ns / 4, Pi % 3, Ns % 4 / 2], m![Ns % 2, Gs]>()
                .vector_fp_binary(FpBinaryOp::SubF, &max_vrf)
                .vector_fp_unary(FpUnaryOp::Exp)
                .vector_widen_concat::<m![R2, Pi / 3, Ns / 4, Pi % 3], m![Ns % 4, Gs]>()
                .vector_final()
                .cast::<bf16, m![Ns % 4 # 8, Gs]>()
                .transpose::<m![R2, Pi / 3, Ns / 4, Ns % 4, Gs], m![Pi % 3 # 16]>()
                .commit_trim::<m![Pi % 3 # 4]>()
                .commit();

            let slice_sum: DmTensor<f32, Chip, C, S, m![R2, Ns, Gs]> = device
                .main
                .begin(s.view())
                .fetch::<m![R2, Ns / 4, Pi], m![Ns % 4, Gs]>()
                .collect::<m![R2, Ns / 4, Pi], m![Ns % 4, Gs]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![R2, Ns / 4, Pi, Ns % 4 / 2], m![Ns % 2, Gs]>()
                .vector_fp_binary(FpBinaryOp::SubF, &max_vrf)
                .vector_fp_unary(FpUnaryOp::Exp)
                .vector_intra_slice_reduce::<Pi, m![R2, Ns / 4, Ns % 4 / 2], m![Ns % 2, Gs]>(IntraSliceReduceOpF32::Add)
                .vector_widen_concat::<m![R2, Ns / 4], m![Ns % 4, Gs]>()
                .vector_final()
                .commit_trim::<m![Ns % 4, Gs]>()
                .commit();
            let sum: DmTensor<f32, Chip, C, S1, m![R2, Ns, Gs]> = device
                .main
                .begin(slice_sum.view())
                .fetch::<m![R2, Ns / 4], m![Ns % 4, Gs]>()
                .collect::<m![R2, Ns / 4], m![Ns % 4, Gs]>()
                .vector_init()
                .vector_inter_slice_reduce::<S1, m![R2, Ns / 4]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .commit_trim::<m![Ns % 4, Gs]>()
                .commit();

            let p_trf: TrfTensor<bf16, Chip, C, S, m![Gs], m![R2, Ns, Pi / 3, Pi % 3 # 4]> = device
                .sub
                .begin(p.view())
                .fetch::<m![Gs, R2, Ns / 2], m![Ns % 2, Pi / 3, Pi % 3 # 4]>()
                .collect::<m![Gs, R2, Ns / 2], m![Ns % 2, Pi / 3, Pi % 3 # 4]>()
                .to_trf();
            let sum_vrf: VrfTensor<f32, Chip, C, S1, m![R2, Ns, Gs]> = device
                .sub
                .begin(sum.view())
                .fetch::<m![R2, Ns / 4], m![Ns % 4, Gs]>()
                .collect::<m![R2, Ns / 4], m![Ns % 4, Gs]>()
                .to_vrf();
            let o: DmTensor<bf16, Chip, C, S1, m![R2, Ns, Ds / 32, Gs, Ds % 32]> = device
                .main
                .begin(v.view())
                .fetch::<m![R2, Ns, Ds / 32, Pi, Ds / 16 % 2], m![Ds % 16]>()
                .fetch_table_lookup::<bf16>()
                .collect::<m![R2, Ns, Ds / 32, Pi, Ds / 16 % 2], m![Ds % 16]>()
                .contract_outer::<m![R2, Ns, Ds / 32, Pi], m![Ds % 32], _, _, _>(&p_trf)
                .contract_packet::<m![Ds % 32]>()
                .contract_time::<m![R2, Ns, Ds / 32]>()
                .contract_lane::<m![R2, Ns, Ds / 32, Gs, Ds % 32 / 8], m![Ds % 8]>(LaneMode::Sequential)
                .vector_init()
                .vector_inter_slice_reduce::<S1, m![R2, Ns, Ds / 32, Gs, Ds % 32 / 8]>(InterSliceReduceOpF32::Add)
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![R2, Ns, Ds / 32, Gs, Ds % 32 / 4], m![Ds % 4]>()
                .vector_fp_div(&sum_vrf)
                .vector_widen_concat::<m![R2, Ns, Ds / 32, Gs, Ds % 32 / 8], m![Ds % 8]>()
                .vector_final()
                .cast::<bf16, m![Ds % 8 # 16]>()
                .commit_trim::<m![Ds % 8]>()
                .commit();
            // Preserve PV's natural commit order; this pass only canonicalizes BF16 bytes.
            device.main.begin(o.view())
                .fetch::<m![R2, Ns, Gs, Ds / 16], m![Ds % 16]>()
                .collect::<m![R2, Ns, Gs, Ds / 16], m![Ds % 16]>()
                .commit_trim::<m![Ds % 16]>()
                .commit()
        }

pub(crate) fn gather_query_direct(device: &mut Device, q: &HbmTensor<bf16, Chip, m![Rq, Hqs]>, rows: &Rows) -> QueryDm {
    let halves: DmTensor<bf16, Chip, C,
        m![Rw % 4, Hqs / 256 % 2, Hh, Hqs / 512, Hqs / 64 % 4], m![R2, Hqs % 64]> =
        q.gather::<m![Rq], m![Hqs]>().by_byte_offsets(rows).to_dm(&mut device.tdma);
    unsafe { halves.reshape() }
}
pub(crate) fn attend_gathered(device: &mut Device, q: &HbmTensor<bf16, Chip, m![Rq, Hqs]>, rows: &Rows,
    kv: &(KvDm, KvDm, CapDm), out: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>) {
    let q = gather_query_direct(device, q, rows);
    let o = group_from_query_dm(device, &q, &kv.2, &kv.0, &kv.1);
    let halves: DmTensorView<'_, bf16, Chip, C, S1, m![R2, Hh, Hqs]> = unsafe { o.view().reshape() };
    out.view_mut().scatter::<m![Rq], m![Hqs]>().by_byte_offsets(rows.view()).from_dm_view(&mut device.tdma, halves);
}
}
