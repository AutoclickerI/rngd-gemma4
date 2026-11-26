//! Q4 Sliding → Sliding segments fuse the next layer's decode groups and prompt rounds.
//! History gathers precede the segment. After QKV writes the newest K/V,
//! each group refreshes the tile containing the last slot of its right-aligned list.

use furiosa_opt_std::prelude::*;
#[cfg(feature = "compile-carried")]
use super::Int8Consts;
#[cfg(feature = "compile-carried")]
use crate::ops_tokenwise_int8::{StaticF8SlidingQkvWeights as Int8SlidingQkvWeights, SlidingOStaticF8 as SlidingOInt8};
use crate::{Chip, axes::Rq};
#[cfg(feature = "compile-carried")]
use crate::axes::{Ds, H, Hh, Hqs, Ns, Rs, Tok1024, Tok1536};
use crate::ops_dsl_attention::prompt_blocks::{self as pb, Band, Bb, Mid, W1};
#[cfg(feature = "compile-carried")]
use crate::ops_dsl_attention::prompt_blocks::{Cr, D, Edge, G2, Ln, N, Q, Tp, Vt, Wt};

axes![Rr = 8]; // the row space in blocks of 1,024 rows (one prompt round's queries)

/// The row-space block (of 1,024 rows) a resident program's first prompt round reads and writes: rows 1,024..; the
/// decode groups take rows `0.. 16 G` below it. The plan places the step's prompt image there
/// (`dsl::PromptPlacement`).
pub const PROMPT_BLOCK: usize = 1;
/// `PROMPT_BLOCK` in row-space rows.
pub const PROMPT_ROW: usize = PROMPT_BLOCK * 1024;

const _: () = assert!(Rr::SIZE * 1024 == Rq::SIZE, "Rr x 1024 rows is the row space");

#[cfg(feature = "compile-carried")]
pub(crate) type Pool = HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>;
/// One decode group's in-program part at the window of `decode_groups::$dg` (`$pi` tiles of 64 entries): its
/// hoisted K/V gathers and cap load, the newest tile's refresh, the compute (`$dg::group`'s passes in the same order)
/// and the q/output moves. `$kvf`, `$p_flit`, `$p_elem`: the PV probabilities' TRF layout, as `decode`'s kernel of
/// that window takes it (`ops_dsl_attention::decode`), so a group's bits are the separate launch's. `$($group)*`
/// names the group's argument struct `Group` and its rows `GroupRows`.
macro_rules! window_group {
    ($m:ident, $dg:ident, $pi:literal, $pq:literal, $cq:literal, $kvf:literal, $p_flit:ty, $p_elem:ty, $p_dm:ty, $p_commit:ty, $p_fetch:ty, $($group:item)*) => {
        pub mod $m {
        use furiosa_opt_std::prelude::*;

        #[cfg(feature = "compile-carried")]
        use super::Pool;
        use crate::Chip;
        use crate::axes::{Ds, Gs, Hh, Ns};
        #[cfg(feature = "compile-carried")]
        use crate::axes::{Hqs, Rq, Rs};
        use crate::ops_dsl_attention::decode_groups::$dg::{self as g8, Pi, Po, Rw};

        $($group)*

        /// A group's tables at this window: E4M3 byte offsets in the gather's list order and the per-slot caps.
        pub type Off = g8::Off;
        pub type Cap = g8::Cap;
        /// This window in positions.
        pub const WINDOW: usize = 64 * Pi::SIZE;

        /// Host: a group's tables (`kv_off` words, `cap` values; `DECODE_ROWS x WINDOW` each) uploaded at this window's
        /// types (typed launches).
        pub async fn upload_tables(device: &mut Device, off: Vec<i32>, cap: Vec<f32>) -> Result<(Off, Cap), Error> {
            let off = HostTensor::<i32, m![Rw, Pi, Po % 32, Po / 32]>::from_vec(off).to_hbm(&mut device.pdma).await?;
            let cap = HostTensor::<f32, m![Rw, Po, Pi]>::from_vec(cap).to_hbm(&mut device.pdma).await?;
            Ok((off, cap))
        }

        /// Host: a group's 16 row-space rows (byte offsets) at this window's row axis.
        pub async fn upload_rows(device: &mut Device, rows: Vec<i32>) -> Result<GroupRows, Error> {
            HostTensor::<i32, m![Rw, Hh]>::from_vec(rows).to_hbm(&mut device.pdma).await
        }

        pub type KvDm = DmTensor<f8e4m3, Chip, g8::C, g8::S, m![Pi, Ns, Ds]>;
        pub type CapDm = DmTensor<f32, Chip, g8::C, g8::S, m![Pi]>;
        /// The last list tile of a group's K or V (64 entries per row: one per slice).
        #[cfg(feature = "compile-carried")]
        type TileDm = DmTensor<f8e4m3, Chip, g8::C, g8::S, m![Pi = 1, Ns, Ds]>;

        /// The list index (`Pi`) of the tile holding slot `W - 1` (the newest entry of a right-aligned list).
        pub const NEWEST_TILE: usize = Pi::SIZE - 1;

        /// One group's hoisted part: its K and V gathers (one 2 KiB E4M3 slot per position) and its cap load.
        #[cfg(feature = "compile-carried")]
        pub(crate) fn gather_group(device: &mut Device, kv_off: &g8::Off, cap: &g8::Cap, k_cache: &Pool, v_cache: &Pool) -> (KvDm, KvDm, CapDm) {
            let k: KvDm = k_cache.gather::<m![Rs], m![Ns, Ds]>().by_byte_offsets(kv_off).to_dm(&mut device.tdma);
            let v: KvDm = v_cache.gather::<m![Rs], m![Ns, Ds]>().by_byte_offsets(kv_off).to_dm(&mut device.tdma);
            let cap_dm: CapDm = cap.to_dm(&mut device.tdma);
            (k, v, cap_dm)
        }

        /// q in the compute owner's physical slices (`gather_query_direct` lands it there; no HBM staging needed).
        pub type QueryDm = DmTensor<bf16, Chip, g8::C, m![Rw % 4, Gs, Ns, Ds / 64], m![Ds % 64]>;

        /// Attention over the query, gathered K/V and cap in DM, in standalone decode pass order.
        #[cfg(feature = "compile-carried")]
        pub(crate) fn group_from_query_dm(device: &mut Device, q_dm: &QueryDm, cap_dm: &CapDm, k: &KvDm, v: &KvDm) -> DmTensor<bf16, Chip, g8::C, g8::S1, m![Ns, Gs, Ds]> {
            let q_trf: TrfTensor<bf16, Chip, g8::C, g8::S, m![Gs], m![Ns, Ds]> = device
                .sub
                .begin(q_dm.view())
                .fetch::<m![Ds % 64 / 16], m![Ds % 16]>()
                .switch::<m![Rw % 4, Po], m![Gs, Ns, Ds / 64, Ds % 64 / 16]>(SwitchConfig::Broadcast01 { slice1: 64, slice0: 1, time0: 4 })
                .collect::<m![Gs, Ns, Ds / 16], m![Ds % 16]>()
                .to_trf();
            let cap_vrf: VrfTensor<f32, Chip, g8::C, g8::S, m![Pi / $cq, Pi % $cq # 8]> = device
                .sub
                .begin(cap_dm.view())
                .fetch::<m![Pi / $cq], m![Pi % $cq]>()
                .collect::<m![Pi / $cq], m![Pi % $cq # 8]>()
                .to_vrf();

            let s: DmTensor<f32, Chip, g8::C, g8::S, m![Pi, Ns, Gs]> = device
                .main
                .begin(k.view())
                .fetch::<m![Pi, Ns, Ds / 16], m![Ds % 16]>()
                .fetch_table_lookup::<bf16>()
                .collect::<m![Pi, Ns, Ds / 16], m![Ds % 16]>()
                .contract_outer::<m![Pi, Ns, Ds / 32], m![Ds % 32], _, _, _>(&q_trf)
                .contract_packet::<m![1]>()
                .contract_time::<m![Pi, Ns]>()
                .contract_lane::<m![Pi, Ns], m![Gs # 8]>(LaneMode::Interleaved)
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_clip(ClipBinaryOpF32::Min, &cap_vrf)
                .vector_final()
                .commit_trim::<m![Gs]>()
                .commit();

            let slice_max: DmTensor<f32, Chip, g8::C, g8::S, m![Ns, Gs]> = device
                .main
                .begin(s.view())
                .fetch::<m![Ns / 4, Pi], m![Ns % 4, Gs]>()
                .collect::<m![Ns / 4, Pi], m![Ns % 4, Gs]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Ns / 4, Pi, Ns % 4 / 2], m![Ns % 2, Gs]>()
                .vector_intra_slice_reduce::<Pi, m![Ns / 4, Ns % 4 / 2], m![Ns % 2, Gs]>(IntraSliceReduceOpF32::Max)
                .vector_widen_concat::<m![Ns / 4], m![Ns % 4, Gs]>()
                .vector_final()
                .commit_trim::<m![Ns % 4, Gs]>()
                .commit();
            let max: DmTensor<f32, Chip, g8::C, g8::SX, m![Ns, Gs]> = device
                .main
                .begin(slice_max.view())
                .fetch::<m![Ns / 4], m![Ns % 4, Gs]>()
                .collect::<m![Ns / 4], m![Ns % 4, Gs]>()
                .vector_init()
                .vector_inter_slice_reduce::<g8::SX, m![Ns / 4]>(InterSliceReduceOpF32::Max)
                .vector_final()
                .commit_trim::<m![Ns % 4, Gs]>()
                .commit();
            let max: DmTensor<f32, Chip, g8::C, g8::S, m![Ns, Gs]> = unsafe { max.reshape() };
            let max_vrf: VrfTensor<f32, Chip, g8::C, g8::S, m![Ns, Gs]> = device
                .sub
                .begin(max.view())
                .fetch::<m![Ns / 4], m![Ns % 4, Gs]>()
                .collect::<m![Ns / 4], m![Ns % 4, Gs]>()
                .to_vrf();

            let p: DmTensor<bf16, Chip, g8::C, g8::S, $p_dm> = device
                .main
                .begin(s.view())
                .fetch::<m![Pi / $pq, Ns / 4, Pi % $pq], m![Ns % 4, Gs]>()
                .collect::<m![Pi / $pq, Ns / 4, Pi % $pq], m![Ns % 4, Gs]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Pi / $pq, Ns / 4, Pi % $pq, Ns % 4 / 2], m![Ns % 2, Gs]>()
                .vector_fp_binary(FpBinaryOp::SubF, &max_vrf)
                .vector_fp_unary(FpUnaryOp::Exp)
                .vector_widen_concat::<m![Pi / $pq, Ns / 4, Pi % $pq], m![Ns % 4, Gs]>()
                .vector_final()
                .cast::<bf16, m![Ns % 4 # 8, Gs]>()
                .transpose::<m![Pi / $pq, Ns / 4, Ns % 4, Gs], m![Pi % $pq # 16]>()
                .commit_trim::<$p_commit>()
                .commit();

            let slice_sum: DmTensor<f32, Chip, g8::C, g8::S, m![Ns, Gs]> = device
                .main
                .begin(s.view())
                .fetch::<m![Ns / 4, Pi], m![Ns % 4, Gs]>()
                .collect::<m![Ns / 4, Pi], m![Ns % 4, Gs]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Ns / 4, Pi, Ns % 4 / 2], m![Ns % 2, Gs]>()
                .vector_fp_binary(FpBinaryOp::SubF, &max_vrf)
                .vector_fp_unary(FpUnaryOp::Exp)
                .vector_intra_slice_reduce::<Pi, m![Ns / 4, Ns % 4 / 2], m![Ns % 2, Gs]>(IntraSliceReduceOpF32::Add)
                .vector_widen_concat::<m![Ns / 4], m![Ns % 4, Gs]>()
                .vector_final()
                .commit_trim::<m![Ns % 4, Gs]>()
                .commit();
            let sum: DmTensor<f32, Chip, g8::C, g8::S1, m![Ns, Gs]> = device
                .main
                .begin(slice_sum.view())
                .fetch::<m![Ns / 4], m![Ns % 4, Gs]>()
                .collect::<m![Ns / 4], m![Ns % 4, Gs]>()
                .vector_init()
                .vector_inter_slice_reduce::<g8::S1, m![Ns / 4]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .commit_trim::<m![Ns % 4, Gs]>()
                .commit();

            let p_trf: TrfTensor<bf16, Chip, g8::C, g8::S, m![Gs], $p_elem> = device
                .sub
                .begin(p.view())
                .fetch::<m![Gs, Ns / $kvf], $p_fetch>()
                .collect::<m![Gs, Ns / $kvf], $p_flit>()
                .to_trf();
            let o: DmTensor<f32, Chip, g8::C, g8::S1, m![Ns, Ds / 32, Gs, Ds % 32]> = device
                .main
                .begin(v.view())
                .fetch::<m![Ns, Ds / 32, Pi, Ds / 16 % 2], m![Ds % 16]>()
                .fetch_table_lookup::<bf16>()
                .collect::<m![Ns, Ds / 32, Pi, Ds / 16 % 2], m![Ds % 16]>()
                .contract_outer::<m![Ns, Ds / 32, Pi], m![Ds % 32], _, _, _>(&p_trf)
                .contract_packet::<m![Ds % 32]>()
                .contract_time::<m![Ns, Ds / 32]>()
                .contract_lane::<m![Ns, Ds / 32, Gs, Ds % 32 / 8], m![Ds % 8]>(LaneMode::Sequential)
                .vector_init()
                .vector_inter_slice_reduce::<g8::S1, m![Ns, Ds / 32, Gs, Ds % 32 / 8]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .commit_trim::<m![Ds % 8]>()
                .commit();

            let sum_vrf: VrfTensor<f32, Chip, g8::C, g8::S1, m![Ns, Gs]> = device
                .sub
                .begin(sum.view())
                .fetch::<m![Ns / 4], m![Ns % 4, Gs]>()
                .collect::<m![Ns / 4], m![Ns % 4, Gs]>()
                .to_vrf();
            device
                .main
                .begin(o.view())
                .fetch::<m![Ns, Gs, Ds / 8], m![Ds % 8]>()
                .collect::<m![Ns, Gs, Ds / 8], m![Ds % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Ns, Gs, Ds / 4], m![Ds % 4]>()
                .vector_fp_div(&sum_vrf)
                .vector_widen_concat::<m![Ns, Gs, Ds / 8], m![Ds % 8]>()
                .vector_final()
                .cast::<bf16, m![Ds % 8 # 16]>()
                .commit_trim::<m![Ds % 8]>()
                .commit()
        }

        /// Gather queries directly into the compute slices, attend to the gathered K/V, and
        /// scatter the results back to the same rows without staging through HBM.
        #[cfg(feature = "compile-carried")]
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn attend_gathered(
            device: &mut Device,
            qin_s: &HbmTensor<bf16, Chip, m![Rq, Hqs]>,
            rows: &GroupRows,
            kv: &(KvDm, KvDm, CapDm),
            out_s: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
        ) {
            let q = gather_query_direct(device, qin_s, rows);
            let o = group_from_query_dm(device, &q, &kv.2, &kv.0, &kv.1);
            scatter_output_direct(device, &o, rows, out_s);
        }

        /// Direct gather into the query consumer's *actual* physical owners. A half is
        /// Hqs = (Ns % 4, Gs, Ds), Hh = Ns / 4; reorder only named axes, not bytes.
        #[cfg(feature = "compile-carried")]
        pub(crate) fn gather_query_direct(device: &mut Device, qin_s: &HbmTensor<bf16, Chip, m![Rq, Hqs]>, rows: &GroupRows) -> QueryDm {
            let halves: DmTensor<bf16, Chip, g8::C,
                m![Rw % 4, Hqs / 256 % 2, Hh, Hqs / 512, Hqs / 64 % 4], m![Hqs % 64]> =
                qin_s.gather::<m![Rq], m![Hqs]>().by_byte_offsets(rows).to_dm(&mut device.tdma);
            // Same cluster,row,head,64-channel chunk and lane at every physical location.
            unsafe { halves.reshape() }
        }

        /// The output's final DM layout read as row halves, scattered straight to `rows`.
        #[cfg(feature = "compile-carried")]
        pub(crate) fn scatter_output_direct(device: &mut Device, o: &DmTensor<bf16, Chip, g8::C, g8::S1, m![Ns, Gs, Ds]>, rows: &GroupRows, out_s: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>) {
            let halves: DmTensorView<'_, bf16, Chip, g8::C, g8::S1, m![Hh, Hqs]> = unsafe { o.view().reshape() };
            out_s.view_mut().scatter::<m![Rq], m![Hqs]>().by_byte_offsets(rows.view()).from_dm_view(&mut device.tdma, halves);
        }

        /// After the body: re-gathers the last tile of a group's list (`kv_off`'s tile `NEWEST_TILE`, copied to a scratch
        /// table) from `k_n`/`v_n` and writes it over the hoisted K/V's same tile in DM through the main context
        /// (fetch, commit into the tile view). A DM -> DM DMA into the tile view does not lower in Opt 0.8.1 (it copies the
        /// whole padded element and exceeds the source range), so the copy runs on the tensor unit.
        #[cfg(feature = "compile-carried")]
        pub(crate) fn refresh_newest(device: &mut Device, kv_off: &g8::Off, k_n: &Pool, v_n: &Pool, kv: &mut (KvDm, KvDm, CapDm)) {
            let mut last = HbmTensor::<i32, Chip, m![Rw, Pi = 1, Po % 32, Po / 32]>::new();
            kv_off.view().tile::<m![Pi], 1, m![Rw, Pi = 1 # $pi, Po % 32, Po / 32]>(NEWEST_TILE).to_hbm_view(&mut device.tdma, last.view_mut());
            let k: TileDm = k_n.gather::<m![Rs], m![Ns, Ds]>().by_byte_offsets(&last).to_dm(&mut device.tdma);
            let v: TileDm = v_n.gather::<m![Rs], m![Ns, Ds]>().by_byte_offsets(&last).to_dm(&mut device.tdma);
            splice(device, &k, &mut kv.0);
            splice(device, &v, &mut kv.1);
        }

        /// Writes `tile` over `dst`'s tile `NEWEST_TILE` (a plain copy through the main context).
        #[cfg(feature = "compile-carried")]
        fn splice(device: &mut Device, tile: &TileDm, dst: &mut KvDm) {
            device
                .main
                .begin(tile.view())
                .fetch::<m![Pi = 1, Ns, Ds / 32], m![Ds % 32]>()
                .collect::<m![Pi = 1, Ns, Ds / 32], m![Ds % 32]>()
                .commit_trim::<m![Ds % 32]>()
                .commit_view(dst.view_mut().tile::<m![Pi], 1, m![Pi = 1 #{!} $pi, Ns, Ds]>(NEWEST_TILE));
        }

        }
    };
}

window_group!(w384, w384_fp8, 6, 3, 2, 2, m![Ns % 2, Pi / 3, Pi % 3 # 4], m![Ns, Pi / 3, Pi % 3 # 4], m![Gs, Ns, Pi / 3, Pi % 3 # 4], m![Pi % 3 # 4], m![Ns % 2, Pi / 3, Pi % 3 # 4],
    /// Row-space byte offsets for a decode group.
    pub type GroupRows = HbmTensor<i32, Chip, m![Rw, Hh]>;
    /// One decode group's tables at window 384.
    #[derive(DeviceSend)]
    pub struct Group<'a> {
        pub kv_off: &'a g8::Off,
        pub cap: &'a g8::Cap,
        pub rows: &'a GroupRows,
    }

);
pub use self::w384 as wtop;
const _: () = assert!(wtop::WINDOW == crate::ops_dsl_attention::window::GROUP_WINDOW, "wtop runs at the group window");

/// A round's offsets for the `[Rs, Ns, Ds]` pools: the Full resident `prompt_blocks::Off<Wn>` without its KV-head axis (its
/// first eighth: the head's offset `n · 256` is zero there). The gather spreads each pool row's eight heads over the
/// eight PEs, as F's per-head offsets do (one pool argument, so the gathers follow the body's staging scatter).
pub type Off0<Wn> = HbmTensor<i32, Chip, m![Wn / 128, Wn % 8, Bb % 2, Wn / 8 % 16, Bb / 2]>;

/// A loaded round, as the Full resident `prompt_blocks::round_wN` takes it.
#[allow(clippy::type_complexity)]
#[cfg(feature = "compile-carried")]
pub(crate) type Loaded<Wn> = (
    DmTensor<bf16, Chip, pb::C, pb::S, m![Q % 8, G2, D]>,
    DmTensor<f8e4m3, Chip, pb::C, pb::SK<Wn>, m![Wn / 128, Wn % 8, D]>,
    DmTensor<f8e4m3, Chip, pb::C, pb::SK<Wn>, m![Wn / 128, Wn % 8, D]>,
    DmTensor<f32, Chip, pb::C, pb::S, m![Wn / 128 # 32, Q % 8]>,
    pb::LowerCaps,
);

/// `prompt_blocks::load_wN` on the resident program's arguments: the round's queries from `qin_s` at row-space block
/// `block`, its windows from `k_n`/`v_n` (`Off0`), its masks as F gathers them.
#[cfg(feature = "compile-carried")]
macro_rules! resident_load {
    ($load:ident, $wn:ident) => {
        pub(crate) fn $load(
            device: &mut Device,
            qin_s: &HbmTensor<bf16, Chip, m![Rq, Hqs]>,
            block: usize,
            off: &Off0<$wn>,
            keys: &pb::Keys<'_>,
            masks: &pb::Masks<'_>,
            k_n: &Pool,
            v_n: &Pool,
        ) -> Loaded<$wn> {
            // SAFETY: `[Rq, Hqs]` is `[Rr, Q, N, G2, D]` byte for byte (1,024 rows per block, a kernel row's two halves
            // are heads 0..8 and 8..16).
            let qv: HbmTensorView<'_, bf16, Chip, m![Rr, Q, N, G2, D]> = unsafe { qin_s.view().reshape() };
            let q_dm: DmTensor<bf16, Chip, pb::C, pb::S, m![Q % 8, G2, D]> =
                qv.tile::<m![Rr], 1, m![Rr = 1 # 8, Q, N, G2, D]>(block).to_dm(&mut device.tdma);
            let kg: DmTensor<f8e4m3, Chip, m![Ns / 4], m![Ns % 4, Bb, $wn / 8 % 16], m![$wn / 128, $wn % 8, Ds]> = k_n.gather::<m![Rs], m![Ns, Ds]>().by_byte_offsets(off).to_dm(&mut device.tdma);
            let vg: DmTensor<f8e4m3, Chip, m![Ns / 4], m![Ns % 4, Bb, $wn / 8 % 16], m![$wn / 128, $wn % 8, Ds]> = v_n.gather::<m![Rs], m![Ns, Ds]>().by_byte_offsets(off).to_dm(&mut device.tdma);
            let tm: DmTensor<f32, Chip, pb::C, m![N % 4, Bb, Q / 8 % 16], m![Tp, Ln]> = masks.tm_rows.gather::<m![Vt], m![Tp, Ln]>().by_byte_offsets(keys.tm).to_dm(&mut device.tdma);
            let cap_lo: DmTensor<f32, Chip, pb::C, m![N % 4, Bb, Q / 8 % 16], m![Edge, Wt, Ln]> = masks.cap_rows.gather::<m![Cr], m![Edge, Wt, Ln]>().by_byte_offsets(keys.cap).to_dm(&mut device.tdma);
            // The windows as the key slices of the query slices' sub-rings (F's reshape; `Ns` is `N`).
            unsafe { (q_dm, kg.reshape(), vg.reshape(), tm.reshape(), cap_lo.reshape()) }
        }
    };
}

#[cfg(feature = "compile-carried")]
resident_load!(load_rs_band, Band);
#[cfg(feature = "compile-carried")]
resident_load!(load_rs_mid, Mid);
// The final single-tile prompt round uses neither mask.
#[cfg(feature = "compile-carried")]
pub(crate) fn load_rs_w1(
    device: &mut Device,
    qin_s: &HbmTensor<bf16, Chip, m![Rq, Hqs]>,
    block: usize,
    off: &Off0<W1>,
    _keys: &pb::Keys<'_>,
    _masks: &pb::Masks<'_>,
    k_n: &Pool,
    v_n: &Pool,
) -> pb::LoadedSingle {
    // SAFETY: as `resident_load`.
    let qv: HbmTensorView<'_, bf16, Chip, m![Rr, Q, N, G2, D]> = unsafe { qin_s.view().reshape() };
    let q_dm: DmTensor<bf16, Chip, pb::C, pb::S, m![Q % 8, G2, D]> =
        qv.tile::<m![Rr], 1, m![Rr = 1 # 8, Q, N, G2, D]>(block).to_dm(&mut device.tdma);
    let kg: DmTensor<f8e4m3, Chip, m![Ns / 4], m![Ns % 4, Bb, W1 / 8 % 16], m![W1 / 128, W1 % 8, Ds]> = k_n.gather::<m![Rs], m![Ns, Ds]>().by_byte_offsets(off).to_dm(&mut device.tdma);
    let vg: DmTensor<f8e4m3, Chip, m![Ns / 4], m![Ns % 4, Bb, W1 / 8 % 16], m![W1 / 128, W1 % 8, Ds]> = v_n.gather::<m![Rs], m![Ns, Ds]>().by_byte_offsets(off).to_dm(&mut device.tdma);
    unsafe { (q_dm, kg.reshape(), vg.reshape()) }
}

/// Stores a round's output rows into the row space at block `block`, through `out_q`: `out_s` retyped as
/// `[Rr, Q, N, G2, D]` (the host binds `out_s`'s buffer twice; a reshaped `view_mut` tile does not lower in Opt 0.8.1:
/// "mir: expected a HbmTensorViewMut"). The two bindings alias, so the scheduler does not order them; nothing needs it:
/// the program's other `out_s` accesses are the body's read of layer `l`'s attention output, which every round's
/// output depends on through `qin_s` (the body writes the rounds' queries after its O projection), and the decode
/// groups' scatters, to rows `0.. 16 G` of block 0, which no round writes.
#[cfg(feature = "compile-carried")]
pub(crate) fn store_round(device: &mut Device, o: &pb::Out, out_q: &mut HbmTensor<bf16, Chip, m![Rr, Q, N, G2, D]>, block: usize) {
    o.view().to_hbm_view(&mut device.tdma, out_q.view_mut().tile::<m![Rr], 1, m![Rr = 1 #{!} 8, Q, N, G2, D]>(block));
}

/// A (9, 4, 1) prompt image's per-step tables (three rounds) and the constant mask rows.
#[derive(DeviceSend)]
pub struct Prompt941<'a> {
    pub off0: &'a Off0<Band>,
    pub tm0: &'a pb::Key,
    pub cap0: &'a pb::Key,
    pub off1: &'a Off0<Mid>,
    pub tm1: &'a pb::Key,
    pub cap1: &'a pb::Key,
    pub off2: &'a Off0<W1>,
    pub tm2: &'a pb::Key,
    pub cap2: &'a pb::Key,
    pub tm_rows: &'a pb::TmRows,
    pub cap_rows: &'a pb::CapRows,
    pub cap_hi: &'a pb::CapHi,
}

/// A (9, 9, 9) prompt image's per-step tables (three rounds) and the constant mask rows.
#[derive(DeviceSend)]
pub struct Prompt999<'a> {
    pub off0: &'a Off0<Band>,
    pub tm0: &'a pb::Key,
    pub cap0: &'a pb::Key,
    pub off1: &'a Off0<Band>,
    pub tm1: &'a pb::Key,
    pub cap1: &'a pb::Key,
    pub off2: &'a Off0<Band>,
    pub tm2: &'a pb::Key,
    pub cap2: &'a pb::Key,
    pub tm_rows: &'a pb::TmRows,
    pub cap_rows: &'a pb::CapRows,
    pub cap_hi: &'a pb::CapHi,
}

#[cfg(feature = "compile-carried")]
use crate::ops_dsl_attention::paired::pair384 as dgpair;
/// Paired Sliding attention with Q4 feed-forward kernels, loaded by image path.
#[cfg(feature = "compile-carried")]
pub mod q4 {
    use super::*;
    use crate::ops_core::q4serve::mid_ss_q4_port_t1024_f1_body;

macro_rules! paired_hoist_q4_entries {
    ($tok:ident, $body:ident, $name:ident, $gs:ident, hoist [$($h:ident: $hw:ident),*], post [$($g:ident: $w:ident),*], attend [$($a:ident: $aw:ident),+], $pt:ident, [$(($off:ident, $tm:ident, $cap:ident, $load:ident, $round:ident, $r:literal)),+]) => {
        /// Layer `l`'s INT8 `mid_ss`, layer `l + 1`'s decode groups (hoisted or post-body gathers) and its prompt rounds
        /// (module doc).
        #[device(chip = 1)]
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            device: &mut Device,
            groups: $gs<'_>,
            out_s: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
            rows_s: &HbmTensor<i32, Chip, m![$tok, Hh]>,
            stage_s: &HbmTensor<i32, Chip, m![$tok]>,
            publish_s: &HbmTensor<i32, Chip, m![$tok]>,
            rope_s: &HbmTensor<i32, Chip, m![$tok]>,
            x: &mut HbmTensor<bf16, Chip, m![$tok, H]>,
            consts: Int8Consts<'_>,
            o: SlidingOInt8<'_>,
            ffn: crate::ops_core::q4serve::Q4FfnWeights<'_>,
            layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
            next: Int8SlidingQkvWeights<'_>,
            k_l: &mut Pool,
            v_l: &mut Pool,
            qin_s: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
            k_n: &mut Pool,
            v_n: &mut Pool,
            prompt: $pt<'_>,
            out_q: &mut HbmTensor<bf16, Chip, m![Rr, Q, N, G2, D]>,
        ) {
            $(let mut $h = ($hw::gather_group(device, groups.$h.kv_off, groups.$h.cap, k_n, v_n), groups.$h);)*
            $body::<crate::device::tokenwise_int8::ffn_fast::GlOf<$tok>>(device, &*out_s, rows_s, stage_s, publish_s, rope_s, x, consts, o, ffn, layer_scalar, next, k_l, v_l, qin_s, k_n, v_n);
            $($hw::refresh_newest(device, $h.1.kv_off, k_n, v_n, &mut $h.0);)*
            $(let $g = ($w::gather_group(device, groups.$g.kv_off, groups.$g.cap, k_n, v_n), groups.$g);)*
            $($aw::attend_gathered(device, qin_s, $a.1.rows, &$a.0, out_s);)+
            let masks = pb::Masks { tm_rows: prompt.tm_rows, cap_rows: prompt.cap_rows, cap_hi: prompt.cap_hi };
            $(let $off = $load(device, qin_s, PROMPT_BLOCK + $r, prompt.$off, &pb::Keys { tm: prompt.$tm, cap: prompt.$cap }, &masks, k_n, v_n);)+
            let cap_hi: pb::Cap = prompt.cap_hi.to_dm(&mut device.tdma);
            $(
                let o = pb::$round(device, $off, &cap_hi);
                store_round(device, &o, out_q, PROMPT_BLOCK + $r);
            )+
        }
    };
}

#[cfg(feature = "compile-carried")]
use crate::ops_core::q4serve::mid_ss_q4_port_t1536_body;
#[cfg(feature = "compile-carried")]
include!("resident_g12_q4.rs");
#[cfg(feature = "compile-carried")]
include!("resident_t1024_g12_q4.rs");
}

// These entry definitions are rebuilt explicitly; the server loads their carried images.
#[cfg(feature = "compile-carried")]
include!("resident_t128.rs");

#[cfg(feature = "compile-carried")]
#[derive(DeviceSend)]
pub struct PairGroups12<'a> {
    pub p0: dgpair::Group<'a>,
    pub p1: dgpair::Group<'a>,
    pub p2: dgpair::Group<'a>,
    pub p3: dgpair::Group<'a>,
    pub p4: dgpair::Group<'a>,
    pub p5: dgpair::Group<'a>,
}

#[cfg(feature = "compile-carried")]
use crate::ops_dsl_attention::paired::pair384 as dgpair_pv;
