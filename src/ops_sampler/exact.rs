//! Full greedy contract on raw BF16 bits. Integer reductions are local to a slice.
//! The host's half::bf16::to_f32 quiets NaNs before total_cmp; preserve that rule too.
use furiosa_opt_std::prelude::*;
use crate::{Chip, axes::*};

type SC = m![Nseg / 256];
type SS = m![Nseg % 256];

/// Rank (best first) in the upper 16 bits of a 25-bit key, local id in its low 9 bits.
/// `raw` aliases the post-softcap BF16 bytes as signed 16-bit integers, without an FP cast.
/// `mask`: local id, or i32::MAX when banned. An all-banned set uses the unbanned mask,
/// selecting the overall highest value.
#[device(chip = 1)]
pub fn segment(
    device: &mut Device,
    raw: &HbmTensor<i16, Chip, m![Out128, Nseg, Lseg]>,
    mask: &HbmTensor<i32, Chip, m![Nseg, Lseg]>,
    out: &mut HbmTensor<i32, Chip, m![Nseg, Out128]>,
) {
    use BitReq::{Ignore, One, Zero};
    let x: DmTensor<i16, Chip, SC, SS, m![Out128, Lseg]> = raw.to_dm(&mut device.tdma);
    let mask: DmTensor<i32, Chip, SC, SS, m![Lseg]> = mask.to_dm(&mut device.tdma);
    let mask: VrfTensor<i32, Chip, SC, SS, m![Lseg]> = device.sub.begin(mask.view())
        .fetch::<m![Lseg / 8], m![Lseg % 8]>().collect::<m![Lseg / 8], m![Lseg % 8]>().to_vrf();
    let ranked: DmTensor<i32, Chip, SC, SS, m![Out128, Lseg]> = device.main.begin(x.view())
        .fetch::<m![Out128, Lseg / 16], m![Lseg % 16]>().fetch_cast::<i32>()
        .collect::<m![Out128, Lseg / 8], m![Lseg % 8]>().vector_init()
        .vector_intra_slice_tag(TagMode::Comparison([Cmp::Less(0), Cmp::Greater(0x7f80), Cmp::Greater(-128), Cmp::False]))
        // A positive NaN, or a negative value greater than -128, gets its quiet bit.
        .vector_logic(LogicBinaryOpI32::BitOr, Branched::imm(TagGuard::matches([Ignore, One, Ignore, Ignore]), 0x40)
            .imm(TagGuard::matches([One, Zero, One, Ignore]), 0x40))
        .vector_logic(LogicBinaryOpI32::BitXor, Branched::imm(TagGuard::matches([Zero, Ignore, Ignore, Ignore]), 0x7fff))
        .vector_logic(LogicBinaryOpI32::BitAnd, 0xffff)
        .vector_logic(LogicBinaryOpI32::LeftShift, 9)
        .vector_final().commit_trim::<m![Lseg % 8]>().commit();
    let keys: DmTensor<i32, Chip, SC, SS, m![Out128]> = device.main.begin(ranked.view())
        .fetch::<m![Out128, Lseg / 8], m![Lseg % 8]>().collect::<m![Out128, Lseg / 8], m![Lseg % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpI32::BitOr, &mask)
        .vector_narrow_split::<m![Out128, Lseg / 4], m![Lseg % 4]>()
        .vector_intra_slice_reduce::<Lseg, m![Out128], m![1 # 4]>(IntraSliceReduceOpI32::Min)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .transpose::<m![Out128 / 2], m![Out128 % 2 # 8]>().commit_trim::<m![Out128 % 2]>().commit();
    keys.view().to_hbm_view(&mut device.tdma, out.view_mut());
}

/// Gather two rows into each slice (8 contiguous bytes per segment), reduce the
/// value rank locally, then reduce global token ids among equal-rank segments.
#[device(chip = 1)]
pub fn finish(
    device: &mut Device,
    keys: &HbmTensor<i32, Chip, m![Nseg, Out128]>,
    segment_ids: &HbmTensor<i32, Chip, m![Nseg]>,
    out: &mut HbmTensor<i32, Chip, m![Out128]>,
) {
    type LocalCluster = m![1 # 2];
    type LocalSlice = m![Out128 / 2 # 256];
    // DMA moves a contiguous 8-byte row pair. The TU transpose handles at most two
    // segment rows per output packet; direct 4-byte row-major DMA is illegal.
    let interleaved: DmTensor<i32, Chip, LocalCluster, LocalSlice, m![Nseg, Out128 % 2]> = keys.to_dm(&mut device.tdma);
    let keys: DmTensor<i32, Chip, LocalCluster, LocalSlice, m![Out128 % 2, Nseg]> = device.main.begin(interleaved.view())
        .fetch::<m![Nseg], m![Out128 % 2 # 8]>()
        .collect::<m![Nseg], m![Out128 % 2 # 8]>()
        .transpose::<m![Nseg / 2, Out128 % 2], m![Nseg % 2 # 8]>()
        .commit_trim::<m![Nseg % 2]>().commit();
    let ids: DmTensor<i32, Chip, LocalCluster, LocalSlice, m![Nseg]> = segment_ids.to_dm(&mut device.tdma);
    let ids: VrfTensor<i32, Chip, LocalCluster, LocalSlice, m![Nseg]> = device.sub.begin(ids.view())
        .fetch::<m![Nseg / 8], m![Nseg % 8]>().collect::<m![Nseg / 8], m![Nseg % 8]>().to_vrf();
    let minimum: DmTensor<i32, Chip, LocalCluster, LocalSlice, m![Out128 % 2, 1 # 8]> = device.main.begin(keys.view())
        .fetch::<m![Out128 % 2, Nseg / 8], m![Nseg % 8]>().collect::<m![Out128 % 2, Nseg / 8], m![Nseg % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpI32::BitAnd, -512)
        .vector_narrow_split::<m![Out128 % 2, Nseg / 4], m![Nseg % 4]>()
        .vector_intra_slice_reduce::<Nseg, m![Out128 % 2], m![1 # 4]>(IntraSliceReduceOpI32::Min)
        .vector_widen_pad::<m![1 # 8]>().vector_final().commit_trim::<m![1 # 8]>().commit();
    let minimum: VrfTensor<i32, Chip, LocalCluster, LocalSlice, m![Out128 % 2, 1 # 8]> = device.sub.begin(minimum.view())
        .fetch::<m![Out128 % 2], m![1 # 8]>().collect::<m![Out128 % 2], m![1 # 8]>().to_vrf();
    let excluded: DmTensor<i32, Chip, LocalCluster, LocalSlice, m![Out128 % 2, Nseg]> = device.main.begin(keys.view())
        .fetch::<m![Out128 % 2, Nseg / 8], m![Nseg % 8]>().collect::<m![Out128 % 2, Nseg / 8], m![Nseg % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_logic(LogicBinaryOpI32::BitAnd, -512)
        .vector_fxp(FxpBinaryOp::SubFxp, &minimum)
        .vector_clip(ClipBinaryOpI32::Min, 512)
        .vector_final().commit_trim::<m![Nseg % 8]>().commit();
    let candidates: DmTensor<i32, Chip, LocalCluster, LocalSlice, m![Out128 % 2, Nseg]> = device.main.begin(keys.view())
        .fetch::<m![Out128 % 2, Nseg / 8], m![Nseg % 8]>().collect::<m![Out128 % 2, Nseg / 8], m![Nseg % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_logic(LogicBinaryOpI32::BitAnd, 511).vector_logic(LogicBinaryOpI32::BitOr, &ids)
        .vector_final().commit_trim::<m![Nseg % 8]>().commit();
    let candidates: VrfTensor<i32, Chip, LocalCluster, LocalSlice, m![Out128 % 2, Nseg]> = device.sub.begin(candidates.view())
        .fetch::<m![Out128 % 2, Nseg / 8], m![Nseg % 8]>().collect::<m![Out128 % 2, Nseg / 8], m![Nseg % 8]>().to_vrf();
    let picked: DmTensor<i32, Chip, LocalCluster, LocalSlice, m![Out128 % 2]> = device.main.begin(excluded.view())
        .fetch::<m![Out128 % 2, Nseg / 8], m![Nseg % 8]>().collect::<m![Out128 % 2, Nseg / 8], m![Nseg % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpI32::LeftShift, 21).vector_logic(LogicBinaryOpI32::BitOr, &candidates)
        .vector_narrow_split::<m![Out128 % 2, Nseg / 4], m![Nseg % 4]>()
        .vector_intra_slice_reduce::<Nseg, m![Out128 % 2], m![1 # 4]>(IntraSliceReduceOpI32::Min)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .transpose::<m![1], m![Out128 % 2 # 8]>().commit_trim::<m![Out128 % 2]>().commit();
    picked.view().to_hbm_view(&mut device.tdma, out.view_mut());
}
