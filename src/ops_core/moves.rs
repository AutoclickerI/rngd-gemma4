//! DMA moves between the segment layouts, the attention row spaces and the KV pools.
//! Bucket geometry determines the slices and elements of each transfer.

use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;
use crate::device::layout::Cluster;

/// A step's queries `q [T, Qs]` as head halves into the row space `qin` at `rows [T, Hh]`.
pub(crate) fn scatter_q<T: AxisName, HS: M, HE: M>(
    device: &mut Device,
    q: HbmTensor<bf16, Chip, m![T, Qs]>,
    rows: &HbmTensor<i32, Chip, m![T, Hh]>,
    qin: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
) {
    let q: HbmTensor<bf16, Chip, m![T, Hh, Hqs]> = unsafe { q.reshape() };
    let halves: DmTensor<bf16, Chip, Cluster, HS, HE> = q.to_dm(&mut device.tdma);
    qin.view_mut().scatter::<m![Rq], m![Hqs]>().by_byte_offsets(rows.view()).from_dm_view(&mut device.tdma, halves.view());
}

/// Encode FP8 KV as `e4m3(clamp(f32(x), -448, 448))` of K/V rows in a
/// pool's DM layout (`KS` slices, `KE` element ending in the head axis `D`, `Ds` or `Df`;
/// `FT`/`CT` are `KE` with `D` split into `D / 16` / `D / 8` for the fetch and the collect). No scale; the explicit clamp keeps the
/// Cast Engine's round-to-nearest-even inside the finite range (no NaN; -0 stays -0).
pub(crate) fn e4m3_rows<KS: M, KE: M, D: AxisName, FT: M, CT: M>(
    device: &mut Device,
    rows: &DmTensor<bf16, Chip, Cluster, KS, KE>,
) -> DmTensor<f8e4m3, Chip, Cluster, KS, KE> {
    device
        .main
        .begin(rows.view())
        .fetch::<FT, m![D % 16]>()
        .fetch_cast::<f32>()
        .collect::<CT, m![D % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Min, 448f32)
        .vector_clip(ClipBinaryOpF32::Max, -448f32)
        .vector_final()
        .cast::<f8e4m3, m![D % 8 # 32]>()
        .commit_trim::<m![D % 8]>()
        .commit()
}

/// `scatter_kv` into an FP8 pool: the step's bf16 K or V rows narrowed to E4M3 (`e4m3_rows`) in
/// the scatter's DM layout, then scattered at `slots [T]`.
pub(crate) fn scatter_kv_e4m3<T: AxisName, KS: M, KE: M, FT: M, CT: M>(
    device: &mut Device,
    kv: HbmTensor<bf16, Chip, m![T, Ps]>,
    slots: &HbmTensor<i32, Chip, m![T]>,
    pool: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
) {
    let kv: HbmTensor<bf16, Chip, m![T, Ns, Ds]> = unsafe { kv.reshape() };
    let rows: DmTensor<bf16, Chip, Cluster, KS, KE> = kv.to_dm(&mut device.tdma);
    let rows8 = e4m3_rows::<KS, KE, Ds, FT, CT>(device, &rows);
    pool.view_mut().scatter::<m![Rs], m![Ns, Ds]>().by_byte_offsets(slots.view()).from_dm_view(&mut device.tdma, rows8.view());
}

/// `e4m3_rows` on rows in any cluster layout `CL` (the arithmetic is per element; `scatter_kv_e4m3_cl`).
pub(crate) fn e4m3_rows_cl<CL: M, KS: M, KE: M, D: AxisName, FT: M, CT: M>(
    device: &mut Device,
    rows: &DmTensor<bf16, Chip, CL, KS, KE>,
) -> DmTensor<f8e4m3, Chip, CL, KS, KE> {
    device
        .main
        .begin(rows.view())
        .fetch::<FT, m![D % 16]>()
        .fetch_cast::<f32>()
        .collect::<CT, m![D % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Min, 448f32)
        .vector_clip(ClipBinaryOpF32::Max, -448f32)
        .vector_final()
        .cast::<f8e4m3, m![D % 8 # 32]>()
        .commit_trim::<m![D % 8]>()
        .commit()
}

/// `publish_kv` on an FP8 pool (a byte copy of the staged E4M3 rows).
pub(crate) fn publish_kv8<T: AxisName, KS: M, KE: M>(
    device: &mut Device,
    pool: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
    stage: &HbmTensor<i32, Chip, m![T]>,
    publish: &HbmTensor<i32, Chip, m![T]>,
) {
    let _ = &device;
    let rows: DmTensor<f8e4m3, Chip, Cluster, KS, KE> = pool.gather::<m![Rs], m![Ns, Ds]>().by_byte_offsets(stage).to_dm(&mut device.tdma);
    pool.view_mut().scatter::<m![Rs], m![Ns, Ds]>().by_byte_offsets(publish.view()).from_dm_view(&mut device.tdma, rows.view());
}

/// Publish already-quantized staging rows with real work on both clusters.
/// CL, KS and KE must partition T, Ns and Ds exactly; published destinations
/// must be distinct and must not overwrite another row's staging source.
/// Padding may write its own staging row, as in the planner.
pub(crate) fn publish_kv8_cl<T: AxisName, CL: M, KS: M, KE: M>(
    device: &mut Device,
    pool: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
    stage: &HbmTensor<i32, Chip, m![T]>,
    publish: &HbmTensor<i32, Chip, m![T]>,
) {
    let rows: DmTensor<f8e4m3, Chip, CL, KS, KE> =
        pool.gather::<m![Rs], m![Ns, Ds]>().by_byte_offsets(stage).to_dm(&mut device.tdma);
    pool.view_mut().scatter::<m![Rs], m![Ns, Ds]>()
        .by_byte_offsets(publish.view()).from_dm_view(&mut device.tdma, rows.view());
}
