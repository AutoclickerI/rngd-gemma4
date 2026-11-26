//! Media soft rows into a step's embedded input (`host::media`).
//!
//! The batched encoders (`ops_vision_rows`, `ops_audio_rows`) scatter each media row's soft token
//! into a `[Tok1024, H]` tensor at its step row. A Tok1024 step passes its own `x`; a smaller
//! bucket's `x` (`[T, H]`) receives the rows from a Tok1024 staging tensor through `put_rows`, one
//! launch per step: `PutRows` rows are gathered from the staging tensor and scattered to the same
//! row offsets of `x`. The host pads the index with the last media row's offset (a rewrite of the
//! same bytes), so every other row of `x` keeps the token embedding the engine wrote there.
//! Plain DMA: values are unchanged bit for bit.
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{H, MediaPool, Tok1024};
use crate::device::layout::Cluster;

axes![PutRows = 1024];

/// Copies rows `index[k] / (H x 2)` of `staged` into the same rows of `x`.
#[device(chip = 1)]
pub fn put_rows<T: AxisName>(
    device: &mut Device,
    staged: &HbmTensor<bf16, Chip, m![Tok1024, H]>,
    index: &HbmTensor<i32, Chip, m![PutRows]>,
    x: &mut HbmTensor<bf16, Chip, m![T, H]>,
) {
    let rows: DmTensor<bf16, Chip, Cluster, m![PutRows % 256], m![PutRows / 256, H]> = staged.gather::<m![Tok1024], m![H]>().by_byte_offsets(index).to_dm(&mut device.tdma);
    x.view_mut().scatter::<m![T], m![H]>().by_byte_offsets(index.view()).from_dm_view(&mut device.tdma, rows.view());
}

/// `put_rows` with distinct source and destination rows for a wide bucket.
#[device(chip = 1)]
pub fn put_rows_at<T: AxisName>(
    device: &mut Device,
    staged: &HbmTensor<bf16, Chip, m![Tok1024, H]>,
    source: &HbmTensor<i32, Chip, m![PutRows]>,
    destination: &HbmTensor<i32, Chip, m![PutRows]>,
    x: &mut HbmTensor<bf16, Chip, m![T, H]>,
) {
    let rows: DmTensor<bf16, Chip, Cluster, m![PutRows % 256], m![PutRows / 256, H]> = staged.gather::<m![Tok1024], m![H]>().by_byte_offsets(source).to_dm(&mut device.tdma);
    x.view_mut().scatter::<m![T], m![H]>().by_byte_offsets(destination.view()).from_dm_view(&mut device.tdma, rows.view());
}

/// Additional resident-media rows after the fused embedding's first batch.
/// This reads the already encoded pool directly and preserves earlier batches.
#[device(chip = 1)]
pub fn put_pool_rows<T: AxisName>(
    device: &mut Device,
    pool: &HbmTensor<bf16, Chip, m![MediaPool, H]>,
    source: &HbmTensor<i32, Chip, m![PutRows]>,
    destination: &HbmTensor<i32, Chip, m![PutRows]>,
    x: &mut HbmTensor<bf16, Chip, m![T, H]>,
) {
    let rows: DmTensor<bf16, Chip, Cluster, m![PutRows % 256], m![PutRows / 256, H]> = pool.gather::<m![MediaPool], m![H]>().by_byte_offsets(source).to_dm(&mut device.tdma);
    x.view_mut().scatter::<m![T], m![H]>().by_byte_offsets(destination.view()).from_dm_view(&mut device.tdma, rows.view());
}
