//! KV pool layout and the per-step publication tables, as pure host functions.
//!
//! Each layer has a KV pool. Attention gathers slot `s` at
//! `base + s * row_bytes`. Slot 0 is reserved for padding.
//!
//! Sliding pool (per layer, `[Rs, Ns, Ds]`, `Ao` rings of `Ts` rows): rings 0 and 1 are reserved:
//! row 0 is the padding slot (never written, never valid, zero), rows `STAGE_ROW0 + i` stage step
//! row `i`'s K/V before attention (up to `MAX_STEP_ROWS`). Owner slot `o < OWNERS` is ring
//! `o + RESERVED_RINGS`, with position `p` in ring row `p % Ts`.
//!
//! Full pool (per layer, `[Rf, Df]`, `Kp` pages of `Tf` rows): page 0 is the zero page (and holds
//! the padding slot 0); an owner's page list (allocator pages `0.. FULL_PAGES`) maps its
//! positions `j * Tf..` to physical page `pages[j] + 1`. Padding rows write to the last page.
use crate::axes::{Ao, Df, Ds, Kp, Ns, Rf, Rs, Tf, Tok1536, Ts};
use furiosa_opt_std::prelude::AxisName;

/// Sliding rings that hold the padding slot and the staging rows.
pub const RESERVED_RINGS: usize = 2;
/// Admitted Sliding owner slots. Each owner holds a 1,024-row E4M3 ring per Sliding layer.
pub const OWNERS: usize = Ao::SIZE - RESERVED_RINGS;
/// First staging row. Step row `i` stages at `STAGE_ROW0 + i` across the reserved rings.
pub const STAGE_ROW0: usize = 1;
/// Admitted Full pages per layer (allocator pages `0.. FULL_PAGES`).
pub const FULL_PAGES: usize = Kp::SIZE - 2;
/// The physical Full page padding rows publish into.
pub const FULL_PAD_PAGE: usize = Kp::SIZE - 1;
/// Largest step: the biggest token bucket.
pub const MAX_STEP_ROWS: usize = Tok1536::SIZE;

const SLIDING_ROW_BYTES: usize = Ns::SIZE * Ds::SIZE;

const _: () = assert!(Rs::SIZE == Ao::SIZE * Ts::SIZE && Rf::SIZE == Kp::SIZE * Tf::SIZE, "flat pool axes");
const _: () = assert!(STAGE_ROW0 + MAX_STEP_ROWS <= RESERVED_RINGS * Ts::SIZE, "the staging rows fit the reserved rings");

/// The Sliding pool row of an owner's position in its ring.
pub fn sliding_ring_row(owner: usize, position: usize) -> usize {
    debug_assert!(owner < OWNERS);
    (owner + RESERVED_RINGS) * Ts::SIZE + position % Ts::SIZE
}

/// The Full pool row of row `row` of allocator page `page`.
pub fn full_page_row(page: usize, row: usize) -> usize {
    debug_assert!(page < FULL_PAGES && row < Tf::SIZE);
    (page + 1) * Tf::SIZE + row
}

/// One step row as the attention tables see it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttnRow<'a> {
    /// Request policy, retained on every prompt and decode row.
    /// `None` on a padding row.
    pub owner: Option<usize>,
    pub position: usize,
    /// The owner's first position in this step and the step row holding it: the owner's rows
    /// of the step are consecutive, row `first_row + (q - first_position)` holds position `q`.
    pub first_position: usize,
    pub first_row: usize,
    /// The owner's Full pages (empty on a padding row).
    pub pages: &'a [usize],
}

impl AttnRow<'_> {
    pub const PADDING: AttnRow<'static> = AttnRow { owner: None, position: 0, first_position: 0, first_row: 0, pages: &[] };
}

/// Byte offset into a Sliding E4M3 pool.
fn sliding_row_bytes(row: usize) -> i32 {
    debug_assert!(row < Rs::SIZE);
    (row * SLIDING_ROW_BYTES) as i32
}

/// Staging offsets in bytes for the step's rows.
pub fn sliding_stage(n: usize) -> Vec<i32> {
    assert!(n <= MAX_STEP_ROWS);
    (0..n).map(|i| sliding_row_bytes(STAGE_ROW0 + i)).collect()
}

/// Publication offsets in bytes into each owner's ring. Padding rows rewrite their staging rows.
pub fn sliding_publish(rows: &[AttnRow]) -> Vec<i32> {
    rows.iter()
        .enumerate()
        .map(|(i, row)| match row.owner {
            Some(owner) => sliding_row_bytes(sliding_ring_row(owner, row.position)),
            None => sliding_row_bytes(STAGE_ROW0 + i),
        })
        .collect()
}

/// Full pool rows for publication. Padding rows write the padding page.
fn full_publication_rows<'a>(rows: &'a [AttnRow<'_>]) -> impl Iterator<Item = usize> + 'a {
    rows.iter().enumerate().map(|(i, row)| match row.owner {
        Some(_) => {
            let page = *row.pages.get(row.position / Tf::SIZE).expect("position has a Full page");
            assert!(page < FULL_PAGES, "Full page out of range");
            full_page_row(page, row.position % Tf::SIZE)
        }
        None => FULL_PAD_PAGE * Tf::SIZE + i % Tf::SIZE,
    })
}

/// Publication byte offsets for E4M3 Full KV.
pub fn full_publish_e4m3(rows: &[AttnRow]) -> Vec<i32> {
    full_publication_rows(rows).map(|row| (row * Df::SIZE) as i32).collect()
}
