//! The DSL planner selects E4M3 attention kernels and builds their lookup tables.
//! `attention` uploads the tables once per step and launches each kernel over
//! its region of the family's row space.
//!
//! A row's output depends on its position and visible keys.
//! - Decode kernels serve decode rows and Full prompt rows at or above `FULL_PREFILL_REACH`.
//!   Prefill kernels serve the other prompt rows.
//! - A decode launch holds 8 rows, sorted by visible count, longest first, and runs the smallest window covering
//!   its longest row. A list slot `j` of a window `W` is slice `j % 64`, index `j / 64`. Sliding lists are
//!   right-aligned: a row of `n` visible entries holds entry `k` (oldest first) at slot `W - n + k`, so its newest
//!   entry sits at the last slot (slice 63, index `Pi - 1`) in every window, which a program that places the newest
//!   K/V at compile time needs. Full lists are left-aligned (entry `k` at slot `k`). A slice holds the same entries in every
//!   window variant, behind or after its masked slots, which add exact zeros.
//! - A prefill kernel's row `i` is position `base + i`, and each 128-row block sees a window of positions in position
//!   order. Rows outside the chunk are dummy rows; positions outside the owner's live history (before 0, before the
//!   Sliding ring's oldest, after the step's rows) read spread-out finite pool rows and are masked for every real
//!   row.
//!
//! **Kernels and pieces.** Sliding uses the c512 band kernel at positions at or above `WINDOW`
//! and the c512 W128 or W384 causal kernel below it. Full below position 2,048 uses c512 causal
//! kernels with windows in multiples of 256, or c1024 at W1024 and W2048 for aligned rounds.
//!
//! **Row space.** Kernel row `j` of a launch at row-space row `s` reads and writes rows `s + 2 j` (head half 0)
//! and `s + 2 j + 1` (half 1): a kernel row is a step row's two halves. Decode launches take 16 rows each, first.
//! Prefill launches follow, each region starting so that its real rows begin after every earlier region; its
//! leading dummy rows may overlap earlier regions, so prefill launches run in reverse placement order (each
//! launch's real rows are written last by itself), then the decode launches.
//! Under the Full-side resident programs (`FullGroups`, `resident_full`) a Full plan keeps every prefill region at
//! or past the decode rows, because the program runs its decode launches before the layer's prefill launches.
//!
//! **Tables.** Per step, one i32 upload per family: a decode launch's `kv_off` `[8][Pi][Po % 32][Po / 32]` byte
//! offsets and its f32 `cap` `[8][Po][Pi]` (as bits); a prefill launch's offsets per round in the gather's walk
//! order. The tile masks (fixed by a causal kernel's base; all `+MAX` for a band kernel) and the edge caps are
//! constants of the kernels, uploaded once.
use std::collections::HashMap;

use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;
use crate::host::engine::attention::{RowSpace, WordsAt};
use crate::host::engine::attention_geometry::{Family, SINK_ROW, half_width};
/// The Sliding window the DSL kernels serve (`window`).
pub(crate) use crate::ops_dsl_attention::window::SLIDING_WINDOW as WINDOW;
use crate::host::engine::tables::{self, AttnRow, STAGE_ROW0};
use crate::ops_dsl_attention as dk;

type BoxError = Box<dyn std::error::Error>;

/// Decode windows (positions a launch covers per row), smallest first.
pub const SLIDING_DECODE: [usize; 2] = [256, WINDOW];
pub const FULL_DECODE: [usize; 5] = [512, 1024, 2048, 3072, 4096];
/// Full positions the planner serves: the largest Full decode window (a Full row at or past it has no plan).
pub const FULL_REACH: usize = FULL_DECODE[FULL_DECODE.len() - 1];
/// Full prompt rows below this position run the prefill kernels (their windows stop at 2,048: P fits a TRF lane).
pub const FULL_PREFILL_REACH: usize = 2048;
/// Rows of one decode launch (one per PE).
pub const DECODE_ROWS: usize = 8;

thread_local! {
    /// Logical step row -> physical dense/staging row. Cached owner-ring rows are unchanged.
    static STAGE_ROWS: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Change only the current step's staging addresses while planning the original logical spans.
/// Query grouping, row-space regions and launch ordering retain the unpermuted plan.
pub(crate) struct StageRows(Vec<usize>);

impl StageRows {
    pub fn new(rows: &[usize]) -> Self { Self(STAGE_ROWS.with(|r| r.replace(rows.to_vec()))) }
}

impl Drop for StageRows {
    fn drop(&mut self) { STAGE_ROWS.with(|r| *r.borrow_mut() = std::mem::take(&mut self.0)); }
}

thread_local! {
    /// The smallest Sliding decode window the planner picks on this thread (`SlidingDecodeFloor`; 0: the smallest
    /// covering window).
    static SLIDING_DECODE_FLOOR: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// While alive, this thread's plans run every Sliding decode launch at a window of at least `w` (the fused and resident
/// paths bind the plan's tables at their group window, `window::GROUP_WINDOW`). A row's bits are the same in every window
/// (module doc).
pub struct SlidingDecodeFloor(usize);

impl SlidingDecodeFloor {
    pub fn new(w: usize) -> Self {
        Self(SLIDING_DECODE_FLOOR.with(|f| f.replace(w)))
    }
}

impl Drop for SlidingDecodeFloor {
    fn drop(&mut self) {
        SLIDING_DECODE_FLOOR.with(|f| f.set(self.0));
    }
}

thread_local! {
    /// The prompt images this thread's Sliding plans place at a fixed row-space row (`PromptPlacement`).
    static PROMPT_PLACEMENT: std::cell::Cell<Option<(&'static [[u8; 3]], usize)>> = const { std::cell::Cell::new(None) };
}

/// While alive, this thread's Sliding plans run their prompt rows on the cheapest of `images` that holds the step's
/// blocks (else, as always, the cheapest image that holds them) and place an image of `images` at row-space row `row`.
/// The Sliding resident programs (`resident`) run those images' rounds in place at that row. A row's bits do not depend
/// on the row or on the image that holds it (`prompt_blocks` doc).
pub struct PromptPlacement(Option<(&'static [[u8; 3]], usize)>);

impl PromptPlacement {
    pub fn new(images: &'static [[u8; 3]], row: usize) -> Self {
        Self(PROMPT_PLACEMENT.with(|f| f.replace(Some((images, row)))))
    }
}

impl Drop for PromptPlacement {
    fn drop(&mut self) {
        PROMPT_PLACEMENT.with(|f| f.set(self.0));
    }
}

thread_local! {
    static FULL_GROUPS: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Plan the bucket's resident Full groups and reserve disjoint prefill rows.
pub struct FullGroups(Option<usize>);

impl FullGroups {
    pub fn new(bucket: usize) -> Self {
        Self(FULL_GROUPS.with(|f| f.replace(super::resident_full::program(bucket))))
    }
}

impl Drop for FullGroups {
    fn drop(&mut self) {
        FULL_GROUPS.with(|f| f.set(self.0));
    }
}

/// Words of one round's tile mask (`[8][4][16][32][8]` Sliding, `[2][4][64][32][8]` Full).
const TM_WORDS: usize = 131_072;
/// Words of a Sliding edge cap `[8][4][16][128][8]` and of the Full `cap_hi` `[2][4][64][128][2]`.
const SLIDING_CAP_WORDS: usize = 524_288;
const FULL_CAP_WORDS: usize = 131_072;
const CAP_ATTEND: f32 = f32::MAX;
const CAP_MASKED: f32 = -1e30;

/// One E4M3 attention kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Kernel {
    /// `DECODE_ROWS` rows over windows of `w` positions.
    Decode { family: Family, w: usize },
    /// Sliding prefill of 512 rows. Each block sees `WINDOW + 128` positions, with edge tiles clipped per row.
    Band,
    /// Causal prefill of `c` rows at the fixed base `base()`: each round's window is its last block's whole history.
    Causal { family: Family, c: usize, w: usize },
    /// Sliding prompt attention of a step's blocks in one launch (`ops_dsl_attention::prompt_blocks`): rounds of four
    /// 128-row blocks, round `r` over windows of `tiles[r]` tiles of 128 (descending; 0 past the last round).
    Prompt { tiles: [u8; 3] },
}

impl Kernel {
    /// The attention catalog preloaded at startup.
    pub fn all() -> Vec<Kernel> {
        let mut all: Vec<Kernel> = SLIDING_DECODE.iter().map(|&w| Kernel::Decode { family: Family::Sliding, w }).collect();
        all.extend(FULL_DECODE.iter().map(|&w| Kernel::Decode { family: Family::Full, w }));
        all.extend(SLIDING_PREFILL_CLASSES);
        all.extend([
            Kernel::Causal { family: Family::Full, c: 1024, w: 1024 },
            Kernel::Causal { family: Family::Full, c: 1024, w: 2048 },
        ]);
        all.extend((256..=2048).step_by(256).map(|w| Kernel::Causal { family: Family::Full, c: 512, w }));
        all.extend(prompt_images().into_iter().map(|tiles| Kernel::Prompt { tiles }));
        all
    }

    pub fn family(&self) -> Family {
        match *self {
            Kernel::Decode { family, .. } | Kernel::Causal { family, .. } => family,
            Kernel::Band | Kernel::Prompt { .. } => Family::Sliding,
        }
    }

    /// Kernel rows (each a step row's 16 heads).
    pub fn rows(&self) -> usize {
        match *self {
            Kernel::Decode { .. } => DECODE_ROWS,
            Kernel::Band => 512,
            Kernel::Causal { c, .. } => c,
            Kernel::Prompt { tiles } => 512 * prompt_rounds(tiles),
        }
    }

    /// Each round's window (prefill kernels; 512 rows per round).
    pub fn windows(&self) -> Vec<usize> {
        match *self {
            Kernel::Decode { .. } => Vec::new(),
            Kernel::Band => vec![128 * dk::prompt_blocks::TILES],
            Kernel::Causal { c: 512, w, .. } => vec![w],
            Kernel::Causal { family: Family::Full, w: 2048, .. } => vec![1536, 2048],
            Kernel::Causal { .. } => vec![512, 1024],
            Kernel::Prompt { tiles } => tiles[..prompt_rounds(tiles)].iter().map(|&t| 128 * t as usize).collect(),
        }
    }

    /// The position of a causal kernel's row 0 (its tile masks are built for it); a band kernel takes any base.
    pub fn base(&self) -> Option<isize> {
        match *self {
            Kernel::Causal { c: 512, w, .. } => Some(w as isize - 512),
            Kernel::Causal { family: Family::Full, w: 2048, .. } => Some(1024),
            Kernel::Causal { .. } => Some(0),
            _ => None,
        }
    }

    /// Words of one round's offsets for a window of `w`.
    fn offset_words(&self, w: usize) -> usize {
        match self.family() {
            Family::Sliding => 32 * w,
            Family::Full => 8 * w,
        }
    }
}

/// Estimate prompt cycles from tiles and edge recomputation to rank fitting images.
fn prompt_cycles(tiles: [u8; 3]) -> u64 {
    20_000
        + tiles[..prompt_rounds(tiles)]
            .iter()
            .map(|&t| 5_000 + 6_500 * t as u64 + if t > 1 { 2_100 * (1 + (t as usize == dk::prompt_blocks::TILES) as u64) } else { 0 })
            .sum::<u64>()
}

/// Rounds of a prompt image.
fn prompt_rounds(tiles: [u8; 3]) -> usize {
    tiles.iter().take_while(|&&t| t > 0).count()
}

/// The prompt images in the build (`ops_dsl_attention::prompt_blocks::images`), as round tiles.
fn prompt_images() -> Vec<[u8; 3]> {
    dk::prompt_blocks::images()
        .into_iter()
        .map(|(t, _)| {
            let mut tiles = [0u8; 3];
            for (i, &x) in t.iter().enumerate() {
                tiles[i] = x as u8;
            }
            tiles
        })
        .collect()
}

// Opt discovers typed launch sites, while serving binds the same kernels with raw buffers.
macro_rules! dsl_catalog {
    ($path:ident, $embed:ident, [$($pat:pat => [$($m:ident)::+] $f:ident),* $(,)?]) => {
        fn $path(kernel: Kernel) -> Option<&'static str> {
            match kernel {
                $($pat => Some($($m)::+::$f.path()),)*
                _ => None,
            }
        }

        async fn $embed() {
            $(let _ = launch($($m)::+::$f, embedding_args()).await;)*
        }
    };
}

#[inline(never)]
pub(crate) fn embedding_args<T>() -> T {
    unreachable!("typed launch site for Opt registry embedding")
}

// The Sliding kernels over the E4M3 Sliding pools (`int8::SlidingKv`).
dsl_catalog!(path_fp8_s0, embed_sliding, [
    Kernel::Decode { family: Family::Sliding, w: 256 } => [dk::decode::w256_fp8] swa_decode_fp8_w256,
    Kernel::Decode { family: Family::Sliding, w: 384 } => [dk::decode::w384_fp8] swa_decode_fp8_w384,
    Kernel::Band => [dk::prefill::band512_fp8] swa_prefill_fp8_band_c512,
    Kernel::Causal { family: Family::Sliding, c: 512, w: 128 } => [dk::prefill::w128_fp8] swa_prefill_fp8_w128_c512,
    Kernel::Causal { family: Family::Sliding, c: 512, w: 384 } => [dk::prefill::w384_fp8] swa_prefill_fp8_w384_c512,
]);

/// The Sliding E4M3 kernels' registry paths: the catalog's, and the prompt images'.
fn path_fp8_s(kernel: Kernel) -> Option<&'static str> {
    match kernel {
        Kernel::Prompt { tiles } => {
            dk::prompt_blocks::images().into_iter().find(|(t, _)| t.iter().map(|&x| x as u8).eq(tiles[..prompt_rounds(tiles)].iter().copied())).map(|(_, p)| p)
        }
        _ => path_fp8_s0(kernel),
    }
}

dsl_catalog!(path_fp8_f, embed_full, [
    Kernel::Decode { family: Family::Full, w: 512 } => [dk::decode_full::f512_fp8] full_decode_fp8_w512,
    Kernel::Decode { family: Family::Full, w: 1024 } => [dk::decode_full::f1024_fp8] full_decode_fp8_w1024,
    Kernel::Decode { family: Family::Full, w: 2048 } => [dk::decode_full::f2048_fp8] full_decode_fp8_w2048,
    Kernel::Decode { family: Family::Full, w: 3072 } => [dk::decode_full::f3072_fp8] full_decode_fp8_w3072,
    Kernel::Decode { family: Family::Full, w: 4096 } => [dk::decode_full::f4096_fp8] full_decode_fp8_w4096,
    Kernel::Causal { family: Family::Full, c: 1024, w: 1024 } => [dk::prefill_full::w1024c1024_fp8] full_prefill_fp8_w1024_c1024,
    Kernel::Causal { family: Family::Full, c: 1024, w: 2048 } => [dk::prefill_full::w2048c1024_fp8] full_prefill_fp8_w2048_c1024,
    Kernel::Causal { family: Family::Full, c: 512, w: 256 } => [dk::prefill_full::w256_fp8] full_prefill_fp8_w256_c512,
    Kernel::Causal { family: Family::Full, c: 512, w: 512 } => [dk::prefill_full::w512_fp8] full_prefill_fp8_w512_c512,
    Kernel::Causal { family: Family::Full, c: 512, w: 768 } => [dk::prefill_full::w768_fp8] full_prefill_fp8_w768_c512,
    Kernel::Causal { family: Family::Full, c: 512, w: 1024 } => [dk::prefill_full::w1024_fp8] full_prefill_fp8_w1024_c512,
    Kernel::Causal { family: Family::Full, c: 512, w: 1280 } => [dk::prefill_full::w1280_fp8] full_prefill_fp8_w1280_c512,
    Kernel::Causal { family: Family::Full, c: 512, w: 1536 } => [dk::prefill_full::w1536_fp8] full_prefill_fp8_w1536_c512,
    Kernel::Causal { family: Family::Full, c: 512, w: 1792 } => [dk::prefill_full::w1792_fp8] full_prefill_fp8_w1792_c512,
    Kernel::Causal { family: Family::Full, c: 512, w: 2048 } => [dk::prefill_full::w2048_fp8] full_prefill_fp8_w2048_c512,
]);

/// Resolve an E4M3 attention kernel to its registry path.
fn path(kernel: Kernel) -> Result<&'static str, String> {
    let path = match kernel.family() {
        Family::Sliding => path_fp8_s(kernel),
        Family::Full => path_fp8_f(kernel),
    };
    path.ok_or_else(|| format!("{kernel:?}: no E4M3-KV DSL kernel in the catalog"))
}

// ---- planning ----------------------------------------------------------------------------------------------

/// One launch of a plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Launch {
    pub kernel: Kernel,
    /// First row-space row of the launch's region (`2 x kernel.rows()` rows).
    pub row: usize,
    /// Word offsets of the launch's per-step tables in `Plan::words`: decode `[kv_off, cap]`, prefill one offset
    /// table per round.
    pub tables: Vec<usize>,
}

/// One family's DSL launches for one step and the tables they read.
#[derive(Clone, Debug, Default)]
pub struct Plan {
    /// `[T, Hh]` byte offsets into the row space (padding rows use the sink row).
    pub rows: Vec<i32>,
    /// In run order.
    pub launches: Vec<Launch>,
    /// The step's table words.
    pub words: Vec<i32>,
    /// Six pairs of full-window Sliding decode groups, including masked sink groups.
    pub paired_decode: Vec<[usize; 3]>,
    /// Full resident program and the number of launches it runs at the end of the plan.
    pub full_groups: Option<(usize, usize)>,
    /// Same row-space layout, with resident-only Full widths widened to loaded
    /// standalone kernels. Its tables are included in this plan's one upload.
    pub full_standalone: Option<Vec<Launch>>,
}

impl Plan {
    /// Each pair describes a dense row's query/output halves; the attention region it names stays fixed.
    pub(crate) fn permute_step_rows(&mut self, physical_to_logical: &[usize]) {
        assert_eq!(self.rows.len(), physical_to_logical.len() * 2);
        self.rows = physical_to_logical.iter().flat_map(|&i| [self.rows[2 * i], self.rows[2 * i + 1]]).collect();
    }

    fn add_full_standalone(&mut self) -> Result<(), String> {
        if !self.launches.iter().any(|l| matches!(l.kernel,
            Kernel::Decode { family: Family::Full, w } if !FULL_DECODE.contains(&w))) {
            return Ok(());
        }
        let mut standalone = self.launches.clone();
        for launch in &mut standalone {
            let Kernel::Decode { family: Family::Full, w } = launch.kernel else { continue };
            if FULL_DECODE.contains(&w) { continue; }
            let wide = *FULL_DECODE.iter().find(|&&n| n >= w)
                .ok_or_else(||format!("no standalone Full window covers resident width {w}"))?;
            let (old_pi, new_pi) = (w / 64, wide / 64);
            let (old_off,old_cap) = (launch.tables[0],launch.tables[1]);
            let off = self.words.len().next_multiple_of(256);
            let cap = off + DECODE_ROWS * wide;
            self.words.resize(cap + DECODE_ROWS * wide, 0);
            self.words[cap..cap + DECODE_ROWS * wide].fill(CAP_MASKED.to_bits() as i32);
            for row in 0..DECODE_ROWS {
                for tile in 0..old_pi {
                    for lane in 0..64 {
                        let old_index = ((row * old_pi + tile) * 32 + lane % 32) * 2 + lane / 32;
                        let new_index = ((row * new_pi + tile) * 32 + lane % 32) * 2 + lane / 32;
                        self.words[off + new_index] = self.words[old_off + old_index];
                        self.words[cap + (row * 64 + lane) * new_pi + tile] =
                            self.words[old_cap + (row * 64 + lane) * old_pi + tile];
                    }
                }
            }
            // New positions gather the immutable zero page and stay masked.
            // Query/output rows, launch order and all original cap bits remain.
            launch.kernel = Kernel::Decode { family: Family::Full, w: wide };
            launch.tables = vec![off,cap];
        }
        self.full_standalone = Some(standalone);
        Ok(())
    }
}

/// A step's rows grouped by owner: `(first step row, rows)`.
fn spans(rows: &[AttnRow]) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        if row.owner.is_none() {
            continue;
        }
        match out.last_mut() {
            Some((first, n)) if row.first_row == *first && rows[*first].owner == row.owner && *first + *n == i => *n += 1,
            _ => out.push((i, 1)),
        }
    }
    out
}

/// The pool row holding position `q` of `row`'s owner while the step's launches run:
/// the step's own Sliding positions are in the staging rows, earlier ones in the owner's ring; Full positions in
/// the owner's pages.
fn pool_row(family: Family, row: &AttnRow, q: usize) -> usize {
    match family {
        Family::Sliding if q >= row.first_position => {
            let logical = row.first_row + (q - row.first_position);
            STAGE_ROW0 + STAGE_ROWS.with_borrow(|rows| if rows.is_empty() { logical } else { rows[logical] })
        },
        Family::Sliding => tables::sliding_ring_row(row.owner.expect("a valid row"), q),
        Family::Full => tables::full_page_row(*row.pages.get(q / Tf::SIZE).expect("position has a Full page"), q % Tf::SIZE),
    }
}

/// Visible positions of a row at `p`: their count (the window variant's need).
fn visible_count(family: Family, p: usize) -> usize {
    match family {
        Family::Sliding => (p + 1).min(WINDOW),
        Family::Full => p + 1,
    }
}

/// Spread-out pool rows for masked entries: one hot address would stall the gathers.
struct Spread {
    rows: usize,
    n: usize,
}

impl Spread {
    fn new(family: Family) -> Self {
        let rows = match family {
            Family::Sliding => Rs::SIZE,
            Family::Full => Rf::SIZE,
        };
        Self { rows, n: 0 }
    }

    fn next(&mut self) -> usize {
        self.n += 1;
        (self.n * 7919) % self.rows
    }

}

/// The second owner of a paired Full c1024 launch.  The first owner's coordinates live on `Piece`. This record lets the
/// layout and table builder carry the second request without changing the
/// Sliding ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PairPart {
    first: usize,
    start: usize,
    len: usize,
    /// Position of kernel row zero for the shared c1024 image.
    base: isize,
    span_start: usize,
    span_end: usize,
}

/// One prefill launch's rows: `len` step rows from `first` (positions `start..`), kernel row `i` at position
/// `base + i`, within the owner's span (positions `span_start.. span_end`).  A Full c1024 piece may carry one
/// additional 512-row owner in the image's second round; the two owner spans are kept disjoint in the table builder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Piece {
    first: usize,
    start: usize,
    len: usize,
    kernel: Kernel,
    base: isize,
    /// Span identity within this step; pieces from one request must never pair.
    owner: usize,
    second: Option<PairPart>,
}

// Use the step families' Sliding prefill members. W384 masks the extra leading tiles on short causal spans.
const SLIDING_PREFILL_CLASSES: [Kernel; 3] = [
    Kernel::Causal { family: Family::Sliding, c: 512, w: 128 },
    Kernel::Causal { family: Family::Sliding, c: 512, w: 384 },
    Kernel::Band,
];

/// Sliding pieces of a prompt span of `n` rows at `start` (step rows from `first`).
fn sliding_pieces(first: usize, start: usize, n: usize) -> Vec<Piece> {
    let mut out = Vec::new();
    let end = start + n;
    let [w128, w384, band] = SLIDING_PREFILL_CLASSES;
    let mut k = start;
    let mut push = |k: usize, len: usize, kernel: Kernel, base: isize| out.push(Piece { first: first + (k - start), start: k, len, kernel, base, owner: usize::MAX, second: None });
    while k < end {
        if k >= WINDOW {
            let c = band.rows();
            // Keep the original band rounds' bases, including a partial first round.
            let stop = end.min(k + 2 * c);
            let tail = stop.saturating_sub(c);
            if tail > k {
                push(k, tail - k, band, tail as isize - c as isize);
                k = tail;
            }
            push(k, stop - k, band, stop as isize - c as isize);
            k = stop;
            continue;
        }
        let stop = end.min(WINDOW);
        let causal = if stop <= 128 { w128 } else { w384 };
        push(k, stop - k, causal, causal.base().unwrap());
        k = stop;
    }
    out
}

/// Use the faster c1024 image for a full first round at W512 or second round at W2048.
/// The c512 image is faster for its W1024 piece.
fn full_equal_row_promotion(k: usize, end: usize, w: usize) -> Option<Kernel> {
    if end - k != 512 {
        return None;
    }
    match (k, end, w) {
        (0, 512, 512) => Some(Kernel::Causal { family: Family::Full, c: 1024, w: 1024 }),
        (k, _, 2048) if k >= 1024 => Some(Kernel::Causal { family: Family::Full, c: 1024, w: 2048 }),
        _ => None,
    }
}

fn full_pieces(first: usize, start: usize, n: usize) -> Vec<Piece> {
    let mut out = Vec::new();
    let end = start + n;
    debug_assert!(end <= FULL_PREFILL_REACH);
    let mut k = start;
    let mut push = |k: usize, len: usize, kernel: Kernel, base: isize| out.push(Piece { first: first + (k - start), start: k, len, kernel, base, owner: usize::MAX, second: None });
    while k < end {
        let w = end.next_multiple_of(256);
        if w <= k + 512 {
            push(k, end - k, Kernel::Causal { family: Family::Full, c: 512, w }, w as isize - 512);
            k = end;
            continue;
        }
        let w1 = k / 256 * 256 + 512;
        push(k, w1 - k, Kernel::Causal { family: Family::Full, c: 512, w: w1 }, w1 as isize - 512);
        k = w1;
    }
    out
}

/// Join aligned c512 Full pieces from different spans into one c1024 image.
///
/// The c1024 kernels have two fixed rounds.  Their natural row bases are
/// `(base, base + 512)`, so joining arbitrary pieces would require a
/// per-entry mask (and would change the arithmetic at a leading partial
/// tile).  We therefore pair only complete 512-row pieces whose starts are
/// exactly those two bases.  Each round's offset table is then built from a
/// different owner and the output rows are scattered independently below.
fn pair_full_pieces(pieces: &mut Vec<(Piece, usize, usize)>) {
    let mut used = vec![false; pieces.len()];
    let mut paired = Vec::with_capacity(pieces.len());
    for i in 0..pieces.len() {
        if used[i] {
            continue;
        }
        let (mut p, span_start, span_end) = pieces[i];
        let Some((want_w, want_start, pair_kernel)) = (match p.kernel {
            Kernel::Causal { family: Family::Full, c: 512, w: 512 } if p.len == 512 && p.start == p.base as usize =>
                Some((1024, p.base as usize + 512, Kernel::Causal { family: Family::Full, c: 1024, w: 1024 })),
            Kernel::Causal { family: Family::Full, c: 512, w: 1536 } if p.len == 512 && p.start == p.base as usize =>
                Some((2048, p.base as usize + 512, Kernel::Causal { family: Family::Full, c: 1024, w: 2048 })),
            _ => None,
        }) else {
            promote_unpaired_full_piece(&mut p);
            paired.push((p, span_start, span_end));
            continue;
        };
        let j = ((i + 1)..pieces.len()).find(|&j| {
            if used[j] { return false; }
            let (q, _, _) = pieces[j];
            matches!(q.kernel, Kernel::Causal { family: Family::Full, c: 512, w } if w == want_w)
                && q.len == 512
                && q.start == want_start
                && q.owner != p.owner
        });
        let Some(j) = j else {
            promote_unpaired_full_piece(&mut p);
            paired.push((p, span_start, span_end));
            continue;
        };
        let (q, q_span_start, q_span_end) = pieces[j];
        used[j] = true;
        p.kernel = pair_kernel;
        p.second = Some(PairPart {
            first: q.first,
            start: q.start,
            len: q.len,
            base: p.base,
            span_start: q_span_start,
            span_end: q_span_end,
        });
        paired.push((p, span_start, span_end));
    }
    *pieces = paired;
}

/// Use the faster c1024 image for an isolated W512 or W2048 half.
fn promote_unpaired_full_piece(p: &mut Piece) {
    let end = p.start + p.len;
    let w = end.next_multiple_of(256);
    if let Some(kernel) = full_equal_row_promotion(p.start, end, w) {
        p.kernel = kernel;
        p.base = kernel.base().expect("a promoted Full kernel has a base");
    }
}

/// One owner's rows of a step as the planner routes them: `n` step rows from `first`, positions `start..`, prompt
/// rows or decode rows. `plan` reads them off the step's rows (`spans`); the scheduler builds them for a step it is
/// packing (`schedule::fits`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepSpan {
    pub first: usize,
    pub start: usize,
    pub n: usize,
    pub prompt: bool,
}

/// One family's routing of a step's rows.
struct Routing {
    /// Decode rows and Full prompt rows beyond the prefill kernels' reach.
    decode: Vec<usize>,
    /// Pieces in placement order.
    pieces: Vec<(Piece, usize, usize)>,
    /// Sliding: the prompt rows `pieces` serve, as runs of consecutive positions (the prompt images' input).
    runs: Vec<Run>,
}

/// A run of one owner's consecutive Sliding prompt rows: `n` step rows from `first`, positions `start..`, in the
/// owner's span of positions `span.0.. span.1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Run {
    owner: usize,
    first: usize,
    start: usize,
    n: usize,
    span: (usize, usize),
}

/// Routes each span by family, kind and position.
fn route(family: Family, spans: &[StepSpan]) -> Routing {
    let mut r = Routing { decode: Vec::new(), pieces: Vec::new(), runs: Vec::new() };
    for (s, &StepSpan { first, start, n, prompt }) in spans.iter().enumerate() {
        if !prompt {
            r.decode.extend(first..first + n);
            continue;
        }
        let span = (start, start + n);
        match family {
            Family::Sliding => {
                r.pieces.extend(sliding_pieces(first, start, n).into_iter().map(|mut p| {
                    p.owner = s;
                    (p, span.0, span.1)
                }));
                r.runs.push(Run { owner: s, first, start, n, span });
            }
            Family::Full => {
                let prefill = n.min(FULL_PREFILL_REACH.saturating_sub(start));
                if prefill > 0 {
                    r.pieces.extend(full_pieces(first, start, prefill).into_iter().map(|mut p| {
                        p.owner = s;
                        (p, span.0, span.1)
                    }));
                }
                r.decode.extend(first + prefill..first + n);
            }
        }
    }
    if family == Family::Full {
        // Pairing is directional (round 0 then round 1), so establish the
        // kernel order before looking for the second half.  Re-sort after the
        // pass because a joined piece changes its kernel from c512 to c1024.
        r.pieces.sort_by_key(|(p, ..)| p.kernel);
        pair_full_pieces(&mut r.pieces);
    }
    r.pieces.sort_by_key(|(p, ..)| p.kernel);
    r
}

/// Where a step's launches sit in the row space.
struct Layout {
    /// Rows of the decode launches, first: `2 x DECODE_ROWS` per launch.
    decode_rows: usize,
    /// Per piece in placement order: its region's first row and its first real row. Each region starts so that its
    /// real rows begin at or after the end of every earlier region; its leading dummy rows may overlap earlier
    /// regions (so prefill launches run in reverse placement order).
    regions: Vec<(usize, usize)>,
    /// Rows the launches take, bounded by `SINK_ROW`.
    row_space: usize,
}

fn layout(decode: usize, pieces: &[(Piece, usize, usize)], clear: bool) -> Layout {
    layout_above(decode, pieces, clear, 0)
}

/// `layout` with no region reaching below row `above` (a placed prompt image's end: the pieces run after the image's
/// program, so their leading dummy rows must not overwrite the image's rows).
fn layout_above(decode: usize, pieces: &[(Piece, usize, usize)], clear: bool, above: usize) -> Layout {
    let decode_rows = 2 * DECODE_ROWS * decode.div_ceil(DECODE_ROWS);
    let mut next_real = decode_rows.max(above);
    // `clear` (`FullGroups`): no region reaches below the decode rows.
    let floor = if clear { decode_rows } else { 0 }.max(above);
    let regions = pieces
        .iter()
        .map(|(p, ..)| {
            let lead = (p.start as isize - p.base) as usize;
            let region = next_real.saturating_sub(2 * lead).max(floor);
            next_real = region + 2 * p.kernel.rows();
            (region, region + 2 * lead)
        })
        .collect();
    Layout { decode_rows, regions, row_space: next_real.max(decode_rows) }
}

/// The row-space rows `plan` takes for `family` on a step of `spans` (step-row order):
/// the same routing and placement, without the tables. A step fits when this is at most `SINK_ROW`; the scheduler
/// packs no step past it (`schedule::fits`).
pub fn row_space_need(family: Family, spans: &[StepSpan]) -> usize {
    let routing = route(family, spans);
    layout(routing.decode.len(), &routing.pieces, false).row_space
}

/// One 128-row block of a prompt image: its run, its first position and the tiles it needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PromptBlock {
    run: usize,
    p0: usize,
    need: usize,
}

/// The runs' blocks, most tiles first (stable): below `WINDOW` aligned to 128 positions, from `WINDOW` on starting at
/// the run's first position (or `WINDOW`).
fn prompt_blocks(runs: &[Run]) -> Vec<PromptBlock> {
    let mut out = Vec::new();
    for (i, r) in runs.iter().enumerate() {
        let end = r.start + r.n;
        let mut p = r.start / 128 * 128;
        while p < end.min(WINDOW) {
            out.push(PromptBlock { run: i, p0: p, need: dk::prompt_blocks::tiles_needed(p) });
            p += 128;
        }
        let mut p = r.start.max(WINDOW);
        while p < end {
            out.push(PromptBlock { run: i, p0: p, need: dk::prompt_blocks::tiles_needed(p) });
            p += 128;
        }
    }
    out.sort_by_key(|b| std::cmp::Reverse(b.need));
    out
}

/// The cheapest prompt image whose rounds hold `blocks` (four per round, in order), if any.
fn prompt_image(blocks: &[PromptBlock]) -> Option<[u8; 3]> {
    prompt_images().into_iter().filter(|&t| prompt_holds(blocks, t)).min_by_key(|&t| prompt_cycles(t))
}

/// Whether the prompt image of round tiles `t` holds `blocks` (four per round, in order).
fn prompt_holds(blocks: &[PromptBlock], t: [u8; 3]) -> bool {
    let needs: Vec<usize> = blocks.chunks(4).map(|c| c[0].need).collect();
    prompt_rounds(t) >= needs.len() && needs.iter().zip(t).all(|(&n, w)| w as usize >= n)
}

/// The runs a program image of `images` holds, most rows first, and that image; `None` when no run fits or all do.
fn split_runs(runs: &[Run], images: &[[u8; 3]]) -> Option<(Vec<Run>, [u8; 3])> {
    let mut order: Vec<usize> = (0..runs.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(runs[i].n));
    let mut taken = vec![false; runs.len()];
    let mut best: Option<[u8; 3]> = None;
    for &i in &order {
        taken[i] = true;
        let sub: Vec<Run> = runs.iter().zip(&taken).filter(|(_, t)| **t).map(|(r, _)| *r).collect();
        let blocks = prompt_blocks(&sub);
        match images.iter().copied().filter(|&t| prompt_holds(&blocks, t)).min_by_key(|&t| prompt_cycles(t)) {
            Some(t) => best = Some(t),
            None => taken[i] = false,
        }
    }
    let tiles = best?;
    let sub: Vec<Run> = runs.iter().zip(&taken).filter(|(_, t)| **t).map(|(r, _)| *r).collect();
    // The cheapest image that holds the final selection (selection only grew while some image held it).
    let blocks = prompt_blocks(&sub);
    let tiles = images.iter().copied().filter(|&t| prompt_holds(&blocks, t)).min_by_key(|&t| prompt_cycles(t)).unwrap_or(tiles);
    (sub.len() < runs.len()).then_some((sub, tiles))
}

/// Places the prompt image at row-space row `region`: round `r`'s block `b` (blocks in order, four per round) at
/// kernel rows `512 r + 128 b..`, its tables (offsets, tile-mask and cap keys per round) appended to the plan's
/// words, and the step rows the blocks hold mapped to their kernel rows.
fn place_prompt(plan: &mut Plan, rows: &[AttnRow], runs: &[Run], blocks: &[PromptBlock], tiles: [u8; 3], region: usize, row_bytes: i32) -> Launch {
    let kernel = Kernel::Prompt { tiles };
    let spread = |i: usize| (i * 7919 + 4099) % Rs::SIZE;
    let mut tables = Vec::new();
    for (r, &t) in tiles[..prompt_rounds(tiles)].iter().enumerate() {
        let group: Vec<&PromptBlock> = blocks.iter().skip(4 * r).take(4).collect();
        let entries: Vec<Box<dyn Fn(usize) -> usize + '_>> = group
            .iter()
            .map(|b| -> Box<dyn Fn(usize) -> usize + '_> {
                let run = runs[b.run];
                let row = &rows[run.first];
                // The owner's live positions: the ring holds the `WINDOW` (of its 1,024 rows) before its span, the staging
                // rows the span.
                let lo = run.span.0.saturating_sub(WINDOW);
                Box::new(move |p: usize| if p >= lo && p < run.span.1 { pool_row(Family::Sliding, row, p) } else { spread(p + 1) })
            })
            .collect();
        let bs: Vec<dk::prompt_blocks::Block<'_>> =
            group.iter().zip(&entries).map(|(b, e)| dk::prompt_blocks::Block { p0: b.p0, entry: e.as_ref() }).collect();
        let (off, tm, cap) = dk::prompt_blocks::round_tables(t as usize, [bs.first(), bs.get(1), bs.get(2), bs.get(3)], &|i| spread(i + 7 * r));
        for words in [off, tm, cap] {
            tables.push(plan.words.len());
            plan.words.extend(words);
        }
        for (b, blk) in group.iter().enumerate() {
            let run = runs[blk.run];
            for j in 0..128 {
                let p = blk.p0 + j;
                if p < run.start || p >= run.start + run.n {
                    continue;
                }
                let step_row = run.first + (p - run.start);
                let kernel_row = 512 * r + 128 * b + j;
                plan.rows[2 * step_row] = (region + 2 * kernel_row) as i32 * row_bytes;
                plan.rows[2 * step_row + 1] = (region + 2 * kernel_row + 1) as i32 * row_bytes;
            }
        }
    }
    Launch { kernel, row: region, tables }
}

/// Plan one family's attention. A row without a prompt flag counts as prompt.
/// Return an error if a window or row-space region cannot fit.
pub fn plan(family: Family, rows: &[AttnRow], prompt: &[bool]) -> Result<Plan, String> {
    let half = half_width(family);
    let row_bytes = (half * 2) as i32;
    let mut plan = Plan { rows: vec![SINK_ROW as i32 * row_bytes; rows.len() * 2], ..Plan::default() };
    let mut spread = Spread::new(family);

    let step_spans: Vec<StepSpan> =
        spans(rows).into_iter().map(|(first, n)| StepSpan { first, start: rows[first].position, n, prompt: prompt.get(first).copied().unwrap_or(true) }).collect();
    let Routing { mut decode, pieces, runs } = route(family, &step_spans);
    // Use a prompt image when it covers the Sliding prompt blocks. Otherwise use separate pieces.
    let prompt_choice = if family == Family::Sliding && !runs.is_empty() {
        let blocks = prompt_blocks(&runs);
        // Under a resident step's placement, the cheapest of its programs' images that holds the blocks: the program
        // then runs the rounds in place, so consecutive layers reuse its image.
        let placed_image = PROMPT_PLACEMENT.with(|f| f.get()).and_then(|(images, _)| {
            images.iter().copied().filter(|&t| prompt_holds(&blocks, t)).min_by_key(|&t| prompt_cycles(t))
        });
        placed_image.or_else(|| prompt_image(&blocks)).map(|tiles| (blocks, tiles))
    } else {
        None
    };
    // Split (Tok1536): no program image holds every run, so the image takes the runs it holds and the other runs keep
    // their pieces, placed above the image (run after the program, `layout_above`).
    let mut split_runs_img: Option<Vec<Run>> = None;
    let mut split_pieces: Vec<(Piece, usize, usize)> = Vec::new();
    let mut split_above = 0;
    let (mut prompt_choice, mut runs_for_image) = (prompt_choice, None::<Vec<Run>>);
    if prompt_choice.is_none() && family == Family::Sliding && !runs.is_empty() && rows.len() == 1536
        && let Some((images, row)) = PROMPT_PLACEMENT.with(|f| f.get())
        && let Some((sub, tiles)) = split_runs(&runs, images)
    {
        let decode_rows = 2 * DECODE_ROWS * decode.len().div_ceil(DECODE_ROWS);
        let above = row + 2 * Kernel::Prompt { tiles }.rows();
        let rest: Vec<(Piece, usize, usize)> = {
            let mut v = Vec::new();
            for r in runs.iter().filter(|r| !sub.contains(r)) {
                v.extend(sliding_pieces(r.first, r.start, r.n).into_iter().map(|mut p| {
                    p.owner = r.owner;
                    (p, r.span.0, r.span.1)
                }));
            }
            v.sort_by_key(|(p, ..)| p.kernel);
            v
        };
        if row >= decode_rows && layout_above(decode.len(), &rest, false, above).row_space <= SINK_ROW {
            let blocks = prompt_blocks(&sub);
            prompt_choice = Some((blocks, tiles));
            split_above = above;
            split_pieces = rest;
            runs_for_image = Some(sub.clone());
            split_runs_img = Some(sub);
        }
    }
    let pieces = if split_runs_img.is_some() { std::mem::take(&mut split_pieces) } else if prompt_choice.is_some() { Vec::new() } else { pieces };
    let mut placed = layout_above(decode.len(), &pieces, false, split_above);
    if placed.row_space > SINK_ROW {
        return Err(format!("{family:?} DSL launches need {} row-space rows; {SINK_ROW} exist", placed.row_space));
    }
    // Rows longest first, eight per decode launch.
    let count = |i: usize| match family {
        Family::Sliding => (rows[i].position + 1).min(WINDOW),
        Family::Full => visible_count(family, rows[i].position),
    };
    let oldest = |i: usize| rows[i].position + 1 - count(i);
    decode.sort_by_key(|&i| std::cmp::Reverse(count(i)));
    let windows: &[usize] = match family {
        Family::Sliding => &SLIDING_DECODE,
        Family::Full => &FULL_DECODE,
    };
    let floor = match family {
        Family::Sliding => SLIDING_DECODE_FLOOR.with(|f| f.get()),
        Family::Full => 0,
    };
    let kv_row_bytes = match family {
        Family::Sliding => Ns::SIZE * Ds::SIZE,
        Family::Full => Df::SIZE,
    };
    let right_aligned = family == Family::Sliding;
    let mut block_w = Vec::with_capacity(decode.len().div_ceil(DECODE_ROWS));
    for part in decode.chunks(DECODE_ROWS) {
        let longest = count(part[0]);
        let window = windows.iter().copied().find(|&w| w >= longest.max(floor))
            .ok_or_else(|| format!("no {family:?} DSL decode window holds {longest} positions"))?;
        block_w.push(window);
    }
    // The Full-side resident program (`FullGroups`): its blocks at their groups' windows, their launches last.
    let mut in_program = vec![false; block_w.len()];
    if family == Family::Full
        && !block_w.is_empty()
        && let Some(program) = FULL_GROUPS.with(|f| f.get())
    {
        let windows = decode.chunks(DECODE_ROWS).map(|part| count(part[0]));
        let at = super::resident_full::place(program, windows);
        let served = at.iter().flatten().count();
        if served != 0 {
            let clear = layout(decode.len(), &pieces, true);
            if clear.row_space <= SINK_ROW {
                placed = clear;
                for (b, w) in at.into_iter().enumerate() {
                    if let Some(w) = w {
                        block_w[b] = w;
                        in_program[b] = true;
                    }
                }
                plan.full_groups = Some((program, served));
            }
        }
    }
    let mut decode_launches = Vec::new();
    let mut program_launches = Vec::new();
    let mut cursor = 0;
    for (b, part) in decode.chunks(DECODE_ROWS).enumerate() {
        let w = block_w[b];
        let kernel = Kernel::Decode { family, w };
        let pi = w / 64;
        let off_at = plan.words.len();
        let cap_at = off_at + DECODE_ROWS * w;
        plan.words.resize(off_at + 2 * DECODE_ROWS * w, 0);
        for r in 0..DECODE_ROWS {
            let token = part.get(r).copied();
            let (entries, oldest) = match token {
                Some(t) => (count(t), oldest(t)),
                None => (0, 0),
            };
            // The row's entries fill slots `lead.. lead + entries`, oldest first; the other slots are masked.
            let lead = if right_aligned { w - entries } else { 0 };
            for i in 0..pi {
                for s in 0..64 {
                    let slot = i * 64 + s;
                    let attend = (lead..lead + entries).contains(&slot);
                    let pool = match token {
                        Some(t) if attend => pool_row(family, &rows[t], oldest + (slot - lead)),
                        _ => spread.next(),
                    };
                    plan.words[off_at + ((r * pi + i) * 32 + s % 32) * 2 + s / 32] = (pool * kv_row_bytes) as i32;
                    plan.words[cap_at + (r * 64 + s) * pi + i] = (if attend { CAP_ATTEND } else { CAP_MASKED }).to_bits() as i32;
                }
            }
            if let Some(t) = token {
                plan.rows[2 * t] = (cursor + 2 * r) as i32 * row_bytes;
                plan.rows[2 * t + 1] = (cursor + 2 * r + 1) as i32 * row_bytes;
            }
        }
        let launch = Launch { kernel, row: cursor, tables: vec![off_at, cap_at] };
        if in_program[b] { program_launches.push(launch) } else { decode_launches.push(launch) }
        cursor += 2 * DECODE_ROWS;
    }
    decode_launches.extend(program_launches);
    debug_assert_eq!(cursor, placed.decode_rows, "the layout's decode rows");

    // Prefill launches, placed by kernel (one program switch per kernel) at their `layout` regions.
    let mut prefill_launches = Vec::new();
    for ((p, span_start, span_end), &(region, real)) in pieces.into_iter().zip(&placed.regions) {
        for t in 0..p.len {
            plan.rows[2 * (p.first + t)] = (real + 2 * t) as i32 * row_bytes;
            plan.rows[2 * (p.first + t) + 1] = (real + 2 * t + 1) as i32 * row_bytes;
        }
        // A paired c1024 launch owns the second round's 512 rows on behalf
        // of another request.  Scatter those rows to the same row-space
        // region before the device launch; the engine's qin/out buffers then
        // naturally gather and return each request's rows independently.
        if let Some(second) = p.second {
            debug_assert_eq!(p.kernel.rows(), 1024, "a paired piece must use c1024");
            for t in 0..second.len {
                let kernel_row = 512 + t;
                plan.rows[2 * (second.first + t)] = (real + 2 * kernel_row) as i32 * row_bytes;
                plan.rows[2 * (second.first + t) + 1] = (real + 2 * kernel_row + 1) as i32 * row_bytes;
            }
        }
        // The owner's live positions: the Sliding ring holds the `WINDOW` (of its 1,024 rows) before the span, the staging
        // rows the span.
        let mut tables = Vec::new();
        for (r, &w) in p.kernel.windows().iter().enumerate() {
            // Full c1024's two rounds are the two independent owner blocks.
            // For an unpaired promotion, the primary owner supplies both
            // rounds and the unused rows are masked by the existing span
            // bounds.  Sliding pieces never carry `second` and retain the
            // original one-owner route.
            let (owner_first, owner_base, owner_span_start, owner_span_end) = if family == Family::Full && r == 1 {
                p.second.map_or((p.first, p.base, span_start, span_end), |s| (s.first, s.base, s.span_start, s.span_end))
            } else {
                (p.first, p.base, span_start, span_end)
            };
            let lo = match family {
                Family::Sliding => owner_span_start.saturating_sub(WINDOW),
                Family::Full => 0,
            };
            let row = &rows[owner_first];
            tables.push(plan.words.len());
            let pos = |blk: usize, entry: usize| owner_base + (128 * (4 * r + blk + 1)) as isize - w as isize + entry as isize;
            let mut slot = |q: isize| -> usize {
                if q >= lo as isize && q < owner_span_end as isize { pool_row(family, row, q as usize) } else { spread.next() }
            };
            let tiles = w / 128;
            match family {
                Family::Sliding => {
                    for n in 0..8 {
                        for t in 0..tiles {
                            for w8 in 0..8 {
                                for b2 in 0..2 {
                                    for sl in 0..16 {
                                        for bh in 0..2 {
                                            let pool = slot(pos(2 * bh + b2, 128 * t + 8 * sl + w8));
                                            plan.words.push(((pool * 8 + n) * Ds::SIZE) as i32);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Family::Full => {
                    for n in 0..8 {
                        for t in 0..tiles {
                            for w2 in 0..2 {
                                for w32 in 0..32 {
                                    for wh in 0..2 {
                                        let pool = slot(pos(n % 4, 128 * t + 64 * wh + 2 * w32 + w2));
                                        plan.words.push((pool * Df::SIZE) as i32);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        prefill_launches.push(Launch { kernel: p.kernel, row: region, tables });
    }
    if let Some((blocks, tiles)) = prompt_choice {
        let region = match PROMPT_PLACEMENT.with(|f| f.get()) {
            Some((images, row)) if images.contains(&tiles) && row >= placed.decode_rows => row,
            _ => placed.decode_rows,
        };
        let image_runs = runs_for_image.as_deref().unwrap_or(&runs);
        let launch = place_prompt(&mut plan, rows, image_runs, &blocks, tiles, region, row_bytes);
        let row_space = (region + 2 * launch.kernel.rows()).max(placed.row_space);
        if row_space > SINK_ROW {
            return Err(format!("the prompt image needs {} row-space rows; {SINK_ROW} exist", row_space));
        }
        prefill_launches.push(launch);
    }
    // A later region's leading dummy rows may overlap earlier regions: run it first.
    plan.launches = prefill_launches.into_iter().rev().chain(decode_launches).collect();
    if family == Family::Sliding && [1024, 1536].contains(&rows.len()) {
        append_pair_tables(&mut plan, rows.len());
    }
    if family == Family::Full && plan.full_groups.is_some() {
        plan.add_full_standalone()?;
    }
    Ok(plan)
}

/// A pair reads two original groups without changing an offset, cap bit or output row.
/// Pair the last groups the resident family can serve. Earlier groups run separately.
fn append_pair_tables(plan: &mut Plan, bucket: usize) {
    use super::resident::{PAIRED_GROUPS, paired_prompt};
    let Some(Kernel::Prompt { tiles }) = plan.launches.first().map(|l| l.kernel) else { return };
    if !paired_prompt(bucket, tiles) {
        return;
    }
    let decode: Vec<_> = plan.launches.iter().filter(|l| matches!(l.kernel, Kernel::Decode { .. })).collect();
    if decode.is_empty() || decode.iter().any(|l| l.kernel != (Kernel::Decode { family: Family::Sliding, w: WINDOW })) {
        return;
    }
    let span = DECODE_ROWS * WINDOW;
    let row_bytes = Hqs::SIZE * 2;
    let first = decode.len().saturating_sub(PAIRED_GROUPS);
    let mut groups: Vec<_> = decode[first..].iter().map(|l| ([l.tables[0], l.tables[1]], l.row)).collect();
    if groups.len() < PAIRED_GROUPS {
        // Spread masked gathers over the pool and scatter only to the sink row.
        let off = plan.words.len();
        plan.words.extend((1..=span).map(|i| ((i * 7919) % Rs::SIZE * Ns::SIZE * Ds::SIZE) as i32));
        let cap = plan.words.len();
        plan.words.extend(std::iter::repeat_n((-1e30f32).to_bits() as i32, span));
        groups.resize(PAIRED_GROUPS, ([off, cap], SINK_ROW));
    }
    for pair in groups.chunks_exact(2) {
        let mut offsets = [0; 3];
        for i in 0..2 {
            offsets[i] = plan.words.len();
            for (tables, _) in pair {
                plan.words.extend_from_within(tables[i]..tables[i] + span);
            }
        }
        offsets[2] = plan.words.len();
        for &(_, row) in pair {
            plan.words.extend((0..2 * DECODE_ROWS).map(|i| {
                let q = if row == SINK_ROW { SINK_ROW } else { row + i };
                (q * row_bytes) as i32
            }));
        }
        plan.paired_decode.push(offsets);
    }
}

// ---- constants and device state ------------------------------------------------------------------------------

/// A family's constant tables: each prefill kernel's tile masks per round and the edge caps, as i32 words.
pub struct Consts {
    pub words: Vec<i32>,
    pub tmask: HashMap<Kernel, Vec<usize>>,
    pub cap_lo: usize,
    pub cap_hi: usize,
    /// Sliding: the prompt images' constant rows (`prompt_blocks::tm_rows`, then `cap_rows`).
    pub prompt_tm: usize,
    pub prompt_cap: usize,
}

/// Words of the prompt images' `TM_ROWS` `[9][32][8]` and `CAP_ROWS` `[35][2][128][8]`.
const PROMPT_TM_WORDS: usize = 9 * 32 * 8;
const PROMPT_CAP_WORDS: usize = dk::prompt_blocks::CAP_WORDS;
/// Words of a prompt round's mask keys `[8][4][16]`.
const PROMPT_KEY_WORDS: usize = 8 * 4 * 16;

impl Consts {
    pub fn new(family: Family) -> Self {
        let bits = |x: f32| x.to_bits() as i32;
        let mut c = Consts { words: Vec::new(), tmask: HashMap::new(), cap_lo: 0, cap_hi: 0, prompt_tm: 0, prompt_cap: 0 };
        match family {
            Family::Sliding => {
                // `[n][block][slice][w][lane]`: lane `m` of slice `sl` is block query `8 sl + m`.
                for hi in [false, true] {
                    if hi {
                        c.cap_hi = c.words.len();
                    }
                    for _ in 0..32 {
                        for sl in 0..16 {
                            for w in 0..128 {
                                for m in 0..8 {
                                    let qi = 8 * sl + m;
                                    let attend = if hi { w <= qi } else { w > qi };
                                    c.words.push(bits(if attend { CAP_ATTEND } else { CAP_MASKED }));
                                }
                            }
                        }
                    }
                }
            }
            Family::Full => {}
        }
        if family == Family::Sliding {
            // 4 KiB-aligned regions.
            c.words.resize(c.words.len().next_multiple_of(1024), 0);
            c.prompt_tm = c.words.len();
            c.words.extend(dk::prompt_blocks::tm_rows().into_iter().map(bits));
            c.words.resize(c.words.len().next_multiple_of(1024), 0);
            c.prompt_cap = c.words.len();
            c.words.extend(dk::prompt_blocks::cap_rows().into_iter().map(bits));
            c.words.resize(c.words.len().next_multiple_of(1024), 0);
            debug_assert_eq!((c.prompt_cap - c.prompt_tm).min(PROMPT_TM_WORDS), PROMPT_TM_WORDS);
        }
        match family {
            Family::Sliding => {}
            Family::Full => {
                // `[half][quarter][slice][w][group]`: query `2 sl + g` of the block.
                for _ in 0..8 {
                    for sl in 0..64 {
                        for w in 0..128 {
                            for g in 0..2 {
                                c.words.push(bits(if w <= 2 * sl + g { CAP_ATTEND } else { CAP_MASKED }));
                            }
                        }
                    }
                }
            }
        }
        for kernel in Kernel::all().into_iter().filter(|k| k.family() == family && !matches!(k, Kernel::Decode { .. } | Kernel::Prompt { .. })) {
            let base = kernel.base();
            let mut at = Vec::new();
            for (r, &w) in kernel.windows().iter().enumerate() {
                at.push(c.words.len());
                let tiles = w / 128;
                let masked = |blk: usize, t: usize| {
                    t < tiles && base.is_some_and(|b| b + (128 * (4 * r + blk + 1 + t)) as isize - (w as isize) < 0)
                };
                let (outer, slices) = match family {
                    Family::Sliding => (8, 16),
                    Family::Full => (2, 64),
                };
                for _ in 0..outer {
                    for blk in 0..4 {
                        for _ in 0..slices {
                            for t in 0..32 {
                                let m = bits(if masked(blk, t) { CAP_MASKED } else { CAP_ATTEND });
                                c.words.extend(std::iter::repeat_n(m, 8));
                            }
                        }
                    }
                }
            }
            c.tmask.insert(kernel, at);
        }
        c
    }

    /// Tile masks then caps of a prefill kernel, in its argument order, as `(word, words)`.
    fn regions(&self, kernel: Kernel) -> Vec<(usize, usize)> {
        if let Kernel::Prompt { .. } = kernel {
            return vec![(self.prompt_tm, PROMPT_TM_WORDS), (self.prompt_cap, PROMPT_CAP_WORDS), (self.cap_hi, SLIDING_CAP_WORDS)];
        }
        let mut out: Vec<(usize, usize)> = self.tmask[&kernel].iter().map(|&at| (at, TM_WORDS)).collect();
        match kernel.family() {
            Family::Sliding => out.extend([(self.cap_lo, SLIDING_CAP_WORDS), (self.cap_hi, SLIDING_CAP_WORDS)]),
            Family::Full => out.push((self.cap_hi, FULL_CAP_WORDS)),
        }
        out
    }
}

/// E4M3 attention kernels and their constant tables.
pub struct Dsl {
    consts: [Consts; 2],
    consts_hbm: [HbmTensor<i32, Chip, m![Iw22]>; 2],
    functions: HashMap<Kernel, std::sync::Arc<furiosa_opt_std::backend::npu::Function>>,
}

fn slot(family: Family) -> usize {
    match family {
        Family::Sliding => 0,
        Family::Full => 1,
    }
}

impl Dsl {
    /// Upload the constants and preload every E4M3 attention kernel.
    pub async fn new(device: &mut Device) -> Result<Self, BoxError> {
        // These launches contribute device images to the registry. The preload below
        // resolves those images by path without executing the typed launches.
        if std::hint::black_box(false) {
            embed_sliding().await;
            embed_full().await;
            dk::prompt_blocks::embed_images().await;
        }
        let paths = Kernel::all().into_iter().map(|kernel| path(kernel).map(|p| (kernel, p)))
            .collect::<Result<Vec<_>, _>>()?;
        let consts = [Consts::new(Family::Sliding), Consts::new(Family::Full)];
        let mut upload = async |c: &Consts| -> Result<HbmTensor<i32, Chip, m![Iw22]>, BoxError> {
            let mut words = c.words.clone();
            if words.len() > Iw22::SIZE {
                return Err(format!("{} constant words exceed their tensor", words.len()).into());
            }
            words.resize(Iw22::SIZE, 0);
            Ok(HostTensor::<i32, m![Iw22]>::from_vec(words).to_hbm(&mut device.pdma).await?)
        };
        let consts_hbm = [upload(&consts[0]).await?, upload(&consts[1]).await?];
        let mut functions = HashMap::new();
        for (kernel, path) in paths {
            let function = std::sync::Arc::new(furiosa_opt_std::backend::npu::function(&*device, path, &[]).await?);
            crate::host::engine::step_image::register(path, &function);
            functions.insert(kernel, function);
        }
        Ok(Self { consts, consts_hbm, functions })
    }
}

// ---- launches ------------------------------------------------------------------------------------------------

/// Runs the planned launches by binding the device row space, pool and table buffers.
#[allow(clippy::too_many_arguments)]
pub async fn run<HQ: AxisName, E: M>(
    dsl: &Dsl,
    family: Family,
    launches: &[Launch],
    fused: usize,
    skip_first: usize,
    live: Option<&[bool]>,
    space: &mut RowSpace<HQ>,
    words: WordsAt<'_>,
    k_pool: &HbmTensor<f8e4m3, Chip, E>,
    v_pool: &HbmTensor<f8e4m3, Chip, E>,
) -> Result<(), BoxError> {
    if live.is_some_and(|keep| keep.len() != launches.len()) {
        return Err("live-launch mask length differs from DSL plan".into());
    }
    let half = half_width(family);
    // Resident segments cover the decode suffix and may also cover the first prompt launch.

    let launches = separate_launches(launches, fused, skip_first).ok_or("more covered or skipped launches than the plan has")?;
    if family == Family::Full && crate::host::engine::step_image::full_sequence::enabled() {
        let kept: Vec<_> = launches.iter().enumerate()
            .filter(|(i, _)| live.is_none_or(|keep| keep[skip_first + i]))
            .map(|(_, launch)| launch).collect();
        if (1..=10).contains(&kept.len()) {
            let frames: Vec<_> = kept.iter().map(|launch|
                bind_device_launch(dsl, family, launch, half, space, words, k_pool, v_pool)
            ).collect::<Result<_, _>>()?;
            if let Some((function, inputs, outputs)) = crate::host::engine::step_image::full_sequence::compose(&frames) {
                crate::host::engine::queue::run_full_sequence(function, inputs, outputs).await?;
                return Ok(());
            }
        }
    }
    for (i, launch) in launches.iter().enumerate() {
        if live.is_some_and(|keep| !keep[skip_first + i]) { continue; }
        let (function, inputs, outputs) = bind_device_launch(dsl, family, launch, half, space, words, k_pool, v_pool)?;
        crate::host::engine::queue::run(function, inputs, outputs).await?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn bind_device_launch<HQ: AxisName, E: M>(
    dsl: &Dsl, family: Family, launch: &Launch, half: usize,
    space: &RowSpace<HQ>, words: WordsAt<'_>, k_pool: &HbmTensor<f8e4m3, Chip, E>, v_pool: &HbmTensor<f8e4m3, Chip, E>,
) -> Result<crate::host::engine::step_image::full_sequence::Frame, BoxError> {
    use furiosa_opt_std::runtime::{Buffers, DeviceSend};
    let kernel = launch.kernel;
    let function = dsl.functions.get(&kernel).ok_or_else(|| format!("DSL kernel {kernel:?} is not loaded"))?;
    let tables = &launch.tables;
    let region = |out: &mut Buffers, name: &str, tensor: &dyn DeviceSend, byte: usize, len: Option<usize>| -> Result<(), BoxError> {
        let mut bound = Buffers::new();
        tensor.bind(&mut bound)?;
        let buffer = match bound.len() {
            1 => bound.pop().unwrap(),
            n => return Err(format!("{name}: a tensor binds {n} buffers").into()),
        };
        let end = len.map_or(buffer.size(), |len| byte + len);
        if end > buffer.size() {
            return Err(format!("{name} of {kernel:?}: bytes {byte}..{end} exceed the tensor's {} B", buffer.size()).into());
        }
        out.try_push(buffer.slice(byte..end)).map_err(|_| format!("{kernel:?}: too many launch arguments"))?;
        Ok(())
    };
    let q_byte = launch.row * half * 2;
    let q_len = 2 * kernel.rows() * half * 2;
    let (table, base) = words.tensor()?;
    let consts = &dsl.consts[slot(family)];
    let consts_hbm: &dyn DeviceSend = &dsl.consts_hbm[slot(family)];
    let mut inputs = Buffers::new();
    if let Kernel::Prompt { .. } = kernel {
        // Per round its 1,024 rows of queries and of outputs, offsets and mask keys; the constant rows; the pools.
        let round = 2 * 512 * half * 2;
        for (r, (&w, t)) in kernel.windows().iter().zip(tables.chunks(3)).enumerate() {
            region(&mut inputs, "query", &space.qin, q_byte + r * round, Some(round))?;
            region(&mut inputs, "output", &space.out, q_byte + r * round, Some(round))?;
            region(&mut inputs, "offsets", table, base + t[0] * 4, Some(kernel.offset_words(w) * 4))?;
            region(&mut inputs, "tile keys", table, base + t[1] * 4, Some(PROMPT_KEY_WORDS * 4))?;
            region(&mut inputs, "cap keys", table, base + t[2] * 4, Some(PROMPT_KEY_WORDS * 4))?;
        }
        for (at, n) in consts.regions(kernel) {
            region(&mut inputs, "constants", consts_hbm, at * 4, Some(n * 4))?;
        }
        region(&mut inputs, "k_cache", k_pool, 0, None)?;
        region(&mut inputs, "v_cache", v_pool, 0, None)?;
        return Ok((function.clone(), inputs, Buffers::new()));
    }
    region(&mut inputs, "query", &space.qin, q_byte, Some(q_len))?;
    match kernel {
        Kernel::Decode { w, .. } => {
            for &at in tables {
                region(&mut inputs, "table", table, base + at * 4, Some(DECODE_ROWS * w * 4))?;
            }
        }
        _ => {
            for (&w, &at) in kernel.windows().iter().zip(tables) {
                region(&mut inputs, "offsets", table, base + at * 4, Some(kernel.offset_words(w) * 4))?;
            }
            for (at, n) in consts.regions(kernel) {
                region(&mut inputs, "constants", consts_hbm, at * 4, Some(n * 4))?;
            }
        }
    }
    region(&mut inputs, "k_cache", k_pool, 0, None)?;
    region(&mut inputs, "v_cache", v_pool, 0, None)?;
    let mut outputs = Buffers::new();
    region(&mut outputs, "output", &space.out, q_byte, Some(q_len))?;
    Ok((function.clone(), inputs, outputs))
}

/// The Sliding resident programs (`resident`): binds a prompt launch's per-step tables without its query and output rows
/// and pools: per round the first KV head's eighth of its offsets (`ops_core::resident::Off0`), its tile and cap
/// keys; then the constant mask rows.
pub(crate) fn bind_prompt_tables(
    dsl: &Dsl,
    launch: &Launch,
    words: WordsAt<'_>,
    inputs: &mut furiosa_opt_std::runtime::Buffers,
) -> Result<(), BoxError> {
    use furiosa_opt_std::runtime::{Buffers, DeviceSend};
    let kernel = launch.kernel;
    if !matches!(kernel, Kernel::Prompt { .. }) {
        return Err(format!("{kernel:?} is not a prompt image").into());
    }
    let region = |out: &mut Buffers, tensor: &dyn DeviceSend, byte: usize, len: usize| -> Result<(), BoxError> {
        let mut bound = Buffers::new();
        tensor.bind(&mut bound)?;
        let buffer = match bound.len() {
            1 => bound.pop().unwrap(),
            n => return Err(format!("{kernel:?}: a tensor binds {n} buffers").into()),
        };
        if byte + len > buffer.size() {
            return Err(format!("{kernel:?}: bytes {byte}..{} exceed the tensor's {} B", byte + len, buffer.size()).into());
        }
        out.try_push(buffer.slice(byte..byte + len)).map_err(|_| format!("{kernel:?}: too many launch arguments"))?;
        Ok(())
    };
    let (table, base) = words.tensor()?;
    let consts_hbm: &dyn DeviceSend = &dsl.consts_hbm[slot(Family::Sliding)];
    let tables = &launch.tables;
    for (&w, t) in kernel.windows().iter().zip(tables.chunks(3)) {
        region(inputs, table, base + t[0] * 4, kernel.offset_words(w) / 8 * 4)?;
        region(inputs, table, base + t[1] * 4, PROMPT_KEY_WORDS * 4)?;
        region(inputs, table, base + t[2] * 4, PROMPT_KEY_WORDS * 4)?;
    }
    for (at, n) in dsl.consts[slot(Family::Sliding)].regions(kernel) {
        region(inputs, consts_hbm, at * 4, n * 4)?;
    }
    Ok(())
}

/// Standalone launches after resident coverage and prefix skips.
fn separate_launches(launches: &[Launch], fused: usize, skip_first: usize) -> Option<&[Launch]> {
    let end = launches.len().checked_sub(fused)?;
    launches.get(skip_first..end)
}

/// Only calls actually issued outside confirmed resident coverage need a
/// standalone family opcode. Keep missing paths and invalid counts as failures.
pub(super) fn step_paths(
    plans: &[crate::host::engine::attention::StepPlan; 2],
    full_fused: usize,
) -> Option<Vec<&'static str>> {
    let mut out = Vec::new();
    for (slot, plan) in plans.iter().enumerate() {
        let fused = if slot == 1 { full_fused } else { 0 };
        for launch in separate_launches(&plan.launches, fused, 0)? {
            let path = path(launch.kernel).ok()?;
            if !out.contains(&path) { out.push(path); }
        }
    }
    Some(out)
}
