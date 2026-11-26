//! Admission and step packing (pure host logic).
//!
//! Admission gives a request a KV owner slot and, up front, every Full page its prompt and
//! generation budget can reach, so an admitted request always finishes without preemption;
//! a request that does not fit waits (first in, first out). Each step gives every decoding
//! request one row, then fills the remaining token budget with prompt chunks in admission
//! order, at most `max_chunk` rows per request.
//!
//! Chunks fill the room the step has left: any start runs on a
//! prefill bucket of the chunk geometry (`attention_geometry::PREFILL_PAIRS`) whose cached region holds at least the
//! chunk's history (padded at the front, `attention_geometry::prefill_covers`), so a chunk takes `min(remaining, room, span)`, where the
//! span is the longest chunk both families' buckets serve at that start. Past the largest Full
//! bucket's cached region (1,920) a chunk's Full rows run decode-style and only the room limits
//! it.
//!
//! Every packed step fits both families' attention row space: a chunk that would
//! push the step's DSL launches past `SINK_ROW` rows is cut at an earlier 128-row boundary or
//! left out, and its request waits for the next step. Decode rows are always taken.
use crate::axes::Tf;
use crate::host::engine::dsl::{self, StepSpan};
use crate::host::engine::attention_geometry::{EXCLUDED_FULL, Family, PREFILL_PAIRS, SINK_ROW, prefill_covers};
use furiosa_opt_std::prelude::AxisName;

pub mod row_role;
pub use row_role::{RowOrder, RowSpan};

/// The longest chunk both families' prefill buckets serve at `start` (padded prefixes, the
/// excluded Full buckets left out); `None` past the largest Full bucket's cached region.
pub fn span(start: usize) -> Option<usize> {
    let longest = |family: Family| {
        PREFILL_PAIRS
            .iter()
            .filter(|&&(a, k)| !(family == Family::Full && EXCLUDED_FULL.contains(&(a, k))) && prefill_covers(family, k, start))
            .map(|&(a, k)| a - k)
            .max()
    };
    Some(longest(Family::Sliding)?.min(longest(Family::Full)?))
}

/// Rows of a prompt chunk at `start` with `remaining` prompt rows and `room` rows left in the
/// step: the room, up to the prompt's end and the longest chunk a prefill bucket serves there
/// (the module documentation).
pub fn chunk_rows(start: usize, remaining: usize, room: usize) -> usize {
    let n = remaining.min(room);
    match span(start) {
        Some(span) => n.min(span),
        None => n,
    }
}

/// Full pages that hold `positions` positions.
pub fn pages_for(positions: usize) -> usize {
    positions.div_ceil(Tf::SIZE)
}

/// Free KV owner slots and Full pages.
#[derive(Clone)]
pub struct Allocator {
    owners: Vec<usize>,
    pages: Vec<usize>,
}

impl Allocator {
    pub fn new(owners: usize, pages: usize) -> Self {
        Self { owners: (0..owners).rev().collect(), pages: (0..pages).rev().collect() }
    }

    /// An owner slot and the pages for `positions` positions, or `None` if either is short.
    pub fn admit(&mut self, positions: usize) -> Option<(usize, Vec<usize>)> {
        let need = pages_for(positions);
        if self.owners.is_empty() || self.pages.len() < need {
            return None;
        }
        let owner = self.owners.pop().unwrap();
        let pages = self.pages.split_off(self.pages.len() - need).into_iter().rev().collect();
        Some((owner, pages))
    }

    pub fn release(&mut self, owner: usize, mut pages: Vec<usize>) {
        debug_assert!(!self.owners.contains(&owner));
        self.owners.push(owner);
        pages.reverse();
        self.pages.append(&mut pages);
    }
}

/// One active request's demand on the next step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Demand {
    /// Prompt positions not yet computed; 0 once the request decodes.
    pub remaining_prompt: usize,
    /// The next prompt position (its count of computed positions).
    pub position: usize,
}

/// Ready-row cutoff for repacking a wide step.
pub const NEXT_STEP_ROWS: usize = 512;

/// Rows each request gets in the next step (aligned with `demands`, admission order): one per
/// decoding request, then prompt chunks of at most `max_chunk` rows in order while the step's
/// `max_rows` last, each ended by `chunk_rows`. A wide step that would leave fewer than `NEXT_STEP_ROWS` rows
/// ready becomes a full 1,024-row step when one fits; the rows past 1,024 move to the next step.
pub fn pack(demands: &[Demand]) -> Vec<usize> {
    let rows = pack_rows(demands, super::tables::MAX_STEP_ROWS);
    if rows.iter().sum::<usize>() > 1024 && ready_after(demands, &rows) < NEXT_STEP_ROWS {
        let narrow = pack_rows(demands, 1024);
        if narrow.iter().sum::<usize>() == 1024 {
            return narrow;
        }
    }
    rows
}

/// Rows ready for the step after one that gives `rows[i]` to `demands[i]`: one per request decoding by then and
/// the next chunk of every prompt still open. Requests that arrive or end in between are not known here.
fn ready_after(demands: &[Demand], rows: &[usize]) -> usize {
    demands
        .iter()
        .zip(rows)
        .map(|(d, &n)| match d.remaining_prompt.saturating_sub(n) {
            0 => 1,
            left => chunk_rows(d.position + n, left, super::tables::MAX_STEP_ROWS),
        })
        .sum()
}

fn pack_rows(demands: &[Demand], max_rows: usize) -> Vec<usize> {
    let mut rows: Vec<usize> = demands.iter().map(|d| usize::from(d.remaining_prompt == 0)).collect();
    let mut budget = max_rows.saturating_sub(rows.iter().sum());
    for (i, d) in demands.iter().enumerate() {
        if budget == 0 {
            break;
        }
        if d.remaining_prompt > 0 {
            let want = chunk_rows(d.position, d.remaining_prompt, budget);
            budget -= fit_chunk(demands, &mut rows, i, want);
        }
    }
    debug_assert!(fits(demands, &rows), "a packed step overflows the row space: {demands:?} {rows:?}");
    rows
}

/// Whether a step giving `rows[i]` rows to `demands[i]` fits both families' attention row space:
/// the DSL launches of its spans, laid out in admission order as the engine lays out the step,
/// take at most `SINK_ROW` rows (`dsl::row_space_need`, the planner's own routing and placement).
/// Both families read E4M3 pools with no fallback; steps must fit their DSL regions. A decode span's position does not matter (its rows take decode
/// launches).
pub fn fits(demands: &[Demand], rows: &[usize]) -> bool {
    row_space_needs(demands, rows).iter().all(|&need| need <= SINK_ROW)
}

/// The row-space rows of a step giving `rows[i]` rows to `demands[i]`, Sliding then Full (`fits`).
pub fn row_space_needs(demands: &[Demand], rows: &[usize]) -> [usize; 2] {
    let mut spans: Vec<StepSpan> = Vec::new();
    let mut first = 0;
    for (d, &n) in demands.iter().zip(rows) {
        if n > 0 {
            spans.push(StepSpan { first, start: d.position, n, prompt: d.remaining_prompt > 0 });
            first += n;
        }
    }
    [Family::Sliding, Family::Full].map(|family| dsl::row_space_need(family, &spans))
}

/// Sets `rows[i]` to the longest chunk of at most `want` rows with which the step still `fits`
/// and returns it: `want`, else a chunk ending at an earlier 128-row boundary (where a causal
/// prefill piece's window ends), longest first, else none: the request waits for the next step.
/// `rows` holds the step so far (every decode row and the earlier chunks).
fn fit_chunk(demands: &[Demand], rows: &mut [usize], i: usize, want: usize) -> usize {
    let d = &demands[i];
    rows[i] = want;
    if want == 0 || fits(demands, rows) {
        return want;
    }
    let shorter: Vec<usize> = (d.position / 128 + 1..).map(|b| b * 128 - d.position).take_while(|&n| n < want).collect();
    for n in shorter.into_iter().rev() {
        rows[i] = n;
        if rows[i] > 0 && fits(demands, rows) {
            note_row_space_cut(want, rows[i]);
            return rows[i];
        }
    }
    rows[i] = 0;
    note_row_space_cut(want, 0);
    0
}

/// Notes a prompt chunk the row space cut (`fit_chunk`): the first notes and every 100th go to
/// the log.
fn note_row_space_cut(want: usize, got: usize) {
    static CUTS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = CUTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if n <= 3 || n.is_multiple_of(100) {
        eprintln!("ENGINE_ROW_SPACE: a {want}-row prompt chunk cut to {got} rows to fit the attention row space ({n} cuts so far)");
    }
}
