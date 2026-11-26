//! Attention geometry shared by the DSL planner, the scheduler and the resident programs: the two attention
//! families, the row space's sink row, the table-word padding, a launch's host cost and the prefill chunk
//! geometry the scheduler packs prompt chunks by (`schedule::span`).
use super::dsl::WINDOW;
use crate::axes::{Hqf, Hqs, Rq};
use furiosa_opt_std::prelude::AxisName;

/// Query heads of one row-space row (one head half of a step row's 16 query heads).
pub const QUERY_HEADS: usize = 8;

/// The row space row that takes the moves of rows no launch reads.
pub const SINK_ROW: usize = Rq::SIZE - 1;
/// Zero words after a plan's table words, so the last launch's mask region never ends at the
/// end of the upload (`attention::Words`).
pub const MASK_PAD_WORDS: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Family {
    Sliding,
    Full,
}

impl Family {
    pub const fn head_dim(self) -> usize {
        match self {
            Family::Sliding => 256,
            Family::Full => 512,
        }
    }

    /// Bytes of one query or output row (8 heads).
    pub const fn row_bytes(self) -> usize {
        QUERY_HEADS * self.head_dim() * 2
    }
}

/// Elements of one row-space query or output row (8 heads).
pub fn half_width(family: Family) -> usize {
    match family {
        Family::Sliding => Hqs::SIZE,
        Family::Full => Hqf::SIZE,
    }
}

/// The 14 prefill `(A, K)` chunk buckets of each family up to A = 2048: a chunk of `A - K` rows over a
/// cached region of `K` positions (`schedule::span`).
pub const PREFILL_PAIRS: [(usize, usize); 14] = [
    (128, 0),
    (256, 0),
    (256, 128),
    (512, 0),
    (512, 256),
    (512, 384),
    (1024, 0),
    (1024, 512),
    (1024, 768),
    (1024, 896),
    (2048, 1024),
    (2048, 1536),
    (2048, 1792),
    (2048, 1920),
];

/// Full prefill chunks with these shapes use a larger bucket to meet the attention error bound.
pub const EXCLUDED_FULL: [(usize, usize); 4] = [(128, 0), (256, 0), (256, 128), (512, 256)];

/// Whether a prefill bucket with cached capacity `K` serves a chunk at `start`: Full sees the whole
/// history (`start` positions; the cached region holds `K`), Sliding the window before the chunk
/// (`min(start, window - 1)` positions; the cached region holds `min(K, window)`).
pub fn prefill_covers(family: Family, k: usize, start: usize) -> bool {
    match family {
        Family::Sliding => k.min(WINDOW) >= start.min(WINDOW - 1),
        Family::Full => k >= start,
    }
}
