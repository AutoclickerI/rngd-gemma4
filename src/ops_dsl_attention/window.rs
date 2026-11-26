//! Sliding attention geometry. A query at position `q` attends keys at positions `k` with
//! `0 <= q - k < W`, including itself. The serving window is 384 positions for all modalities.
//!
//! Host masks, band rounds, prefill kernels and resident decode groups derive their geometry
//! from these constants. The 1,024-slot Sliding KV ring still holds every window position;
//! older rows are excluded by the host tables and per-row masks.

use furiosa_opt_std::prelude::AxisName;

pub const SLIDING_WINDOW: usize = 384;

/// Tiles of 128 entries a band block sees: its own 128 positions and the `W` before its last one.
pub const TILES: usize = SLIDING_WINDOW / 128 + 1;

/// The decode window every full-window in-program group runs at.
pub const GROUP_WINDOW: usize = SLIDING_WINDOW;

const _: () = assert!(SLIDING_WINDOW % 128 == 0, "a band block's window is whole 128-entry tiles");
const _: () = assert!(SLIDING_WINDOW <= crate::axes::Ts::SIZE, "an owner's ring holds its window");
