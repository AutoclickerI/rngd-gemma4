//! Fixed-capacity Sliding prompt attention: one launch per layer serves every prompt chunk of a
//! step, as rounds of four 128-row blocks. A block is one owner's 128 consecutive positions `p0.. p0 + 128`; it sees
//! the `128 T` list entries ending at its last position (`T` tiles of 128, gathered per block as
//! `prefill::band512_fp8` gathers), and its row `j` (position `p0 + j`) attends to the entries from `max(0, p0 + j -
//! W + 1)` to `p0 + j` (`W` the Sliding window, `window::SLIDING_WINDOW`; the band rounds have `TILES = W / 128 + 1`
//! tiles). A round's `T` is a compile-time constant (one image per tuple of round
//! windows; the host picks the cheapest image whose rounds hold the step's blocks, sorted by need); within a round,
//! what differs per block is only which tiles hold positions before 0:
//! - `T = TILES`, `p0 >= W`: no padding. The first tile's cap starts the window.
//! - `p0 < W`, a multiple of 128: the first `v = T - (p0 + 128) / 128` tiles. Rows at positions `< W` therefore
//!  sit in blocks aligned to 128 positions and later rows may start anywhere.
//!
//! A round of `TILES` tiles must be a `band` round (its first tile clipped by `cap_lo`); a `causal` round has fewer
//! (its blocks' rows are all below `W`, so they see their whole history).
//!
//! So the masks are small gathers from constant rows, keyed per block by `v` (no per-entry mask uploads, no runtime
//! loop bounds):
//! - `tm` (`[W / 128 # 32, Q % 8]` per slice, the causal kernels' tile mask): row `v` of `TM_ROWS` masks tiles
//!   `< v`. It clips every tile's scores (`prefill`'s `mid = all`); the last tile (the diagonal) is then recomputed
//!   with `cap_hi`.
//! - `T = TILES` only, `cap_lo` (`[Wt, Q % 8]` per slice): the band kernel's first-tile cap of the slice's queries
//!   (`v = 0`), or all masked (`v > 0`); the first tile is recomputed with it.
//!
//! Masked entries add exact zeros, so a row's bits depend only on its position and keys.
//! Dummy rows read and write only their own block's row-space rows.

use furiosa_opt_std::prelude::*;

axes![
    N = 8,       // KV heads: one per PE (2 clusters x 4 PEs)
    G2 = 2,      // query heads per KV head
    D = 256,     // head dim
    Q = 512,     // a round's queries: Q / 128 block (sub-ring), Q / 8 % 16 slice, Q % 8 lane
    Bb = 4,      // a round's blocks
    Wt = 128,    // a window tile (edge caps): position 8 x source slice + position in it
    Sh = 835584, // cache rows of one KV head: pool row x 8 + KV head (`axes::Rs` x 8)
    Cr = 35,     // paired lower caps: 16 first-edge, 3 uniform, 16 second-edge
    Edge = 2,   // lower-edge masks for tiles 0 and 1
    Vt = 9,      // constant tile-mask rows: `v` leading tiles masked, 0..= 8
    Tp = 32,     // a tile-mask row's tiles (a window's tiles padded to 32, as `prefill`'s tile masks)
    Ln = 8,      // a constant row's lanes (`Q % 8` once gathered; its own label so the gather keys on one axis)
    W1 = 128,    // round windows: T tiles of 128
    W2 = 256,
    W3 = 384,
    W4 = 512,
];

const _: () = assert!(Sh::SIZE == crate::axes::Rs::SIZE * 8, "Sh is the Sliding pool's rows x KV heads");

/// Tiles of the band window (`window::TILES`: `W / 128 + 1`).
pub const TILES: usize = super::window::TILES;
const _: () = assert!(TILES <= Vt::SIZE, "a tile-mask row per leading masked tile count");
/// A masked score clip (`dsl::CAP_MASKED`).
pub const MASKED: f32 = -1e30;
/// An attending score clip (`dsl::CAP_ATTEND`).
pub const ATTEND: f32 = f32::MAX;

pub type Chip = m![1];
/// One KV head per PE.
pub type C = m![N / 4];
/// Query slices: the PE, its sub-ring (block), 8 queries per slice.
pub type S = m![N % 4, Q / 128 % 4, Q / 8 % 16];
/// The lanes: a slice's 8 queries.
pub type L = m![Q % 8];
/// Per-slice clip of one 128-entry edge tile, per lane.
pub type Cap = DmTensor<f32, Chip, C, S, m![Wt, Q % 8]>;
/// Key slices of a window: the sub-ring's slices, each holding 8 positions of every 128-entry tile.
pub type SK<Wn> = m![N % 4, Q / 128 % 4, Wn / 8 % 16];
/// A round's queries, `[Q][N][G2][D]` (query head `2n + g` reads KV head `n`): 1,024 row-space rows.
pub type Qh = HbmTensor<bf16, Chip, m![Q, N, G2, D]>;
/// Per KV head and block, the byte offset (`slot · row + n · head`) of each of the block's list entries in the
/// gather's walk order `[N][Wn / 128][Wn % 8][Bb % 2][Wn / 8 % 16][Bb / 2]` (as `prefill`'s `off_r`).
pub type Off<Wn> = HbmTensor<i32, Chip, m![N, Wn / 128, Wn % 8, Bb % 2, Wn / 8 % 16, Bb / 2]>;
/// Per KV head, block and slice, a byte offset into the constant rows (`TM_ROWS`: `v · 1 KiB`; `CAP_ROWS`:
/// `(v == 0 ? slice: 16) · 4 KiB`).
pub type Key = HbmTensor<i32, Chip, m![N, Bb, Q / 8 % 16]>;
/// Constant: row `v` masks window tiles `< v` (`[Vt][32][8]`, the lane axis repeats the value).
pub type TmRows = HbmTensor<f32, Chip, m![Vt, Tp, Ln]>;
/// Constant pairs of lower caps. The text edge is tile 1 when promoted into a four-tile round.
pub type CapRows = HbmTensor<f32, Chip, m![Cr, Edge, Wt, Ln]>;
/// Both possible lower edges of a mixed W256/W384 round.
pub type LowerCaps = DmTensor<f32, Chip, C, S, m![Edge, Wt, Q % 8]>;
pub const CAP_WORDS: usize = Cr::SIZE * Edge::SIZE * Wt::SIZE * Ln::SIZE;
/// Constant: the band kernel's `cap_hi` (`[N][Bb][16][Wt][Q % 8]`, `w <= 8 sl + m` attends): `dsl::Consts::cap_hi`.
pub type CapHi = HbmTensor<f32, Chip, m![N, Q / 128 % 4, Q / 8 % 16, Wt, Q % 8]>;
/// The pools, as `[Rs x 8][256]` rows of one KV head.
pub type Cache = HbmTensor<f8e4m3, Chip, m![Sh, D]>;
/// A round's output rows in DM, `[G2][Q % 8][D]` per slice.
pub type Out = DmTensor<bf16, Chip, C, S, m![G2, Q % 8, D]>;

/// A round's rows: its 1,024 row-space rows of queries and of outputs (the same rows of `qin` and `out`).
#[derive(DeviceSend)]
pub struct RoundIo<'a> {
    pub q: &'a Qh,
    pub out: &'a mut Qh,
}

/// A round's mask keys (per step; `cap` is read only by a band round, `TILES` tiles).
#[derive(DeviceSend, Clone, Copy)]
pub struct Keys<'a> {
    pub tm: &'a Key,
    pub cap: &'a Key,
}

/// The constant mask rows (uploaded once, like `dsl::Consts`).
#[derive(DeviceSend, Clone, Copy)]
pub struct Masks<'a> {
    pub tm_rows: &'a TmRows,
    pub cap_rows: &'a CapRows,
    pub cap_hi: &'a CapHi,
}

macro_rules! prompt_round {
    ($load:ident, $wn:ident) => {
        #[allow(clippy::type_complexity)]
        pub(crate) fn $load(
            device: &mut Device,
            q: &Qh,
            off: &Off<$wn>,
            keys: &Keys<'_>,
            masks: &Masks<'_>,
            k_cache: &Cache,
            v_cache: &Cache,
        ) -> (
            DmTensor<bf16, Chip, C, S, m![Q % 8, G2, D]>,
            DmTensor<f8e4m3, Chip, C, SK<$wn>, m![$wn / 128, $wn % 8, D]>,
            DmTensor<f8e4m3, Chip, C, SK<$wn>, m![$wn / 128, $wn % 8, D]>,
            DmTensor<f32, Chip, C, S, m![$wn / 128 # 32, Q % 8]>,
            LowerCaps,
        ) {
            let q_dm: DmTensor<bf16, Chip, C, S, m![Q % 8, G2, D]> = q.to_dm(&mut device.tdma);
            let kg: DmTensor<f8e4m3, Chip, C, m![N % 4, Bb, $wn / 8 % 16], m![$wn / 128, $wn % 8, D]> = k_cache.gather::<m![Sh], m![D]>().by_byte_offsets(off).to_dm(&mut device.tdma);
            let vg: DmTensor<f8e4m3, Chip, C, m![N % 4, Bb, $wn / 8 % 16], m![$wn / 128, $wn % 8, D]> = v_cache.gather::<m![Sh], m![D]>().by_byte_offsets(off).to_dm(&mut device.tdma);
            let tm: DmTensor<f32, Chip, C, m![N % 4, Bb, Q / 8 % 16], m![Tp, Ln]> = masks.tm_rows.gather::<m![Vt], m![Tp, Ln]>().by_byte_offsets(keys.tm).to_dm(&mut device.tdma);
            let cap_lo: DmTensor<f32, Chip, C, m![N % 4, Bb, Q / 8 % 16], m![Edge, Wt, Ln]> = masks.cap_rows.gather::<m![Cr], m![Edge, Wt, Ln]>().by_byte_offsets(keys.cap).to_dm(&mut device.tdma);
            // The windows as the key slices of the query slices' sub-rings; the masks per query slice and lane.
            unsafe { (q_dm, kg.reshape(), vg.reshape(), tm.reshape(), cap_lo.reshape()) }
        }

    };

}

#[cfg(feature = "compile-carried")]
macro_rules! prompt_round_core {
    (@loaded single, $wn:ident) => { (
        DmTensor<bf16, Chip, C, S, m![Q % 8, G2, D]>,
        DmTensor<f8e4m3, Chip, C, SK<$wn>, m![$wn / 128, $wn % 8, D]>,
        DmTensor<f8e4m3, Chip, C, SK<$wn>, m![$wn / 128, $wn % 8, D]>,
    ) };

}

#[cfg(feature = "compile-carried")]
pub(crate) type LoadedSingle = prompt_round_core!(@loaded single, W1);

prompt_round!(load_w1, W1);
prompt_round!(load_w2, W2);
// The band rounds of the smaller windows (`window`): 7 tiles at 768, 5 at 512, 4 at 384.
prompt_round!(load_w3b, W3);
prompt_round!(load_w4r, W4);

// The routed four-tile round serves both the band and middle roles.
pub use self::W4 as Band;
#[cfg(feature = "compile-carried")]
pub(crate) use self::round_w4r as round_band;
pub use self::W4 as Mid;
#[cfg(feature = "compile-carried")]
pub(crate) use self::round_w4r as round_mid;
const _: () = assert!(Band::SIZE == 128 * TILES, "the band axis holds TILES tiles");
/// Tiles of the middle round; at W384 this is a routed band for both modalities.
pub const MID_TILES: usize = Mid::SIZE / 128;
const _: () = assert!(MID_TILES <= TILES, "the middle round fits the band window");

/// Stores a round's output rows to `out` `[Q][N][G2][D]`.
pub(crate) fn store(device: &mut Device, o: &Out, out: &mut Qh) {
    o.view().to_hbm_view(&mut device.tdma, out.view_mut());
}


/// The images: one launch fn per tuple of round windows (rounds in argument order, each `(rows, offsets, keys)`, then
/// the constant masks and pools). Typed launches embed the catalog for raw dispatch.
macro_rules! prompt_images {
    ($($name:ident: [$(($load:ident, $round:ident, $wn:ident, $t:literal, $io:ident, $off:ident, $keys:ident)),+]),+ $(,)?) => {
        $(
        /// Sliding prompt attention of one layer for a step's blocks (module doc), rounds of these tiles:
        #[doc = stringify!($($t),+)]
        #[device(chip = 1)]
        #[allow(clippy::too_many_arguments)]
        pub fn $name(device: &mut Device, $($io: RoundIo<'_>, $off: &Off<$wn>, $keys: Keys<'_>,)+ masks: Masks<'_>, k_cache: &Cache, v_cache: &Cache) {
            // Every round's loads first (each round's keys become its loaded inputs), then per round compute and store.
            $(
                let $keys = $load(device, $io.q, $off, &$keys, &masks, k_cache, v_cache);
            )+
            let cap_hi: Cap = masks.cap_hi.to_dm(&mut device.tdma);
            $(
                let o = $round(device, $keys, &cap_hi);
                store(device, &o, $io.out);
            )+
        }
        )+

        /// Every image: its rounds' tiles (descending) and its registry path.
        pub fn images() -> Vec<(&'static [usize], &'static str)> {
            vec![$((&[$($t),+], $name.path())),+]
        }

        pub(crate) async fn embed_images() {
            $(let _ = launch($name, crate::host::engine::dsl::embedding_args()).await;)+
        }
    };
}

// Prompt images combine routed four-tile bands with shorter causal rounds.
prompt_images!(
    swa_prompt_fp8_4_4_4: [(load_w4r, round_w4r, W4, 4, r0, off0, keys0), (load_w4r, round_w4r, W4, 4, r1, off1, keys1), (load_w4r, round_w4r, W4, 4, r2, off2, keys2)],
    swa_prompt_fp8_4_4_1: [(load_w4r, round_w4r, W4, 4, r0, off0, keys0), (load_w4r, round_w4r, W4, 4, r1, off1, keys1), (load_w1, round_w1, W1, 1, r2, off2, keys2)],
    swa_prompt_fp8_4_4_2: [(load_w4r, round_w4r, W4, 4, r0, off0, keys0), (load_w4r, round_w4r, W4, 4, r1, off1, keys1), (load_w2, round_w2, W2, 2, r2, off2, keys2)],
    swa_prompt_fp8_4_3_1: [(load_w4r, round_w4r, W4, 4, r0, off0, keys0), (load_w3b, round_w3b, W3, 3, r1, off1, keys1), (load_w1, round_w1, W1, 1, r2, off2, keys2)],
    swa_prompt_fp8_4_4: [(load_w4r, round_w4r, W4, 4, r0, off0, keys0), (load_w4r, round_w4r, W4, 4, r1, off1, keys1)],
    swa_prompt_fp8_4_3: [(load_w4r, round_w4r, W4, 4, r0, off0, keys0), (load_w3b, round_w3b, W3, 3, r1, off1, keys1)],
    swa_prompt_fp8_4_2: [(load_w4r, round_w4r, W4, 4, r0, off0, keys0), (load_w2, round_w2, W2, 2, r1, off1, keys1)],
    swa_prompt_fp8_4_1: [(load_w4r, round_w4r, W4, 4, r0, off0, keys0), (load_w1, round_w1, W1, 1, r1, off1, keys1)],
    swa_prompt_fp8_4: [(load_w4r, round_w4r, W4, 4, r0, off0, keys0)],
);

// ---- host side: the constant rows and one round's tables  ---------------------------------------

/// `TM_ROWS` as f32: row `v` masks tiles `< v` (`[Vt][32][8]`).
pub fn tm_rows() -> Vec<f32> {
    (0..Vt::SIZE).flat_map(|v| (0..32).flat_map(move |t| std::iter::repeat_n(if t < v && t < TILES { MASKED } else { ATTEND }, 8))).collect()
}

/// `[Cr][Edge][Wt][8]`: keys 0..15 clip tile 0, 16 masks only tile 0, 17 masks both,
/// 18 attends both, and 19..34 mask tile 0 and clip tile 1. The host retains leading negative-position masks.
pub fn cap_rows() -> Vec<f32> {
    let mut out = Vec::with_capacity(CAP_WORDS);
    for row in 0..Cr::SIZE {
        for edge in 0..2 {
            for w in 0..128 {
                for m in 0..8 {
                    let attend = match (row, edge) {
                        (0..=15, 0) => w > 8 * row + m,
                        (0..=16, 1) | (18, _) => true,
                        (19..=34, 1) => w > 8 * (row - 19) + m,
                        _ => false,
                    };
                    out.push(if attend { ATTEND } else { MASKED });
                }
            }
        }
    }
    out
}

/// One block of a round: its first position `p0` (a multiple of 128 when below `W`) and the pool row of each
/// position the block can see (`entry(p)` for `p0 + 128 - 128 TILES <= p < p0 + 128`, positions before 0 excluded;
/// the owner's rows for its positions, anything finite elsewhere).
pub struct Block<'a> {
    pub p0: usize,
    pub entry: &'a dyn Fn(usize) -> usize,
}

/// Tiles a block at `p0` needs: its history up to `W` positions and its own 128.
pub fn tiles_needed(p0: usize) -> usize {
    (p0 / 128 + 1).min(TILES)
}

/// A round's tables for four blocks over windows of `tiles` tiles (`None`: an unused block, its entries at `spread`
/// rows): the gather offsets (E4M3 pools, `[N][T][W%8][Bb%2][W/8%16][Bb/2]`), the tile-mask keys and the cap keys
/// (`[N][Bb][16]`). A block's window is the `128 tiles` positions ending at `p0 + 127`; `tiles` must hold its need.
pub fn round_tables(tiles: usize, blocks: [Option<&Block<'_>>; 4], spread: &dyn Fn(usize) -> usize) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    assert!((1..=TILES).contains(&tiles));
    let first = |b: usize| blocks[b].map(|blk| {
        assert!(blk.p0 >= super::window::SLIDING_WINDOW || blk.p0 % 128 == 0, "a block below the window starts at a multiple of 128");
        assert!(tiles_needed(blk.p0) <= tiles, "a block at {} needs {} tiles, the round has {tiles}", blk.p0, tiles_needed(blk.p0));
        blk.p0 as isize + 128 - 128 * tiles as isize
    });
    // Leading tiles before position 0 (an unused block: all but the last).
    let v = |b: usize| first(b).map_or(tiles - 1, |f| if f >= 0 { 0 } else { (-f) as usize / 128 });
    let mut off = Vec::with_capacity(8 * 128 * tiles * 4);
    for n in 0..8 {
        for t in 0..tiles {
            for w8 in 0..8 {
                for b2 in 0..2 {
                    for sl in 0..16 {
                        for bh in 0..2 {
                            let b = 2 * bh + b2;
                            let e = 128 * t + 8 * sl + w8;
                            let row = match (blocks[b], first(b)) {
                                (Some(blk), Some(f)) if f + e as isize >= 0 => (blk.entry)((f + e as isize) as usize),
                                _ => spread(4 * e + b),
                            };
                            off.push(((row * 8 + n) * D::SIZE) as i32);
                        }
                    }
                }
            }
        }
    }
    let mut tm = Vec::with_capacity(8 * 4 * 16);
    let mut cap = Vec::with_capacity(8 * 4 * 16);
    for _n in 0..8 {
        for b in 0..4 {
            for sl in 0..16 {
                let vb = v(b);
                tm.push((vb * 32 * 8 * 4) as i32);
                // Recomputed edge scores must also preserve leading masked tiles.
                let key = if vb >= 2 {
                    17
                } else if vb == 1 {
                    16
                } else if tiles == TILES {
                    sl
                } else {
                    18
                };
                cap.push((key * Edge::SIZE * Wt::SIZE * 8 * 4) as i32);
            }
        }
    }
    (off, tm, cap)
}

// S4 arithmetic is the default for every prompt-block entry and resident caller.
pub(crate) use super::prompt_pattn::{round_w1, round_w2, round_w3b, round_w4r};
#[cfg(feature = "compile-carried")]
pub(crate) use super::prompt_pattn::round_w1_nomasks;
