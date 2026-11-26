//! Sliding prefill over E4M3 caches.
//!
//! A chunk of `c` queries at positions `K.. K + c` attends to a gathered list of `min(K, 1024)` cached positions
//! followed by the chunk's own, in position order. Each round computes 512 queries.
//! - **Blocks.** Each PE owns one KV head with its two query heads. Its 64 slices form four 16-slice sub-rings,
//!   and each sub-ring takes a block of 128 queries: 8 per slice on the 8 TRF lanes, the two heads as a time
//!   repeat of every key packet (the stream reaches the contraction at one flit per two cycles and the multiplier
//!   takes two uses of a packet in that time, so one slice processes 16 query heads).
//! - **Windows.** Block `b` (chunk queries `128 b..`, list end `E = min(K, 1024) + 128 (b + 1)`) sees the window of
//!   `W` list entries ending at `E`, gathered into its sub-ring (8 positions of every 128-entry tile per slice) and
//!   all-gathered around the ring into a stream every slice of the block shares. With `K >= SLIDING_WINDOW` (a band
//!   bucket) `W = SLIDING_WINDOW + 128` (`window::TILES` tiles); with `K + c <= 1024` (a
//!   causal bucket, rows below the window only) round `r`'s window is `W_r = K + 512 (r + 1)`, the whole
//!   history of its last block, and a block's entries before the list's start are padding (`kv_off` points them at
//!   spread-out slots, `tmask` masks their tiles).
//! - **Masks.** Scores are clipped to per-slice caps: `tmask` per tile (`-1e30` for a padding tile), `cap_hi` for
//!   the last tile (the causal diagonal) and, in a band bucket, `cap_lo` for the first (the window's start).
//! - **Softmax** in f32 within the slice, over the window in position order, so a row's bits depend only on its
//!   position and keys, not on its chunk (masked entries add exact zeros).
//! - **PV** uses stationary E4M3(256 × exp(score − max)) and streams E4M3 values.
//!   Divide by 256 and by the FP32 sum of unrounded exponentials, then cast to BF16.
//!
//! `dsl_prefill_sliding!` stamps a bucket: the KV element type, `c`, whether it is a band bucket, the block factor of
//! the slice mapping (padded for `c < 512`), and per round its window axis, tile count and input idents.

/// Decode gathered E4M3 KV to BF16.
#[macro_export]
macro_rules! dsl_kv_decode {
    (f8e4m3, $device:ident, $x:ident => $k:ident) => {
        let owned = dequant($device, $x);
        let $k = &owned;
    };
}

/// One Sliding prefill bucket. `$c` is the chunk size and `$band` controls whether the window's
/// start is masked (`K >= 1024`); `block` the sub-ring factor of the slice mapping (`Q / 128 % 4`, or padded to 4
/// for smaller chunks); `rounds` their count; then per round of 512 queries its index, its window axis
/// (`name = size`, a multiple of 128), its tile count and the idents of its offset and tile-mask inputs and of its
/// gathered K and V.
#[macro_export]
macro_rules! dsl_prefill_sliding {
    ($module:ident, $name:ident, $kv:ident, $c:literal, band = $band:tt, block = [$($bq:tt)*], rounds = $nr:literal,
     [$(($r:literal, $wn:ident = $w:literal, tiles = $t:tt, mid = $mid:tt, $off:ident, $tm:ident, $kg:ident, $vg:ident)),* $(,)?]) => {
        pub mod $module {
            use furiosa_opt_std::prelude::*;

            axes![
                N = 8,       // KV heads: one per PE (2 clusters x 4 PEs)
                G2 = 2,      // query heads per KV head
                D = 256,     // head dim
                Q = $c,      // chunk queries: Q / 512 round, Q / 128 % 4 sub-ring, Q / 8 % 16 slice, Q % 8 lane
                Tp = 32,
                Bb = 4,      // a round's blocks
                Wt = 128,    // a window tile (edge caps): position 8 x source slice + position in it
                Sh = 835584, // cache rows of one KV head: pool row x 8 + KV head (`axes::Rs` x 8)
                $($wn = $w),*
            ];

            const _: () = assert!(Sh::SIZE == crate::axes::Rs::SIZE * 8, "Sh is the Sliding pool's rows x KV heads");

            pub type Chip = m![1];
            /// One KV head per PE.
            pub type C = m![N / 4];
            /// Query slices: the PE, its sub-ring (block), 8 queries per slice.
            pub type S = m![N % 4, $($bq)*, Q / 8 % 16];
            /// The lanes: a slice's 8 queries.
            pub type L = m![Q % 8];
            /// Per-slice clip of one 128-entry edge tile, per lane.
            pub type Cap = DmTensor<f32, Chip, C, S, m![Wt, Q % 8]>;
            /// Key slices of a window: the sub-ring's slices, each holding 8 positions of every 128-entry tile.
            pub type SK<Wn> = m![N % 4, $($bq)*, Wn / 8 % 16];
            /// A round's tile masks in HBM: per slice, window tile and lane (only the middle tiles are read).
            pub type TmMap<Wn> = m![N, $($bq)*, Q / 8 % 16, Wn / 128 # 32, Q % 8];
            pub type TmHbm<Wn> = HbmTensor<f32, Chip, TmMap<Wn>>;

            /// Sliding prefill of `c` queries.
            ///
            /// - `q` `[Q, N, G2, D]` bf16: query head `2n + g` reads KV head `n`. Scale 1.0.
            /// - Per round `r`, `off_r` `[N][W_r / 128][W_r % 8][b % 2][W_r / 8 % 16][b / 2]` i32: for KV head `n` and the
            ///   round's block `b` (chunk block `4 r + b`; a block past the chunk repeats block 0), the byte offset
            ///   (`slot · row + n · head`) of list entry `E_b - W_r + w` (the gather's walk order); entries before the
            ///   list start point at spread-out slots.
            /// - Per round, `tmask_r` `[N, blocks, 16, W_r / 128 # 32, Q % 8]` f32: per slice, tile and lane, `-1e30` for a
            ///  padding tile, else `+MAX` (the lane axis repeats the value; only the middle tiles are read).
            /// - `cap_lo`, `cap_hi` `[N, blocks, 16, Wt, Q % 8]` f32: per slice and lane, the clip of the window's first
            ///   and last tile (`+MAX` where the query attends, `-1e30` where it does not).
            /// - `k_cache`, `v_cache` `[Sh, D]`: the `[slot][N][D]` cache read as rows of one KV head.
            #[device(chip = 1)]
            pub fn $name(
                device: &mut Device,
                q: &HbmTensor<bf16, Chip, m![Q, N, G2, D]>,
                $($off: &HbmTensor<i32, Chip, m![N, $wn / 128, $wn % 8, Bb % 2, $wn / 8 % 16, Bb / 2]>,)*
                $($tm: &TmHbm<$wn>,)*
                cap_lo: &HbmTensor<f32, Chip, m![N, $($bq)*, Q / 8 % 16, Wt, Q % 8]>,
                cap_hi: &HbmTensor<f32, Chip, m![N, $($bq)*, Q / 8 % 16, Wt, Q % 8]>,
                k_cache: &HbmTensor<$kv, Chip, m![Sh, D]>,
                v_cache: &HbmTensor<$kv, Chip, m![Sh, D]>,
            ) -> HbmTensor<bf16, Chip, m![Q, N, G2, D]> {
                // The queries first (the first QKᵀ needs them with the first keys), then every round's windows,
                // gathered into their sub-rings: later rounds' gathers run under earlier rounds' work.
                let q_dm: DmTensor<bf16, Chip, C, S, m![Q / 512, Q % 8, G2, D]> = q.to_dm(&mut device.tdma);
                $(
                    let $kg: DmTensor<$kv, Chip, C, m![N % 4, Bb, $wn / 8 % 16], m![$wn / 128, $wn % 8, D]> =
                        k_cache.gather::<m![Sh], m![D]>().by_byte_offsets($off).to_dm(&mut device.tdma);
                    let $vg: DmTensor<$kv, Chip, C, m![N % 4, Bb, $wn / 8 % 16], m![$wn / 128, $wn % 8, D]> =
                        v_cache.gather::<m![Sh], m![D]>().by_byte_offsets($off).to_dm(&mut device.tdma);
                )*
                let cap_lo: Cap = cap_lo.to_dm(&mut device.tdma);
                let cap_hi: Cap = cap_hi.to_dm(&mut device.tdma);
                let mut out = HbmTensor::<bf16, Chip, m![Q, N, G2, D]>::new();
                $(
                    let tm: DmTensor<f32, Chip, C, S, m![$wn / 128 # 32, Q % 8]> = $tm.to_dm(&mut device.tdma);
                    $crate::pattn_prefill_round!($band, $wn, $nr, $t, $mid, device, $r, q_dm, $kg, $vg, tm, cap_lo, cap_hi, out);
                )*
                out
            }


        }
    };
}

/// `$body` in a band bucket only, chosen when the macro expands: Opt 0.9.0 lowers an `if` on a constant to a run-time
/// guard.
#[macro_export]
macro_rules! dsl_band_only {
    (true, { $($body:tt)* }) => { { $($body)* } };
    (false, { $($body:tt)* }) => {};
}

// The two middle band tiles are unclipped.
dsl_prefill_sliding!(band512_fp8, swa_prefill_fp8_band_c512, f8e4m3, 512, band = true, block = [Q / 128 % 4], rounds = 1,
    [(0, W0 = 512, tiles = 4, mid = (1, 2), off0, tm0, k0, v0)]);
dsl_prefill_sliding!(w128_fp8, swa_prefill_fp8_w128_c512, f8e4m3, 512, band = false, block = [Q / 128 % 4], rounds = 1,
    [(0, W0 = 128, tiles = 1, mid = none, off0, tm0, k0, v0)]);
dsl_prefill_sliding!(w384_fp8, swa_prefill_fp8_w384_c512, f8e4m3, 512, band = false, block = [Q / 128 % 4], rounds = 1,
    [(0, W0 = 384, tiles = 3, mid = all, off0, tm0, k0, v0)]);
