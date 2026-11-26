//! Full causal prefill over E4M3 caches.
//!
//! A chunk of `c` queries at positions `K.. K + c` attends to the list of its `K` cached positions then its own.
//! The layer's one KV head of 512 serves all 16 query heads, so the work splits by query rather than by head:
//! - **Blocks.** PE `p` takes head half `p / 4` (the cluster) of query quarter `p % 4`: in round `r` its block is
//!   the 128 chunk queries `128 (4 r + p % 4)..`. Each slice uses each key packet for two
//!   queries and the TRF lanes hold its eight heads.
//! - **Windows.** Round `r`'s window is `W_r = K + 512 (r + 1)`, the whole history of its last block; block `b`'s
//!   window ends at its list end `E = K + 128 (b + 1)`, and entries before the list are padding (spread-out slots,
//!   masked by `tmask`). The window is gathered two positions per slice per 128-entry tile and all-gathered around
//!   the PE's 64 slices.
//! - **Masks, softmax, PV** as in `dsl_prefill_sliding!`: `tmask` per tile, `cap_hi` on the last tile (the causal
//!   diagonal: window entry `w` of the last tile is visible to query `2 s + g` of slice `s` where `w <= 2 s + g`),
//!   f32 softmax in position order, P stationary in PV.

/// One Full prefill bucket. `$kv` is `f8e4m3` or `bf16`; `$c` the chunk rows; `block` the quarter factor of the
/// slice mapping (`Q / 128 % 4`, or padded to 4 for smaller chunks); then per round of 512 queries its index, its
/// window axis (`name = size`, `K + 512 (r + 1)`), its tile count and the idents of its offset and tile-mask inputs
/// and of its gathered K and V.
#[macro_export]
macro_rules! dsl_prefill_full {
    ($module:ident, $name:ident, $kv:ident, $c:literal, block = [$($bq:tt)*], rounds = $nr:literal,
     [$(($r:literal, $wn:ident = $w:literal, tiles = $t:literal, $off:ident, $tm:ident, $kg:ident, $vg:ident)),* $(,)?]) => {
        pub mod $module {
            use furiosa_opt_std::prelude::*;

            axes![
                N = 8,       // PEs: head half N / 4 (the cluster), query quarter N % 4
                Hq = 16,     // query heads: Hq / 8 the half, Hq % 8 the lane
                D = 512,     // head dim
                Q = $c,      // chunk queries: Q / 512 round, Q / 128 % 4 quarter, Q / 2 % 64 slice, Q % 2 group
                Bb = 4,      // a round's blocks
                Wt = 128,    // a window tile (edge cap): position 2 x source slice + position in it
                Sh = 263168, // cache rows: the Full pool's rows (`axes::Rf`)
                $($wn = $w),*
            ];

            const _: () = assert!(Sh::SIZE == crate::axes::Rf::SIZE, "Sh is the Full pool's rows");

            pub type Chip = m![1];
            /// Cluster = head half.
            pub type C = m![Hq / 8];
            /// Query slices: the PE (query quarter), 2 queries per slice.
            pub type S = m![$($bq)*, Q / 2 % 64];
            /// The lanes: the PE's 8 query heads.
            pub type L = m![Hq % 8];
            /// Key slices of a window: the PE's slices, each holding 2 positions of every 128-entry tile.
            pub type SK<Wn> = m![$($bq)*, Wn / 2 % 64];
            /// Per-slice clip of the window's last tile, per query group.
            pub type Cap = DmTensor<f32, Chip, C, S, m![Wt, Q % 2]>;
            /// A round's tile masks in HBM: per slice, tile (padded to 32) and lane.
            pub type TmMap<Wn> = m![Hq / 8, $($bq)*, Q / 2 % 64, Wn / 128 # 32, Hq % 8];
            pub type TmHbm<Wn> = HbmTensor<f32, Chip, TmMap<Wn>>;
            /// The causal diagonal's cap in HBM: per slice, position of the last tile and query group.
            pub type CapMap = m![Hq / 8, $($bq)*, Q / 2 % 64, Wt, Q % 2];
            pub type CapHbm = HbmTensor<f32, Chip, CapMap>;

            /// Full prefill of `c` queries.
            ///
            /// - `q` `[Q, Hq, D]` bf16. Scale 1.0.
            /// - Per round `r`, `off_r` `[N][W_r / 128][W_r % 2][W_r / 2 % 32][W_r / 64 % 2]` i32: for PE `n` (its block
            ///   `4 r + n % 4`; a block past the chunk repeats block 0), the byte offset `slot · row` of list entry
            ///   `E - W_r + w` in the gather's walk order (slice `w / 2 % 64` as `(w / 2 % 32, w / 64 % 2)`); entries
            ///   before the list start point at spread-out slots.
            /// - Per round, `tmask_r` `[Hq / 8, quarters, Q / 2 % 64, W_r / 128 # 32, Hq % 8]` f32: per slice, tile and
            ///  lane, `-1e30` for a padding tile, else `+MAX`.
            /// - `cap_hi` `[Hq / 8, quarters, Q / 2 % 64, Wt, Q % 2]` f32: the last tile's clip per slice and query
            ///   group, `+MAX` where `w <= 2 (Q / 2 % 64) + Q % 2`.
            /// - `k_cache`, `v_cache` `[Sh, D]`.
            #[device(chip = 1)]
            pub fn $name(
                device: &mut Device,
                q: &HbmTensor<bf16, Chip, m![Q, Hq, D]>,
                $($off: &HbmTensor<i32, Chip, m![N, $wn / 128, $wn % 2, $wn / 2 % 32, $wn / 64 % 2]>,)*
                $($tm: &TmHbm<$wn>,)*
                cap_hi: &CapHbm,
                k_cache: &HbmTensor<$kv, Chip, m![Sh, D]>,
                v_cache: &HbmTensor<$kv, Chip, m![Sh, D]>,
            ) -> HbmTensor<bf16, Chip, m![Q, Hq, D]> {
                $(
                    let $kg: DmTensor<$kv, Chip, m![N / 4], m![N % 4, $wn / 2 % 64], m![$wn / 128, $wn % 2, D]> =
                        k_cache.gather::<m![Sh], m![D]>().by_byte_offsets($off).to_dm(&mut device.tdma);
                    let $vg: DmTensor<$kv, Chip, m![N / 4], m![N % 4, $wn / 2 % 64], m![$wn / 128, $wn % 2, D]> =
                        v_cache.gather::<m![Sh], m![D]>().by_byte_offsets($off).to_dm(&mut device.tdma);
                )*
                let q_dm: DmTensor<bf16, Chip, C, S, m![Q / 512, Hq % 8, Q % 2, D]> = q.to_dm(&mut device.tdma);
                let cap_hi: Cap = cap_hi.to_dm(&mut device.tdma);
                let mut out = HbmTensor::<bf16, Chip, m![Q, Hq, D]>::new();
                $(
                    let tm: DmTensor<f32, Chip, C, S, m![$wn / 128 # 32, Hq % 8]> = $tm.to_dm(&mut device.tdma);
                    $crate::dsl_prefill_full_round!($kv, $wn, $nr, $t, device, $r, q_dm, $kg, $vg, tm, cap_hi, out);
                )*
                out
            }


            fn dequant<Wn: AxisName>(
                device: &mut Device,
                x: &DmTensor<f8e4m3, Chip, C, SK<Wn>, m![Wn / 128, Wn % 2, D]>,
            ) -> DmTensor<bf16, Chip, C, SK<Wn>, m![Wn / 128, Wn % 2, D]> {
                device
                    .main
                    .begin(x.view())
                    .fetch::<m![Wn / 128, Wn % 2, D / 16], m![D % 16]>()
                    .fetch_table_lookup::<bf16>()
                    .collect::<m![Wn / 128, Wn % 2, D / 16], m![D % 16]>()
                    .commit_trim::<m![D % 16]>()
                    .commit()
            }

        }
    };
}

/// One round of `dsl_prefill_full!`.
#[macro_export]
macro_rules! dsl_prefill_full_round {
    ($kv:ident, $wn:ident, $nr:literal, $t:literal, $device:ident, $r:literal, $q_dm:ident, $kg:ident, $vg:ident,
     $tm:ident, $cap_hi:ident, $out:ident) => {
        // The round's two queries per slice: lane = head, element = [query group, d].
        let q_trf: TrfTensor<bf16, Chip, C, S, L, m![Q % 2, D]> = $device
            .sub
            .begin($q_dm.view().tile::<m![Q / 512], 1, m![1 # $nr, Hq % 8, Q % 2, D]>($r))
            .fetch::<m![Hq % 8, Q % 2, D / 16], m![D % 16]>()
            .collect::<m![Hq % 8, Q % 2, D / 16], m![D % 16]>()
            .to_trf();
        let kg: DmTensor<$kv, Chip, C, SK<$wn>, m![$wn / 128, $wn % 2, D]> = unsafe { $kg.reshape() };
        let vg: DmTensor<$kv, Chip, C, SK<$wn>, m![$wn / 128, $wn % 2, D]> = unsafe { $vg.reshape() };
        let kg = &kg;
        let vg = &vg;
        $crate::dsl_kv_decode!($kv, $device, kg => k);
        $crate::dsl_kv_decode!($kv, $device, vg => v);

        // 1. Scores [position, query group] x [head] in f32, clipped per tile; the all-gather sends one key (32 flits)
        //    per source slice at a time, repeated for the two queries.
        let tm_vrf: VrfTensor<f32, Chip, C, S, m![$wn / 128 # 32, Hq % 8]> = $device
            .sub
            .begin($tm.view())
            .fetch::<m![$wn / 128 # 32], L>()
            .collect::<m![$wn / 128 # 32], L>()
            .to_vrf();
        let mut s: DmTensor<f32, Chip, C, S, m![$wn, Q % 2, Hq % 8]> = $device
            .main
            .begin(k.view())
            .fetch::<m![$wn / 128, $wn % 2, D / 16], m![D % 16]>()
            .switch::<S, m![$wn / 128, $wn % 2, $wn / 2 % 64, D / 16]>(SwitchConfig::Broadcast01 { slice1: 64, slice0: 1, time0: 32 })
            .collect::<m![$wn / 128, $wn % 2, $wn / 2 % 64, D / 16], m![D % 16]>()
            .contract_outer::<m![$wn / 128, $wn % 2, $wn / 2 % 64, D / 32, Q % 2], m![D % 32], _, _, _>(&q_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![$wn / 128, $wn % 2, $wn / 2 % 64, Q % 2]>()
            .contract_lane::<m![$wn / 128, $wn % 2, $wn / 2 % 64, Q % 2], L>(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Min, &tm_vrf)
            .vector_final()
            .commit_trim::<L>()
            .commit();
        // The last tile, recomputed with the causal diagonal's cap (per query group, broadcast over the lanes).
        let cap_vrf: VrfTensor<f32, Chip, C, S, m![$wn % 128, Q % 2]> = $device
            .sub
            .begin(unsafe { $cap_hi.view().reshape::<Chip, C, S, m![$wn % 128, Q % 2]>() })
            .fetch::<m![$wn % 128 / 4], m![$wn % 128 % 4, Q % 2]>()
            .collect::<m![$wn % 128 / 4], m![$wn % 128 % 4, Q % 2]>()
            .to_vrf();
        $device
            .main
            .begin(k.view().tile::<m![$wn / 128], 1, m![1 # $t, $wn % 128 % 2, D]>($t - 1))
            .fetch::<m![$wn % 2, D / 16], m![D % 16]>()
            .switch::<S, m![$wn % 2, $wn / 2 % 64, D / 16]>(SwitchConfig::Broadcast01 { slice1: 64, slice0: 1, time0: 32 })
            .collect::<m![$wn % 2, $wn / 2 % 64, D / 16], m![D % 16]>()
            .contract_outer::<m![$wn % 2, $wn / 2 % 64, D / 32, Q % 2], m![D % 32], _, _, _>(&q_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![$wn % 2, $wn / 2 % 64, Q % 2]>()
            .contract_lane::<m![$wn % 2, $wn / 2 % 64, Q % 2], L>(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Min, &cap_vrf)
            .vector_final()
            .commit_trim::<L>()
            .commit_view(s.view_mut().tile::<m![$wn / 128], 1, m![1 #{!} $t, $wn % 128, Q % 2, Hq % 8]>($t - 1));

        // 2. Max per (query, head) over the window.
        let max: DmTensor<f32, Chip, C, S, m![Q % 2, Hq % 8]> = $device
            .main
            .begin(s.view())
            .fetch::<m![Q % 2, $wn], L>()
            .collect::<m![Q % 2, $wn], L>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Q % 2, $wn, Hq % 8 / 4], m![Hq % 4]>()
            .vector_intra_slice_reduce::<$wn, m![Q % 2, Hq % 8 / 4], m![Hq % 4]>(IntraSliceReduceOpF32::Max)
            .vector_widen_concat::<m![Q % 2], L>()
            .vector_final()
            .commit_trim::<L>()
            .commit();
        let max_vrf: VrfTensor<f32, Chip, C, S, m![Q % 2, Hq % 8]> = $device
            .sub
            .begin(max.view())
            .fetch::<m![Q % 2], L>()
            .collect::<m![Q % 2], L>()
            .to_vrf();

        // 3. P = exp(s - max) in bf16, transposed to [head, query group, position] for the TRF.
        let p: DmTensor<bf16, Chip, C, S, m![Hq % 8, Q % 2, $wn]> = $device
            .main
            .begin(s.view())
            .fetch::<m![Q % 2, $wn], L>()
            .collect::<m![Q % 2, $wn], L>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Q % 2, $wn, Hq % 8 / 4], m![Hq % 4]>()
            .vector_fp_binary(FpBinaryOp::SubF, &max_vrf)
            .vector_fp_unary(FpUnaryOp::Exp)
            .vector_widen_concat::<m![Q % 2, $wn], L>()
            .vector_final()
            .cast::<bf16, m![Hq % 8 # 16]>()
            .transpose::<m![Q % 2, $wn / 4, Hq % 8], m![$wn % 4 # 16]>()
            .commit_trim::<m![$wn % 4]>()
            .commit();

        // The sum of the same exponentials in f32, in position order.
        let sum: DmTensor<f32, Chip, C, S, m![Q % 2, Hq % 8]> = $device
            .sub
            .begin(s.view())
            .fetch::<m![Q % 2, $wn], L>()
            .collect::<m![Q % 2, $wn], L>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Q % 2, $wn, Hq % 8 / 4], m![Hq % 4]>()
            .vector_fp_binary(FpBinaryOp::SubF, &max_vrf)
            .vector_fp_unary(FpUnaryOp::Exp)
            .vector_intra_slice_reduce::<$wn, m![Q % 2, Hq % 8 / 4], m![Hq % 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_concat::<m![Q % 2], L>()
            .vector_final()
            .commit_trim::<L>()
            .commit();

        // 4. PV: P stationary (lane = head, element = [query group, position]); the values all-gathered as the keys
        //    were, in position order, with the query groups as a time repeat.
        let p_trf: TrfTensor<bf16, Chip, C, S, L, m![Q % 2, $wn]> = $device
            .sub
            .begin(p.view())
            .fetch::<m![Hq % 8, Q % 2, $wn / 16], m![$wn % 16]>()
            .collect::<m![Hq % 8, Q % 2, $wn / 16], m![$wn % 16]>()
            .to_trf();
        let o: DmTensor<f32, Chip, C, S, m![D / 32, Q % 2, Hq % 8, D % 32]> = $device
            .main
            .begin(v.view())
            .fetch::<m![D / 32, $wn / 128, $wn % 2, D / 16 % 2], m![D % 16]>()
            .switch::<S, m![D / 32, $wn / 128, $wn / 2 % 64, $wn % 2, D / 16 % 2]>(SwitchConfig::Broadcast01 {
                slice1: 64,
                slice0: 1,
                time0: 4,
            })
            .collect::<m![D / 32, $wn / 128, $wn / 2 % 64, $wn % 2, D / 16 % 2], m![D % 16]>()
            .contract_outer::<m![D / 32, $wn, Q % 2], m![D % 32], _, _, _>(&p_trf)
            .contract_packet::<m![D % 32]>()
            .contract_time::<m![D / 32, Q % 2]>()
            .contract_lane::<m![D / 32, Q % 2, Hq % 8, D % 32 / 8], m![D % 8]>(LaneMode::Sequential)
            .commit_trim::<m![D % 8]>()
            .commit();

        // 5. Normalize, round, store the round's rows.
        let sum_vrf: VrfTensor<f32, Chip, C, S, m![Q % 2, Hq % 8]> = $device
            .sub
            .begin(sum.view())
            .fetch::<m![Q % 2], L>()
            .collect::<m![Q % 2], L>()
            .to_vrf();
        let o_bf: DmTensor<bf16, Chip, C, S, m![Q % 2, Hq % 8, D]> = $device
            .main
            .begin(o.view())
            .fetch::<m![Q % 2, Hq % 8, D / 8], m![D % 8]>()
            .collect::<m![Q % 2, Hq % 8, D / 8], m![D % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Q % 2, Hq % 8, D / 4], m![D % 4]>()
            .vector_fp_div(&sum_vrf)
            .vector_widen_concat::<m![Q % 2, Hq % 8, D / 8], m![D % 8]>()
            .vector_final()
            .cast::<bf16, m![D % 8 # 16]>()
            .commit_trim::<m![D % 8]>()
            .commit();
        o_bf.view().to_hbm_view(&mut $device.tdma, $out.view_mut().tile::<m![Q / 512], 1, m![1 #{!} $nr, Q % 512, Hq, D]>($r));
    };
}

dsl_prefill_full!(w1024c1024_fp8, full_prefill_fp8_w1024_c1024, f8e4m3, 1024, block = [Q / 128 % 4], rounds = 2,
    [(0, W0 = 512, tiles = 4, off0, tm0, k0, v0), (1, W1 = 1024, tiles = 8, off1, tm1, k1, v1)]);
dsl_prefill_full!(w2048c1024_fp8, full_prefill_fp8_w2048_c1024, f8e4m3, 1024, block = [Q / 128 % 4], rounds = 2,
    [(0, W0 = 1536, tiles = 12, off0, tm0, k0, v0), (1, W1 = 2048, tiles = 16, off1, tm1, k1, v1)]);
dsl_prefill_full!(w256_fp8, full_prefill_fp8_w256_c512, f8e4m3, 512, block = [Q / 128 % 4], rounds = 1,
    [(0, W0 = 256, tiles = 2, off0, tm0, k0, v0)]);
dsl_prefill_full!(w512_fp8, full_prefill_fp8_w512_c512, f8e4m3, 512, block = [Q / 128 % 4], rounds = 1,
    [(0, W0 = 512, tiles = 4, off0, tm0, k0, v0)]);
dsl_prefill_full!(w768_fp8, full_prefill_fp8_w768_c512, f8e4m3, 512, block = [Q / 128 % 4], rounds = 1,
    [(0, W0 = 768, tiles = 6, off0, tm0, k0, v0)]);
dsl_prefill_full!(w1024_fp8, full_prefill_fp8_w1024_c512, f8e4m3, 512, block = [Q / 128 % 4], rounds = 1,
    [(0, W0 = 1024, tiles = 8, off0, tm0, k0, v0)]);
dsl_prefill_full!(w1280_fp8, full_prefill_fp8_w1280_c512, f8e4m3, 512, block = [Q / 128 % 4], rounds = 1,
    [(0, W0 = 1280, tiles = 10, off0, tm0, k0, v0)]);
dsl_prefill_full!(w1536_fp8, full_prefill_fp8_w1536_c512, f8e4m3, 512, block = [Q / 128 % 4], rounds = 1,
    [(0, W0 = 1536, tiles = 12, off0, tm0, k0, v0)]);
dsl_prefill_full!(w1792_fp8, full_prefill_fp8_w1792_c512, f8e4m3, 512, block = [Q / 128 % 4], rounds = 1,
    [(0, W0 = 1792, tiles = 14, off0, tm0, k0, v0)]);
dsl_prefill_full!(w2048_fp8, full_prefill_fp8_w2048_c512, f8e4m3, 512, block = [Q / 128 % 4], rounds = 1,
    [(0, W0 = 2048, tiles = 16, off0, tm0, k0, v0)]);
