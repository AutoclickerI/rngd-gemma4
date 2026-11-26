//! Sliding decode: 8 rows of 16 query heads over
//! 8 KV heads of 256, one row per PE. A row's window positions spread over its PE's 64 slices, `Pi` per slice; the
//! queries stay in the TRF, the keys stream past them (E4M3 decoded in the fetch), the softmax is f32, and P stays
//! stationary in PV. The engine right-aligns a row's visible positions (its newest at slice 63, index `Pi - 1`;
//! `host::engine::dsl`), so every window variant sums a row in the same order.

/// Sliding decode over a window of `64 · $pi` positions (`$pi` per slice); the host picks the variant by the
/// longest row of the launch (rows grouped by context keep the rest short). `$kv_per_flit` KV heads' probabilities
/// fill one 32-byte flit (`$pi · $kv_per_flit = 16`).
macro_rules! dsl_sliding_decode {
    ($module:ident, $name:ident, $kv:ident, $pi:literal, $pq:literal, $cq:literal, $kv_per_flit:literal, $p_flit:ty, $p_elem:ty, $p_dm:ty, $p_commit:ty, $p_fetch:ty) => {
        pub mod $module {
            use furiosa_opt_std::prelude::*;

        axes![
            Rw = 8,      // rows per launch: one per PE (2 clusters x 4 PEs)
            Ns = 8,      // KV heads
            Gs = 2,      // query heads per KV head
            Ds = 256,    // head dim
            Po = 64,     // a row's position slices
            Pi = $pi,    // positions per slice (window = Po x Pi)
            Sl = 104448, // cache slots: the Sliding pool's rows (`axes::Rs`)
            Px = 64      // the row's position slices, as a fresh axis for broadcasts
        ];

        const _: () = assert!(Sl::SIZE == crate::axes::Rs::SIZE, "Sl is the Sliding pool's rows");

        pub type Chip = m![1];
        /// One row per PE: rows 4c..4c+3 on cluster c.
        pub type C = m![Rw / 4];
        /// A row's 64 slices hold its positions 16 apiece.
        pub type S = m![Rw % 4, Po];
        /// The row's first slice (results reduced over the row's slices).
        pub type S1 = m![Rw % 4, 1 # 64];
        /// The row's 64 slices as a broadcast axis.
        pub type SX = m![Rw % 4, Px];

        /// Sliding decode over E4M3 caches.
        ///
        /// - `q` `[Rw, Ns, Gs, Ds]` bf16: query head `Gs·n + g` reads KV head `n`.
        /// - `kv_off` `[Rw, Pi, Po % 32, Po / 32]` i32: byte offset (`slot · 2048`) of window position `Pi·Po + Pi`, in the
        ///   order the gather walks its list, so the list copy into SPM is one contiguous move.
        /// - `cap` `[Rw, Po, Pi]` f32: `+MAX` where the row attends, `-1e30` elsewhere (scores are clipped to it).
        /// - `k_cache`, `v_cache` `[Sl, Ns, Ds]` E4M3.
        ///
        /// Returns `[Rw, Ns, Gs, Ds]` bf16. Scale 1.0 (folded into q by the caller).
        #[device(chip = 1)]
        pub fn $name(
            device: &mut Device,
            q: &HbmTensor<bf16, Chip, m![Rw, Ns, Gs, Ds]>,
            kv_off: &HbmTensor<i32, Chip, m![Rw, Pi, Po % 32, Po / 32]>,
            cap: &HbmTensor<f32, Chip, m![Rw, Po, Pi]>,
            k_cache: &HbmTensor<$kv, Chip, m![Sl, Ns, Ds]>,
            v_cache: &HbmTensor<$kv, Chip, m![Sl, Ns, Ds]>,
        ) -> HbmTensor<bf16, Chip, m![Rw, Ns, Gs, Ds]> {
            // The row's query, loaded once (64 elements per slice of the row) and all-gathered around the row's 64-slice
            // ring into the TRF: the two query heads of a KV head on two lanes, the KV head and the head dim in the
            // element.
            let q_dm: DmTensor<bf16, Chip, C, m![Rw % 4, Gs, Ns, Ds / 64], m![Ds % 64]> = q.to_dm(&mut device.tdma);
            let q_trf: TrfTensor<bf16, Chip, C, S, m![Gs], m![Ns, Ds]> = device
                .sub
                .begin(q_dm.view())
                .fetch::<m![Ds % 64 / 16], m![Ds % 16]>()
                .switch::<m![Rw % 4, Po], m![Gs, Ns, Ds / 64, Ds % 64 / 16]>(SwitchConfig::Broadcast01 {
                    slice1: 64,
                    slice0: 1,
                    time0: 4,
                })
                .collect::<m![Gs, Ns, Ds / 16], m![Ds % 16]>()
                .to_trf();

            // Clip value per position.
            let cap_dm: DmTensor<f32, Chip, C, S, m![Pi]> = cap.to_dm(&mut device.tdma);
            let cap_vrf: VrfTensor<f32, Chip, C, S, m![Pi / $cq, Pi % $cq # 8]> = device
                .sub
                .begin(cap_dm.view())
                .fetch::<m![Pi / $cq], m![Pi % $cq]>()
                .collect::<m![Pi / $cq], m![Pi % $cq # 8]>()
                .to_vrf();

            // Keys and values: one 2 KiB E4M3 slot per position.
            let k: DmTensor<$kv, Chip, C, S, m![Pi, Ns, Ds]> = k_cache.gather::<m![Sl], m![Ns, Ds]>().by_byte_offsets(kv_off).to_dm(&mut device.tdma);
            let v: DmTensor<$kv, Chip, C, S, m![Pi, Ns, Ds]> = v_cache.gather::<m![Sl], m![Ns, Ds]>().by_byte_offsets(kv_off).to_dm(&mut device.tdma);

            // 1. Clipped scores [position, KV head, head] in f32. The fetch reads 16 E4M3 bytes and the table turns
            // them into one 32-byte bf16 flit; two flits make a 64-byte contraction packet.
            let s: DmTensor<f32, Chip, C, S, m![Pi, Ns, Gs]> = dsl_lut!($kv, device.main.begin(k.view()).fetch::<m![Pi, Ns, Ds / 16], m![Ds % 16]>())
                .collect::<m![Pi, Ns, Ds / 16], m![Ds % 16]>()
                .contract_outer::<m![Pi, Ns, Ds / 32], m![Ds % 32], _, _, _>(&q_trf)
                .contract_packet::<m![1]>()
                .contract_time::<m![Pi, Ns]>()
                .contract_lane::<m![Pi, Ns], m![Gs # 8]>(LaneMode::Interleaved)
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_clip(ClipBinaryOpF32::Min, &cap_vrf)
                .vector_final()
                .commit_trim::<m![Gs]>()
                .commit();

            // 2. Max per head: over the slice's 16 positions, then over the row's 64 slices, broadcast back.
            let slice_max: DmTensor<f32, Chip, C, S, m![Ns, Gs]> = device
                .main
                .begin(s.view())
                .fetch::<m![Ns / 4, Pi], m![Ns % 4, Gs]>()
                .collect::<m![Ns / 4, Pi], m![Ns % 4, Gs]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Ns / 4, Pi, Ns % 4 / 2], m![Ns % 2, Gs]>()
                .vector_intra_slice_reduce::<Pi, m![Ns / 4, Ns % 4 / 2], m![Ns % 2, Gs]>(IntraSliceReduceOpF32::Max)
                .vector_widen_concat::<m![Ns / 4], m![Ns % 4, Gs]>()
                .vector_final()
                .commit_trim::<m![Ns % 4, Gs]>()
                .commit();
            let max: DmTensor<f32, Chip, C, SX, m![Ns, Gs]> = device
                .main
                .begin(slice_max.view())
                .fetch::<m![Ns / 4], m![Ns % 4, Gs]>()
                .collect::<m![Ns / 4], m![Ns % 4, Gs]>()
                .vector_init()
                .vector_inter_slice_reduce::<SX, m![Ns / 4]>(InterSliceReduceOpF32::Max)
                .vector_final()
                .commit_trim::<m![Ns % 4, Gs]>()
                .commit();
            let max: DmTensor<f32, Chip, C, S, m![Ns, Gs]> = unsafe { max.reshape() };
            let max_vrf: VrfTensor<f32, Chip, C, S, m![Ns, Gs]> = device
                .sub
                .begin(max.view())
                .fetch::<m![Ns / 4], m![Ns % 4, Gs]>()
                .collect::<m![Ns / 4], m![Ns % 4, Gs]>()
                .to_vrf();

            // 3. Probabilities exp(s - max) in bf16, transposed to [head, KV head, position] for the TRF.
            let p: DmTensor<bf16, Chip, C, S, $p_dm> = device
                .main
                .begin(s.view())
                .fetch::<m![Pi / $pq, Ns / 4, Pi % $pq], m![Ns % 4, Gs]>()
                .collect::<m![Pi / $pq, Ns / 4, Pi % $pq], m![Ns % 4, Gs]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Pi / $pq, Ns / 4, Pi % $pq, Ns % 4 / 2], m![Ns % 2, Gs]>()
                .vector_fp_binary(FpBinaryOp::SubF, &max_vrf)
                .vector_fp_unary(FpUnaryOp::Exp)
                .vector_widen_concat::<m![Pi / $pq, Ns / 4, Pi % $pq], m![Ns % 4, Gs]>()
                .vector_final()
                .cast::<bf16, m![Ns % 4 # 8, Gs]>()
                .transpose::<m![Pi / $pq, Ns / 4, Ns % 4, Gs], m![Pi % $pq # 16]>()
                .commit_trim::<$p_commit>()
                .commit();

            // Sum of the same exponentials in f32: over the slice's positions, then over the row's slices.
            let slice_sum: DmTensor<f32, Chip, C, S, m![Ns, Gs]> = device
                .main
                .begin(s.view())
                .fetch::<m![Ns / 4, Pi], m![Ns % 4, Gs]>()
                .collect::<m![Ns / 4, Pi], m![Ns % 4, Gs]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Ns / 4, Pi, Ns % 4 / 2], m![Ns % 2, Gs]>()
                .vector_fp_binary(FpBinaryOp::SubF, &max_vrf)
                .vector_fp_unary(FpUnaryOp::Exp)
                .vector_intra_slice_reduce::<Pi, m![Ns / 4, Ns % 4 / 2], m![Ns % 2, Gs]>(IntraSliceReduceOpF32::Add)
                .vector_widen_concat::<m![Ns / 4], m![Ns % 4, Gs]>()
                .vector_final()
                .commit_trim::<m![Ns % 4, Gs]>()
                .commit();
            let sum: DmTensor<f32, Chip, C, S1, m![Ns, Gs]> = device
                .main
                .begin(slice_sum.view())
                .fetch::<m![Ns / 4], m![Ns % 4, Gs]>()
                .collect::<m![Ns / 4], m![Ns % 4, Gs]>()
                .vector_init()
                .vector_inter_slice_reduce::<S1, m![Ns / 4]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .commit_trim::<m![Ns % 4, Gs]>()
                .commit();

            // 4. PV: the probabilities stationary (two heads on two lanes), the values decoded in the fetch and
            // streamed [KV head, d-chunk, position]; positions accumulate in the time reducer, slices fold after.
            let p_trf: TrfTensor<bf16, Chip, C, S, m![Gs], $p_elem> = device
                .sub
                .begin(p.view())
                .fetch::<m![Gs, Ns / $kv_per_flit], $p_fetch>()
                .collect::<m![Gs, Ns / $kv_per_flit], $p_flit>()
                .to_trf();
            let o: DmTensor<f32, Chip, C, S1, m![Ns, Ds / 32, Gs, Ds % 32]> = dsl_lut!($kv, device.main.begin(v.view()).fetch::<m![Ns, Ds / 32, Pi, Ds / 16 % 2], m![Ds % 16]>())
                .collect::<m![Ns, Ds / 32, Pi, Ds / 16 % 2], m![Ds % 16]>()
                .contract_outer::<m![Ns, Ds / 32, Pi], m![Ds % 32], _, _, _>(&p_trf)
                .contract_packet::<m![Ds % 32]>()
                .contract_time::<m![Ns, Ds / 32]>()
                .contract_lane::<m![Ns, Ds / 32, Gs, Ds % 32 / 8], m![Ds % 8]>(LaneMode::Sequential)
                .vector_init()
                .vector_inter_slice_reduce::<S1, m![Ns, Ds / 32, Gs, Ds % 32 / 8]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .commit_trim::<m![Ds % 8]>()
                .commit();

            // 5. Normalize, round, reorder to [KV head, head, d].
            let sum_vrf: VrfTensor<f32, Chip, C, S1, m![Ns, Gs]> = device
                .sub
                .begin(sum.view())
                .fetch::<m![Ns / 4], m![Ns % 4, Gs]>()
                .collect::<m![Ns / 4], m![Ns % 4, Gs]>()
                .to_vrf();
            let out: DmTensor<bf16, Chip, C, S1, m![Ns, Gs, Ds]> = device
                .main
                .begin(o.view())
                .fetch::<m![Ns, Gs, Ds / 8], m![Ds % 8]>()
                .collect::<m![Ns, Gs, Ds / 8], m![Ds % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Ns, Gs, Ds / 4], m![Ds % 4]>()
                .vector_fp_div(&sum_vrf)
                .vector_widen_concat::<m![Ns, Gs, Ds / 8], m![Ds % 8]>()
                .vector_final()
                .cast::<bf16, m![Ds % 8 # 16]>()
                .commit_trim::<m![Ds % 8]>()
                .commit();
            out.to_hbm(&mut device.tdma)
        }

        }
    };
}

dsl_sliding_decode!(w256_fp8, swa_decode_fp8_w256, f8e4m3, 4, 4, 4, 4, m![Ns % 4, Pi], m![Ns, Pi], m![Gs, Ns, Pi], m![Pi % 4], m![Ns % 4, Pi]);
// W 384: 6 positions per slice. The P transposition takes at most 4 input rows (Opt: `max_in_rows`), so the positions
// split 2 x 3; each 3-position group commits padded to 4 (8 bytes: a 6-byte packet is not a commit size) and two KV
// heads' 2 x 4 fill the 32-byte TRF flit.
dsl_sliding_decode!(w384_fp8, swa_decode_fp8_w384, f8e4m3, 6, 3, 2, 2, m![Ns % 2, Pi / 3, Pi % 3 # 4], m![Ns, Pi / 3, Pi % 3 # 4], m![Gs, Ns, Pi / 3, Pi % 3 # 4], m![Pi % 3 # 4], m![Ns % 2, Pi / 3, Pi % 3 # 4]);
