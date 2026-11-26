//! Full decode: 8 rows of 16 query heads over the layer's one KV head of 512, one row per PE. Head
//! `h` on lane `h % 8`, the two head groups as a time repeat of each key and value packet; positions left-aligned
//! (visible position `k` at slice `k % 64`, index `k / 64`; `host::engine::dsl`).

macro_rules! full_decode_geometry {
    ($pi:literal) => {
        axes![
            Rw = 8,      // rows per launch: one per PE
            Hq = 16,     // query heads, all over the one KV head
            Df = 512,    // head dim
            Po = 64,     // a row's position slices
            Pi = $pi,    // positions per slice
            Sl = 263168, // cache slots: the Full pool's rows (`axes::Rf`)
            Px = 64      // the row's position slices, as a fresh axis for broadcasts
        ];

        const _: () = assert!(Sl::SIZE == crate::axes::Rf::SIZE, "Sl is the Full pool's rows");

        pub type Chip = m![1];
        pub type C = m![Rw / 4];
        pub type S = m![Rw % 4, Po];
        pub type S1 = m![Rw % 4, 1 # 64];
        pub type SX = m![Rw % 4, Px];
    };
}

/// Full decode over a window of `64 · $pi` positions (`$pi` per slice). `$p_time`/`$p_flit` stage the
/// probabilities `[head % 8, head / 8, position]` into the TRF one 32-byte flit at a time.
macro_rules! dsl_full_decode {
    ($module:ident, $name:ident, $kv:ident, $pi:literal, $p_time:ty, $p_flit:ty, $p_elem:ty) => {
        pub mod $module {
            use furiosa_opt_std::prelude::*;

            full_decode_geometry!($pi);

            /// - `q` `[Rw, Hq, Df]` bf16 (scale folded in).
            /// - `kv_off` `[Rw, Pi, Po % 32, Po / 32]` i32: byte offset (`slot · 512`) of window position
            ///   `Pi_size·Po + Pi`, in the gather's list order.
            /// - `cap` `[Rw, Po, Pi]` f32: `+MAX` where the row attends, `-1e30` elsewhere.
            /// - `k_cache`, `v_cache` `[Sl, Df]` E4M3.
            #[device(chip = 1)]
            pub fn $name(
                device: &mut Device,
                q: &HbmTensor<bf16, Chip, m![Rw, Hq, Df]>,
                kv_off: &HbmTensor<i32, Chip, m![Rw, Pi, Po % 32, Po / 32]>,
                cap: &HbmTensor<f32, Chip, m![Rw, Po, Pi]>,
                k_cache: &HbmTensor<$kv, Chip, m![Sl, Df]>,
                v_cache: &HbmTensor<$kv, Chip, m![Sl, Df]>,
            ) -> HbmTensor<bf16, Chip, m![Rw, Hq, Df]> {
                // The row's query, 128 elements per slice, all-gathered over the row's 64-slice ring into the TRF:
                // head h on lane h % 8, element [h / 8, d].
                let q_dm: DmTensor<bf16, Chip, C, m![Rw % 4, Hq % 8, Hq / 8, Df / 128], m![Df % 128]> =
                    q.to_dm(&mut device.tdma);
                let q_trf: TrfTensor<bf16, Chip, C, S, m![Hq % 8], m![Hq / 8, Df]> = device
                    .sub
                    .begin(q_dm.view())
                    .fetch::<m![Df % 128 / 16], m![Df % 16]>()
                    .switch::<m![Rw % 4, Po], m![Hq % 8, Hq / 8, Df / 128, Df % 128 / 16]>(SwitchConfig::Broadcast01 {
                        slice1: 64,
                        slice0: 1,
                        time0: 8,
                    })
                    .collect::<m![Hq % 8, Hq / 8, Df / 16], m![Df % 16]>()
                    .to_trf();

                let cap_dm: DmTensor<f32, Chip, C, S, m![Pi]> = cap.to_dm(&mut device.tdma);
                let cap_vrf: VrfTensor<f32, Chip, C, S, m![Pi / 4, Pi % 4 # 8]> = device
                    .sub
                    .begin(cap_dm.view())
                    .fetch::<m![Pi / 4], m![Pi % 4]>()
                    .collect::<m![Pi / 4], m![Pi % 4 # 8]>()
                    .to_vrf();

                // One 512-byte E4M3 slot per position.
                let k: DmTensor<$kv, Chip, C, S, m![Pi, Df]> = k_cache.gather::<m![Sl], m![Df]>().by_byte_offsets(kv_off).to_dm(&mut device.tdma);
                let v: DmTensor<$kv, Chip, C, S, m![Pi, Df]> = v_cache.gather::<m![Sl], m![Df]>().by_byte_offsets(kv_off).to_dm(&mut device.tdma);

                // 1. Clipped scores [position, head] in f32; each key packet serves the two head groups in turn.
                let s: DmTensor<f32, Chip, C, S, m![Pi, Hq / 8, Hq % 8]> = dsl_lut!($kv, device.main.begin(k.view()).fetch::<m![Pi, Df / 16], m![Df % 16]>())
                    .collect::<m![Pi, Df / 16], m![Df % 16]>()
                    .contract_outer::<m![Pi, Df / 32, Hq / 8], m![Df % 32], _, _, _>(&q_trf)
                    .contract_packet::<m![1]>()
                    .contract_time::<m![Pi, Hq / 8]>()
                    .contract_lane::<m![Pi, Hq / 8], m![Hq % 8]>(LaneMode::Interleaved)
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_clip(ClipBinaryOpF32::Min, &cap_vrf)
                    .vector_final()
                    .commit_trim::<m![Hq % 8]>()
                    .commit();

                // 2. Max per head over the slice's positions, then over the row's slices, broadcast back.
                let slice_max: DmTensor<f32, Chip, C, S, m![Hq / 8, Hq % 8]> = device
                    .main
                    .begin(s.view())
                    .fetch::<m![Hq / 8, Pi], m![Hq % 8]>()
                    .collect::<m![Hq / 8, Pi], m![Hq % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![Hq / 8, Pi, Hq % 8 / 4], m![Hq % 4]>()
                    .vector_intra_slice_reduce::<Pi, m![Hq / 8, Hq % 8 / 4], m![Hq % 4]>(IntraSliceReduceOpF32::Max)
                    .vector_widen_concat::<m![Hq / 8], m![Hq % 8]>()
                    .vector_final()
                    .commit_trim::<m![Hq % 8]>()
                    .commit();
                let max: DmTensor<f32, Chip, C, SX, m![Hq / 8, Hq % 8]> = device
                    .main
                    .begin(slice_max.view())
                    .fetch::<m![Hq / 8], m![Hq % 8]>()
                    .collect::<m![Hq / 8], m![Hq % 8]>()
                    .vector_init()
                    .vector_inter_slice_reduce::<SX, m![Hq / 8]>(InterSliceReduceOpF32::Max)
                    .vector_final()
                    .commit_trim::<m![Hq % 8]>()
                    .commit();
                let max: DmTensor<f32, Chip, C, S, m![Hq / 8, Hq % 8]> = unsafe { max.reshape() };
                let max_vrf: VrfTensor<f32, Chip, C, S, m![Hq / 8, Hq % 8]> = device
                    .sub
                    .begin(max.view())
                    .fetch::<m![Hq / 8], m![Hq % 8]>()
                    .collect::<m![Hq / 8], m![Hq % 8]>()
                    .to_vrf();

                // 3. Probabilities in bf16, transposed to [head % 8, head / 8, position] for the TRF.
                let p: DmTensor<bf16, Chip, C, S, m![Hq % 8, Hq / 8, Pi]> = device
                    .main
                    .begin(s.view())
                    .fetch::<m![Pi / 4, Hq / 8, Pi % 4], m![Hq % 8]>()
                    .collect::<m![Pi / 4, Hq / 8, Pi % 4], m![Hq % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![Pi / 4, Hq / 8, Pi % 4, Hq % 8 / 4], m![Hq % 4]>()
                    .vector_fp_binary(FpBinaryOp::SubF, &max_vrf)
                    .vector_fp_unary(FpUnaryOp::Exp)
                    .vector_widen_concat::<m![Pi / 4, Hq / 8, Pi % 4], m![Hq % 8]>()
                    .vector_final()
                    .cast::<bf16, m![Hq % 8 # 16]>()
                    .transpose::<m![Pi / 4, Hq / 8, Hq % 8], m![Pi % 4 # 16]>()
                    .commit_trim::<m![Pi % 4]>()
                    .commit();

                let slice_sum: DmTensor<f32, Chip, C, S, m![Hq / 8, Hq % 8]> = device
                    .main
                    .begin(s.view())
                    .fetch::<m![Hq / 8, Pi], m![Hq % 8]>()
                    .collect::<m![Hq / 8, Pi], m![Hq % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![Hq / 8, Pi, Hq % 8 / 4], m![Hq % 4]>()
                    .vector_fp_binary(FpBinaryOp::SubF, &max_vrf)
                    .vector_fp_unary(FpUnaryOp::Exp)
                    .vector_intra_slice_reduce::<Pi, m![Hq / 8, Hq % 8 / 4], m![Hq % 4]>(IntraSliceReduceOpF32::Add)
                    .vector_widen_concat::<m![Hq / 8], m![Hq % 8]>()
                    .vector_final()
                    .commit_trim::<m![Hq % 8]>()
                    .commit();
                let sum: DmTensor<f32, Chip, C, S1, m![Hq / 8, Hq % 8]> = device
                    .main
                    .begin(slice_sum.view())
                    .fetch::<m![Hq / 8], m![Hq % 8]>()
                    .collect::<m![Hq / 8], m![Hq % 8]>()
                    .vector_init()
                    .vector_inter_slice_reduce::<S1, m![Hq / 8]>(InterSliceReduceOpF32::Add)
                    .vector_final()
                    .commit_trim::<m![Hq % 8]>()
                    .commit();

                // 4. PV: P stationary (head % 8 on lanes), the values decoded in the fetch and streamed
                // [d-chunk, position], each packet serving both head groups; positions accumulate in time.
                let p_trf: TrfTensor<bf16, Chip, C, S, m![Hq % 8], $p_elem> = device
                    .sub
                    .begin(p.view())
                    .fetch::<$p_time, $p_flit>()
                    .collect::<$p_time, $p_flit>()
                    .to_trf();
                let o: DmTensor<f32, Chip, C, S1, m![Df / 32, Hq / 8, Hq % 8, Df % 32]> = dsl_lut!($kv, device.main.begin(v.view()).fetch::<m![Df / 32, Pi, Df / 16 % 2], m![Df % 16]>())
                    .collect::<m![Df / 32, Pi, Df / 16 % 2], m![Df % 16]>()
                    .contract_outer::<m![Df / 32, Pi, Hq / 8], m![Df % 32], _, _, _>(&p_trf)
                    .contract_packet::<m![Df % 32]>()
                    .contract_time::<m![Df / 32, Hq / 8]>()
                    .contract_lane::<m![Df / 32, Hq / 8, Hq % 8, Df % 32 / 8], m![Df % 8]>(LaneMode::Sequential)
                    .vector_init()
                    .vector_inter_slice_reduce::<S1, m![Df / 32, Hq / 8, Hq % 8, Df % 32 / 8]>(InterSliceReduceOpF32::Add)
                    .vector_final()
                    .commit_trim::<m![Df % 8]>()
                    .commit();

                // 5. Normalize, round, reorder to [head, d].
                let sum_vrf: VrfTensor<f32, Chip, C, S1, m![Hq / 8, Hq % 8]> = device
                    .sub
                    .begin(sum.view())
                    .fetch::<m![Hq / 8], m![Hq % 8]>()
                    .collect::<m![Hq / 8], m![Hq % 8]>()
                    .to_vrf();
                let out: DmTensor<bf16, Chip, C, S1, m![Hq, Df]> = device
                    .main
                    .begin(o.view())
                    .fetch::<m![Hq / 8, Hq % 8, Df / 8], m![Df % 8]>()
                    .collect::<m![Hq / 8, Hq % 8, Df / 8], m![Df % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![Hq / 8, Hq % 8, Df / 4], m![Df % 4]>()
                    .vector_fp_div(&sum_vrf)
                    .vector_widen_concat::<m![Hq / 8, Hq % 8, Df / 8], m![Df % 8]>()
                    .vector_final()
                    .cast::<bf16, m![Df % 8 # 16]>()
                    .commit_trim::<m![Df % 8]>()
                    .commit();
                out.to_hbm(&mut device.tdma)
            }

        }
    };
}

dsl_full_decode!(f512_fp8, full_decode_fp8_w512, f8e4m3, 8, m![Hq % 8], m![Hq / 8, Pi], m![Hq / 8, Pi]);
dsl_full_decode!(f1024_fp8, full_decode_fp8_w1024, f8e4m3, 16, m![Hq % 8, Hq / 8], m![Pi], m![Hq / 8, Pi]);
dsl_full_decode!(f2048_fp8, full_decode_fp8_w2048, f8e4m3, 32, m![Hq % 8, Hq / 8, Pi / 16], m![Pi % 16], m![Hq / 8, Pi]);
dsl_full_decode!(f3072_fp8, full_decode_fp8_w3072, f8e4m3, 48, m![Hq % 8, Hq / 8, Pi / 16], m![Pi % 16], m![Hq / 8, Pi]);
dsl_full_decode!(f4096_fp8, full_decode_fp8_w4096, f8e4m3, 64, m![Hq % 8, Hq / 8, Pi / 16], m![Pi % 16], m![Hq / 8, Pi]);

// Resident groups use these axes in their carried image ABI.
macro_rules! resident_geometry {
    ($module:ident, $pi:literal) => {
        pub mod $module {
            use furiosa_opt_std::prelude::*;
            full_decode_geometry!($pi);
        }
    };
}

resident_geometry!(f512_bf16, 8);
resident_geometry!(f1024_bf16, 16);
resident_geometry!(f1536_bf16, 24);
resident_geometry!(f2048_bf16, 32);
resident_geometry!(f2560_bf16, 40);
resident_geometry!(f3072_bf16, 48);
