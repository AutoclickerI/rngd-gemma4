//! Resident Sliding → Full programs fuse the decoder segment with Full decode attention.
//!
//! The segment writes the next layer's queries and K/V. Each eight-row group gathers K/V and caps
//! afterward and uses the standalone decode passes on the same query and output rows.
//!
//! S8N uses window capacities 512 / 1,024 / 1,536 / 2,560 and S8 uses 512 / 1,024 / 2,048 / 3,072.
//! Both have 2 / 3 / 1 / 2 groups in that order. The engine places each decode block in the
//! smallest free group that covers it. An empty group runs all-masked tables on the sink row.
//!
//! Arguments: `mid_sf_int8_fp8kv_tN`'s, then `out_f`, then the shape's groups (per group `kv_off`, `cap`, `rows`).
use furiosa_opt_std::prelude::*;

use super::{Int8Consts};
use crate::ops_tokenwise_int8::{SlidingOStaticF8, StaticF8FfnWeights};
use crate::Chip;
use crate::axes::*;
use crate::device::proj_fast::o::fp8c::pair as o_pair;
use crate::device::tokenwise_int8::core16_full;
use crate::device::tokenwise_int8::heads_full1::OneRowPerSlice;
use crate::device::tokenwise_int8::heads_full1::{FullHeadPhaseRows, FullHeadsRowsPasses};
use crate::device::tokenwise_int8::passes::{Passes, TwoPasses};
#[cfg(feature = "compile-carried")]
use crate::device::tokenwise_int8::passes::ThreePasses;
use crate::device::tokenwise_int8::ffn::Lu;
use crate::device::tokenwise_int8::segments_full::publish_sliding_kv8;
use crate::device::tokenwise_int8::ffn_fast;

/// The Full pool (`[Rf, Df]` bf16).
pub type Pool = HbmTensor<f8e4m3, Chip, m![Rf, Df]>;
/// The Full row space (`qin_f`, `out_f`).
pub type Space = HbmTensor<bf16, Chip, m![Rq, Hqf]>;

/// Static FP8 Full Q/K weights for the RSF resident transition. The source
/// weights and row scales are already E4M3/BF16 in the supplied Full layer;
/// this carrier keeps that layout and routes the projection through
/// `core16_full`.
#[derive(DeviceSend)]
pub struct FullQkvStaticF8<'a> {
    pub input_rms_weight: &'a HbmTensor<bf16, Chip, m![Rc, H]>,
    pub wq: &'a HbmTensor<f8e4m3, Chip, m![Qf, H]>,
    pub wsq: &'a HbmTensor<bf16, Chip, m![Qf]>,
    pub wk: &'a HbmTensor<f8e4m3, Chip, m![Pf, H]>,
    pub wsk: &'a HbmTensor<bf16, Chip, m![Pf]>,
    pub q_rms_weight: &'a HbmTensor<bf16, Chip, m![Df]>,
    pub k_rms_weight: &'a HbmTensor<bf16, Chip, m![Df]>,
    pub cos: &'a HbmTensor<bf16, Chip, m![E, Df]>,
    pub sin: &'a HbmTensor<bf16, Chip, m![E, Df]>,
}

/// Full Q/K projection plus the existing fused head-row publication. The
/// static FP8 GEMM preserves the served RSF head and pool ABI.
#[allow(clippy::too_many_arguments)]
pub(super) fn full_qkv_static_rows<
    T: AxisName,
    const HALF: usize,
    const PS2: usize,
    PP: Passes + FullHeadsRowsPasses,
    RING: M,
    XS: M,
    YS: M,
    TT: M,
    TR: M,
    const TS: usize,
    KR: M,
    KTT: M,
    KTR: M,
    KT8: M,
    KT1: M,
>(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![T, H]>,
    next: &FullQkvStaticF8<'_>,
    rope_offsets: &HbmTensor<i32, Chip, m![T]>,
    rows_f: &HbmTensor<i32, Chip, m![T, Hh]>,
    publish_f: &HbmTensor<i32, Chip, m![T]>,
    qin_f: &mut Space,
    k_n: &mut HbmTensor<f8e4m3, Chip, m![Rf, Df]>,
    v_n: &mut HbmTensor<f8e4m3, Chip, m![Rf, Df]>,
) {
    let mut q_raw = HbmTensor::<bf16, Chip, m![T, Qf]>::new();
    let mut k_raw = HbmTensor::<bf16, Chip, m![T, Pf]>::new();
    core16_full::full_qkv_gemm::<T, HALF, PS2, PP, RING, XS, YS, TT, TR, TS, KR, KTT, KTR, KT8, KT1>(
        device,
        x,
        next.input_rms_weight,
        next.wq,
        next.wk,
        next.wsq,
        next.wsk,
        &mut q_raw,
        &mut k_raw,
    );
    <OneRowPerSlice as FullHeadPhaseRows>::run_rows::<T, HALF, PS2, PP>(
        device,
        &q_raw,
        &k_raw,
        next.q_rms_weight,
        next.k_rms_weight,
        rope_offsets,
        next.cos,
        next.sin,
        rows_f,
        publish_f,
        qin_f,
        k_n,
        v_n,
    );
}

/// A resident group shares its window axes with the carried decode layout.
macro_rules! full_group {
    ($m:ident, $src:ident, $p_time:ty, $p_flit:ty, $p_elem:ty) => {
        pub mod $m {
            use furiosa_opt_std::prelude::*;

            use super::{Pool, Space};
            use crate::Chip;
            use crate::axes::{Df, Hh, Hqf, Rf, Rq};
            pub use crate::ops_dsl_attention::decode_full::$src::{C, Hq, Pi, Po, Rw, S, S1, SX};

            /// Byte offsets (`pool row · 1024`) of window position `64 Pi + Po` in the gather's list order.
            pub type Off = HbmTensor<i32, Chip, m![Rw, Pi, Po % 32, Po / 32]>;
            /// `+MAX` where the row attends, `-1e30` elsewhere.
            pub type Cap = HbmTensor<f32, Chip, m![Rw, Po, Pi]>;
            /// The group's 16 row-space rows (kernel row `r`, half `h` at `[r][h]`), byte offsets.
            pub type Rows = HbmTensor<i32, Chip, m![Rw, Hh]>;
            type Kv = DmTensor<f8e4m3, Chip, C, S, m![Pi, Df]>;

            /// One group's tables and rows.
            #[derive(DeviceSend)]
            pub struct Grp<'a> {
                pub kv_off: &'a Off,
                pub cap: &'a Cap,
                pub rows: &'a Rows,
            }

            /// Gather Q and KV, run Full decode and scatter to the same row-space offsets.
            pub(crate) fn attend(device: &mut Device, g: &Grp<'_>, qin_f: &Space, k_cache: &Pool, v_cache: &Pool, out_f: &mut Space) {
                let k: Kv = k_cache.gather::<m![Rf], m![Df]>().by_byte_offsets(g.kv_off).to_dm(&mut device.tdma);
                let v: Kv = v_cache.gather::<m![Rf], m![Df]>().by_byte_offsets(g.kv_off).to_dm(&mut device.tdma);
                let cap_dm: DmTensor<f32, Chip, C, S, m![Pi]> = g.cap.to_dm(&mut device.tdma);

                // Gather each row half directly onto the query consumer's physical
                // cluster/slice/lane. Hh = Hq / 8 and Hqf = (Hq % 8, Df).
                let halves: DmTensor<bf16, Chip, C,
                    m![Rw % 4, Hqf / Df, Hh, Hqf % Df / 128], m![Hqf % 128]> =
                    qin_f.gather::<m![Rq], m![Hqf]>().by_byte_offsets(g.rows).to_dm(&mut device.tdma);
                let q_dm: DmTensor<bf16, Chip, C, m![Rw % 4, Hq % 8, Hq / 8, Df / 128], m![Df % 128]> = unsafe { halves.reshape() };
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
                let cap_vrf: VrfTensor<f32, Chip, C, S, m![Pi / 4, Pi % 4 # 8]> = device
                    .sub
                    .begin(cap_dm.view())
                    .fetch::<m![Pi / 4], m![Pi % 4]>()
                    .collect::<m![Pi / 4], m![Pi % 4 # 8]>()
                    .to_vrf();

                let s: DmTensor<f32, Chip, C, S, m![Pi, Hq / 8, Hq % 8]> = device
                    .main
                    .begin(k.view())
                    .fetch::<m![Pi, Df / 16], m![Df % 16]>()
                    .fetch_table_lookup::<bf16>()
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

                let p_trf: TrfTensor<bf16, Chip, C, S, m![Hq % 8], $p_elem> = device
                    .sub
                    .begin(p.view())
                    .fetch::<$p_time, $p_flit>()
                    .collect::<$p_time, $p_flit>()
                    .to_trf();
                let o: DmTensor<f32, Chip, C, S1, m![Df / 32, Hq / 8, Hq % 8, Df % 32]> = device
                    .main
                    .begin(v.view())
                    .fetch::<m![Df / 32, Pi, Df / 16 % 2], m![Df % 16]>()
                    .fetch_table_lookup::<bf16>()
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

                // The reduction's final owner already holds (Hq, Df) in order.
                let halves: DmTensorView<'_, bf16, Chip, C, S1, m![Hh, Hqf]> = unsafe { out.view().reshape() };
                out_f.view_mut().scatter::<m![Rq], m![Hqf]>().by_byte_offsets(g.rows.view()).from_dm_view(&mut device.tdma, halves);
            }
        }
    };
}

full_group!(w512, f512_bf16, m![Hq % 8], m![Hq / 8, Pi], m![Hq / 8, Pi]);
full_group!(w1024, f1024_bf16, m![Hq % 8, Hq / 8], m![Pi], m![Hq / 8, Pi]);
full_group!(w2048, f2048_bf16, m![Hq % 8, Hq / 8, Pi / 16], m![Pi % 16], m![Hq / 8, Pi]);
full_group!(w3072, f3072_bf16, m![Hq % 8, Hq / 8, Pi / 16], m![Pi % 16], m![Hq / 8, Pi]);

/// 2/3/1/2, the smallest window first.
#[derive(DeviceSend)]
pub struct S8<'a> {
    pub a0: w512::Grp<'a>,
    pub a1: w512::Grp<'a>,
    pub b0: w1024::Grp<'a>,
    pub b1: w1024::Grp<'a>,
    pub b2: w1024::Grp<'a>,
    pub c0: w2048::Grp<'a>,
    pub d0: w3072::Grp<'a>,
    pub d1: w3072::Grp<'a>,
}

/// Tok1024 RSF body with the paired static FP8 Sliding-O path.
macro_rules! body_v9_t1024 {
    ($name:ident) => {
        #[allow(clippy::too_many_arguments)]
        pub(super) fn $name(
            device: &mut Device,
            out_s: &HbmTensor<bf16, Chip, m![Rq, Hqs]>,
            rows_s: &HbmTensor<i32, Chip, m![Tok1024, Hh]>,
            stage_s: &HbmTensor<i32, Chip, m![Tok1024]>,
            publish_s: &HbmTensor<i32, Chip, m![Tok1024]>,
            rope_f: &HbmTensor<i32, Chip, m![Tok1024]>,
            rows_f: &HbmTensor<i32, Chip, m![Tok1024, Hh]>,
            publish_f: &HbmTensor<i32, Chip, m![Tok1024]>,
            x: &mut HbmTensor<bf16, Chip, m![Tok1024, H]>,
            consts: &Int8Consts<'_>,
            o: &SlidingOStaticF8<'_>,
            ffn: &StaticF8FfnWeights<'_>,
            layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
            next: &FullQkvStaticF8<'_>,
            k_l: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
            v_l: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
            qin_f: &mut Space,
            k_n: &mut Pool,
            v_n: &mut Pool,
        ) {
            let (xa0, xa1) = o_pair::stage2::<Tok1024, 256, 512, m![1], m![H / 64 # 64], m![H / 16 % 4], 64>(
                device, &o.sigma, out_s, rows_s, x, consts.eye, o.wo, o.ws, o.post_attn_rms_weight,
            );
            publish_sliding_kv8::<Tok1024, m![Tok1024 % 128, 1 # 2], m![Tok1024 / 128, Ns, Ds]>(device, stage_s, publish_s, k_l, v_l);
            let (xf0, xf1) = ffn_fast::fp8c::pair::stage2_up_dm_pp_s256::<Tok1024, 256, 512, m![1], m![Lu / 1024 # 32, Lu / 256 % 2], m![Lu / 64 % 4], m![1], m![H / 64 # 64], m![H / 16 % 4], m![Tok1024 / 64], m![Tok1024 % 64], 64>(
                device, ffn.sigma, ffn.sigma_down, (&xa0, &xa1), x, consts.eye, ffn.pre_ff_rms_weight, ffn.wug, ffn.wug_scale, ffn.wd,
                ffn.wd_scale, ffn.post_ff_rms_weight, layer_scalar,
            );
            let _ = (xf0, xf1);
            full_qkv_static_rows::<Tok1024, 256, 512, TwoPasses, m![1], m![Qf / 64 % 64], m![Qf / 16 % 4], m![Tok1024 / 64], m![Tok1024 % 64], 64,
                m![Tok1024 / 64], m![Tok1024 / 4 % 16], m![Tok1024 % 4], m![Tok1024 / 8 % 8], m![Tok1024 % 8]>(
                device, x, next, rope_f, rows_f, publish_f, qin_f, k_n, v_n,
            );
        }
    };
}

body_v9_t1024!(body_t1024);
body_v9_t1024!(body_t1024_sub);

/// A program: `$body`, then the groups of `$shape` in its field order.
macro_rules! rsf {
    ($o:ty, $ffn:ty, $next:ty, $name:ident, $tok:ident, $body:ident, $shape:ident { $($field:ident: $w:ident),+ }) => {
        /// The Full-side resident program (module doc).
        #[device(chip = 1)]
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            device: &mut Device,
            out_s: &HbmTensor<bf16, Chip, m![Rq, Hqs]>,
            rows_s: &HbmTensor<i32, Chip, m![$tok, Hh]>,
            stage_s: &HbmTensor<i32, Chip, m![$tok]>,
            publish_s: &HbmTensor<i32, Chip, m![$tok]>,
            rope_f: &HbmTensor<i32, Chip, m![$tok]>,
            rows_f: &HbmTensor<i32, Chip, m![$tok, Hh]>,
            publish_f: &HbmTensor<i32, Chip, m![$tok]>,
            x: &mut HbmTensor<bf16, Chip, m![$tok, H]>,
            consts: Int8Consts<'_>,
            o: $o,
            ffn: $ffn,
            layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
            next: $next,
            k_l: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
            v_l: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
            qin_f: &mut Space,
            k_n: &mut Pool,
            v_n: &mut Pool,
            out_f: &mut Space,
            groups: $shape<'_>,
        ) {
            $body(device, out_s, rows_s, stage_s, publish_s, rope_f, rows_f, publish_f, x, &consts, &o, &ffn, layer_scalar, &next, k_l, v_l, qin_f, k_n, v_n);
            $($w::attend(device, &groups.$field, &*qin_f, &*k_n, &*v_n, out_f);)+
        }
    };
}

rsf!(SlidingOStaticF8<'_>, StaticF8FfnWeights<'_>, FullQkvStaticF8<'_>, rsf_int8_fp8kv_t1024_s8, Tok1024, body_t1024_sub, S8n { a0: w512, a1: w512, b0: w1024, b1: w1024, b2: w1024, c0: w1536, d0: w2560, d1: w2560 });

full_group!(w1536, f1536_bf16, m![Hq % 8, Pi / 8], m![Hq / 8, Pi % 8], m![Pi / 8, Hq / 8, Pi % 8]);
full_group!(w2560, f2560_bf16, m![Hq % 8, Pi / 8], m![Hq / 8, Pi % 8], m![Pi / 8, Hq / 8, Pi % 8]);

#[derive(DeviceSend)]
pub struct S8n<'a> {
    pub a0: w512::Grp<'a>,
    pub a1: w512::Grp<'a>,
    pub b0: w1024::Grp<'a>,
    pub b1: w1024::Grp<'a>,
    pub b2: w1024::Grp<'a>,
    pub c0: w1536::Grp<'a>,
    pub d0: w2560::Grp<'a>,
    pub d1: w2560::Grp<'a>,
}
#[cfg(feature = "compile-carried")]
macro_rules! body_v9_t1536 {
    ($name:ident) => {
        #[allow(clippy::too_many_arguments)]
        pub(super) fn $name(
            device: &mut Device,
            out_s: &HbmTensor<bf16, Chip, m![Rq, Hqs]>,
            rows_s: &HbmTensor<i32, Chip, m![Tok1536, Hh]>,
            stage_s: &HbmTensor<i32, Chip, m![Tok1536]>,
            publish_s: &HbmTensor<i32, Chip, m![Tok1536]>,
            rope_f: &HbmTensor<i32, Chip, m![Tok1536]>,
            rows_f: &HbmTensor<i32, Chip, m![Tok1536, Hh]>,
            publish_f: &HbmTensor<i32, Chip, m![Tok1536]>,
            x: &mut HbmTensor<bf16, Chip, m![Tok1536, H]>,
            consts: &Int8Consts<'_>,
            o: &SlidingOStaticF8<'_>,
            ffn: &StaticF8FfnWeights<'_>,
            layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
            next: &FullQkvStaticF8<'_>,
            k_l: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
            v_l: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
            qin_f: &mut Space,
            k_n: &mut Pool,
            v_n: &mut Pool,
        ) {
            let (xa0, xa1, xa2) = o_pair::stage3::<Tok1536, 256, 512, m![1], m![H / 64 # 64], m![H / 16 % 4], 64>(
                device, &o.sigma, out_s, rows_s, x, consts.eye, o.wo, o.ws, o.post_attn_rms_weight,
            );
            publish_sliding_kv8::<Tok1536, m![Tok1536 % 128, 1 # 2], m![Tok1536 / 128, Ns, Ds]>(device, stage_s, publish_s, k_l, v_l);
            let (xf0, xf1, xf2) = ffn_fast::fp8c::pair::stage3_up_dm_pp_weight_once::<Tok1536, 256, 512, m![1], m![Lu / 1024 # 32, Lu / 256 % 2], m![Lu / 64 % 4], m![1], m![H / 64 # 64], m![H / 16 % 4], m![Tok1536 / 64], m![Tok1536 % 64], 64>(
                device, ffn.sigma, ffn.sigma_down, (&xa0, &xa1, &xa2), x, consts.eye, ffn.pre_ff_rms_weight, ffn.wug, ffn.wug_scale, ffn.wd,
                ffn.wd_scale, ffn.post_ff_rms_weight, layer_scalar,
            );
            let _ = (xf0, xf1, xf2);
            full_qkv_static_rows::<Tok1536, 256, 512, ThreePasses, m![1], m![Qf / 64 % 64], m![Qf / 16 % 4], m![Tok1536 / 64], m![Tok1536 % 64], 64,
                m![Tok1536 / 96], m![Tok1536 / 4 % 24], m![Tok1536 % 4], m![Tok1536 / 8 % 12], m![Tok1536 % 8]>(
                device, x, next, rope_f, rows_f, publish_f, qin_f, k_n, v_n,
            );
        }
    };
}
#[cfg(feature = "compile-carried")]
body_v9_t1536!(body_t1536);
#[cfg(feature = "compile-carried")]
rsf!(SlidingOStaticF8<'_>, StaticF8FfnWeights<'_>, FullQkvStaticF8<'_>, rsf_int8_fp8kv_t1536_s8, Tok1536, body_t1536, S8 { a0: w512, a1: w512, b0: w1024, b1: w1024, b2: w1024, c0: w2048, d0: w3072, d1: w3072 });
