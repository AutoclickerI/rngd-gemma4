//! Small geometries with the Tok512 numerical class: the Tok128 bucket runs these for decode-only steps of
//! at most 128 rows (`host::engine::step_bucket_for`), so a row gets the bits it gets in a Tok512 step.
use super::*;
#[device(chip = 1)]
        pub fn pre_class512_t128(
            device: &mut Device,
            x: &HbmTensor<bf16, Chip, m![Tok128, H]>,
            rope_s: &HbmTensor<i32, Chip, m![Tok128]>,
            rows_s: &HbmTensor<i32, Chip, m![Tok128, Hh]>,
            stage_s: &HbmTensor<i32, Chip, m![Tok128]>,
            consts: Int8Consts<'_>,
            next: Int8SlidingQkvWeights<'_>,
            qin_s: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
            k0: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
            v0: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
        ) {
            let mut q = HbmTensor::<bf16, Chip, m![Tok128, Qs]>::new();
            let mut k = HbmTensor::<bf16, Chip, m![Tok128, Ps]>::new();
            let mut v = HbmTensor::<bf16, Chip, m![Tok128, Ps]>::new();
            pf_qkv::sliding_qkv_stage_class512::<Tok128, 16, 64, 128, OnePass, m![1], m![Nq / 64 % 64], m![Nq / 16 % 4], m![Tok128 / 64], m![Tok128 % 64], 64, 32, 64, 2>(
                device,
                x,
                next.input_rms_weight_signed,
                consts.h64,
                consts.h60,
                consts.eye,
                next.wq,
                next.ws,
                next.q_rms_weight,
                next.k_rms_weight,
                rope_s,
                next.cos,
                next.sin,
                &mut q,
                &mut k,
                &mut v,
            );
            moves::scatter_q::<Tok128, m![Tok128, Hh], m![Hqs]>(device, q, rows_s, qin_s);
            moves::scatter_kv_e4m3::<Tok128, m![Tok128, 1 # 2], m![Ns, Ds], m![Ns, Ds / 16], m![Ns, Ds / 8]>(device, k, stage_s, k0);
            moves::scatter_kv_e4m3::<Tok128, m![Tok128, 1 # 2], m![Ns, Ds], m![Ns, Ds / 16], m![Ns, Ds / 8]>(device, v, stage_s, v0);
        }

#[device(chip = 1)]
        pub fn mid_sf_class512_t128(
            device: &mut Device,
            out_s: &HbmTensor<bf16, Chip, m![Rq, Hqs]>,
            rows_s: &HbmTensor<i32, Chip, m![Tok128, Hh]>,
            stage_s: &HbmTensor<i32, Chip, m![Tok128]>,
            publish_s: &HbmTensor<i32, Chip, m![Tok128]>,
            rope_f: &HbmTensor<i32, Chip, m![Tok128]>,
            rows_f: &HbmTensor<i32, Chip, m![Tok128, Hh]>,
            publish_f: &HbmTensor<i32, Chip, m![Tok128]>,
            x: &mut HbmTensor<bf16, Chip, m![Tok128, H]>,
            consts: Int8Consts<'_>,
            o: crate::ops_tokenwise_int8::SlidingOStaticF8<'_>,
            ffn: crate::ops_tokenwise_int8::StaticF8FfnWeights<'_>,
            layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
            next: super::full_resident::FullQkvStaticF8<'_>,
            k_l: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
            v_l: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
            qin_f: &mut super::full_resident::Space,
            k_n: &mut super::full_resident::Pool,
            v_n: &mut super::full_resident::Pool,
        ) {
            let xa = pf_o::fp8c_drain::sliding_o_stage_rows_static_dm1_f8::<Tok128, 64, 128, m![1], m![H / 64 # 64], m![H / 16 % 4], 64>(
                device, &o.sigma, out_s, rows_s, x, consts.eye, o.wo, o.ws, o.post_attn_rms_weight,
            );
            crate::device::tokenwise_int8::segments_full::publish_sliding_kv8::<Tok128, m![Tok128, 1 # 2], m![Ns, Ds]>(device, stage_s, publish_s, k_l, v_l);
            let _ = ffn_fast::fp8c::ffn_stage_static_dm1_f8::<Tok128, 64, 128, m![1], m![Lu / 1024 # 32, Lu / 256 % 2], m![Lu / 64 % 4], m![1], m![H / 64 # 64], m![H / 16 % 4], m![Tok128 / 64], m![Tok128 % 64], 64>(
                device, ffn.sigma, ffn.sigma_down, &xa, x, consts.eye, ffn.pre_ff_rms_weight, ffn.wug, ffn.wug_scale, ffn.wd,
                ffn.wd_scale, ffn.post_ff_rms_weight, layer_scalar,
            );
            full_qkv_static_rows_c512::<Tok128, 64, 128, OnePass, m![1], m![Qf / 64 % 64], m![Qf / 16 % 4], m![Tok128 / 64], m![Tok128 % 64], 64,
                m![Tok128 / 8], m![Tok128 / 4 % 2], m![Tok128 % 4], m![Tok128 / 128], m![Tok128 % 8]>(
                device, x, &next, rope_f, rows_f, publish_f, qin_f, k_n, v_n,
            );
        }

/// `full_resident::full_qkv_static_rows` (the static FP8 Full Q/K GEMM, then the fused-row head phase) for one row pass,
/// with Tok512's Full RoPE form (`heads_full1::heads_rows_compact_one_pass`) instead of the bucket's `RopePolicy`: the
/// Tok128 class512 SF body's rows get the Full q/K bits Tok512 gives them, while the rotated-INT8 Tok128 stages keep
/// `rope_original`.
#[allow(clippy::too_many_arguments)]
fn full_qkv_static_rows_c512<
    T: AxisName,
    const HALF: usize,
    const PS2: usize,
    PP: crate::device::tokenwise_int8::passes::Passes,
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
    next: &super::full_resident::FullQkvStaticF8<'_>,
    rope_offsets: &HbmTensor<i32, Chip, m![T]>,
    rows_f: &HbmTensor<i32, Chip, m![T, Hh]>,
    publish_f: &HbmTensor<i32, Chip, m![T]>,
    qin_f: &mut super::full_resident::Space,
    k_n: &mut HbmTensor<f8e4m3, Chip, m![Rf, Df]>,
    v_n: &mut HbmTensor<f8e4m3, Chip, m![Rf, Df]>,
) {
    let mut q_raw = HbmTensor::<bf16, Chip, m![T, Qf]>::new();
    let mut k_raw = HbmTensor::<bf16, Chip, m![T, Pf]>::new();
    crate::device::tokenwise_int8::core16_full::full_qkv_gemm::<T, HALF, PS2, PP, RING, XS, YS, TT, TR, TS, KR, KTT, KTR, KT8, KT1>(
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
    crate::device::tokenwise_int8::heads_full1::heads_rows_compact_one_pass::<T, HALF, PS2>(
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

// Plain bodies for the decode-only resident forms; existing entry points retain their source.
#[cfg(feature = "compile-carried")]
pub(crate) fn mid_ss_q4_class512_t128_body(
    device: &mut Device,
    out_s: &HbmTensor<bf16, Chip, m![Rq, Hqs]>,
    rows_s: &HbmTensor<i32, Chip, m![Tok128, Hh]>,
    stage_s: &HbmTensor<i32, Chip, m![Tok128]>,
    publish_s: &HbmTensor<i32, Chip, m![Tok128]>,
    rope_s: &HbmTensor<i32, Chip, m![Tok128]>,
    x: &mut HbmTensor<bf16, Chip, m![Tok128, H]>,
    consts: Int8Consts<'_>,
    o: crate::ops_tokenwise_int8::SlidingOStaticF8<'_>,
    ffn: super::q4serve::Q4FfnWeights<'_>,
    layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
    next: crate::ops_tokenwise_int8::StaticF8SlidingQkvWeights<'_>,
    k_l: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
    v_l: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
    qin_s: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
    k_n: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
    v_n: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
) {
    let xa0 = pf_o::fp8c_drain::sliding_o_stage_rows_static_dm1_f8::<Tok128, 64, 128, m![1], m![H / 64 # 64], m![H / 16 % 4], 64>(
        device, o.sigma, out_s, rows_s, x, consts.eye, o.wo, o.ws, o.post_attn_rms_weight,
    );
    moves::publish_kv8::<Tok128, m![Tok128 % 128, 1 # 2], m![Tok128 / 128, Ns, Ds]>(device, k_l, stage_s, publish_s);
    moves::publish_kv8::<Tok128, m![Tok128 % 128, 1 # 2], m![Tok128 / 128, Ns, Ds]>(device, v_l, stage_s, publish_s);
    let xf0 = ffn_fast::ug4::serve::stage_dm1_sl2::<Tok128, 64, 128, m![1], m![Lu / 1024 # 32, Lu / 256 % 2], m![Lu / 64 % 4], m![1], m![H / 64 # 64], m![H / 16 % 4], m![Tok128 / 64], m![Tok128 % 64], 64, 64, 2>(
        device, ffn.sigma, ffn.sigma_down, &xa0, x, consts.eye, ffn.pre_ff_rms_weight, ffn.wug, ffn.wug_scale, ffn.wd, ffn.wd_scale,
        ffn.post_ff_rms_weight, layer_scalar, ffn.pre_signed, ffn.codes, ffn.scales, consts.h64, consts.h60,
    );
    pf_qkv::fp8_xchg_drain::stage1::<Tok128, 64, 128, OnePass, m![1], m![Nq / 64 % 64], m![Nq / 16 % 4], m![Tok128 / 64], m![Tok128 % 64], 64>(
        device, next.sigma, &xf0, next.input_rms_weight, consts.eye, next.wq, next.ws, next.q_rms_weight, next.k_rms_weight, rope_s,
        next.cos, next.sin, rows_s, stage_s, qin_s, k_n, v_n,
    );
}

#[cfg(feature = "compile-carried")]
pub(crate) fn mid_fs_q4_class512_t128_body(
    device: &mut Device,
    out_f: &HbmTensor<bf16, Chip, m![Rq, Hqf]>,
    rows_f: &HbmTensor<i32, Chip, m![Tok128, Hh]>,
    rope_s: &HbmTensor<i32, Chip, m![Tok128]>,
    rows_s: &HbmTensor<i32, Chip, m![Tok128, Hh]>,
    stage_s: &HbmTensor<i32, Chip, m![Tok128]>,
    x: &mut HbmTensor<bf16, Chip, m![Tok128, H]>,
    consts: Int8Consts<'_>,
    o: FullOInt8<'_>,
    ffn: super::q4serve::Q4FfnWeights<'_>,
    layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
    next: Int8SlidingQkvWeights<'_>,
    qin_s: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
    k_n: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
    v_n: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
) {
    let xa = crate::device::tokenwise_int8::core8_full::full_o_stage_i8_rows_dm1_half::<Tok128, 64, 128, OnePass, m![1], m![H / 64 # 64], m![H / 16 % 4], m![Tok128 / 64], m![Tok128 % 64], 64, 64>(
        device, out_f, rows_f, x, consts.eye, consts.h64, consts.h8, o.o_signs, o.wo, o.ws, o.post_attn_rms_weight,
    );
    let xf = ffn_fast::ug4::serve::stage_dm1_sl2::<Tok128, 64, 128, m![1], m![Lu / 1024 # 32, Lu / 256 % 2], m![Lu / 64 % 4], m![1], m![H / 64 # 64], m![H / 16 % 4], m![Tok128 / 64], m![Tok128 % 64], 64, 64, 2>(
        device, ffn.sigma, ffn.sigma_down, &xa, x, consts.eye, ffn.pre_ff_rms_weight, ffn.wug, ffn.wug_scale, ffn.wd, ffn.wd_scale,
        ffn.post_ff_rms_weight, layer_scalar, ffn.pre_signed, ffn.codes, ffn.scales, consts.h64, consts.h60,
    );
    pf_qkv::int8s::sliding_qkv_stage_rows_dm1_i8s::<Tok128, 64, 128, OnePass, m![1], m![Nq / 64 % 64], m![Nq / 16 % 4], m![Tok128 / 64], m![Tok128 % 64], 64>(
        device, &xf, next.input_rms_weight_signed, consts.eye, next.wq, next.ws, next.q_rms_weight, next.k_rms_weight, rope_s,
        next.cos, next.sin, rows_s, stage_s, qin_s, k_n, v_n,
    );
}
