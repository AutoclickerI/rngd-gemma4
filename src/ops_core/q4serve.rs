//! Q4 Gate/Up with odd-grid activations clipped at 0.8, packed weight rows and
//! static-FP8 Down. RSS and RFS layers use these bodies in standalone or resident
//! segments according to the step geometry.
use furiosa_opt_std::prelude::*;

use super::*;
use crate::device::tokenwise_int8::ffn_fast::fp8c::Lp;
use crate::device::tokenwise_int8::ffn_fast::ug4::{Hb, Xc2};
use crate::device::tokenwise_int8::rotate_ff::{Kd, Kq4};

/// Bind the static-FP8 FFN slots first, followed by the three Q4 Gate/Up tensors.
/// The FP8 Gate/Up slots preserve the carried image ABI.
#[derive(DeviceSend)]
pub struct Q4FfnWeights<'a> {
    pub sigma_down: &'a HbmTensor<f32, Chip, m![1 # 8]>,
    pub sigma: &'a HbmTensor<f32, Chip, m![1 # 8]>,
    pub pre_ff_rms_weight: &'a HbmTensor<bf16, Chip, m![Rc, H]>,
    pub wug: &'a HbmTensor<f8e4m3, Chip, m![Lp, H]>,
    pub wug_scale: &'a HbmTensor<f32, Chip, m![Lu]>,
    pub wd: &'a HbmTensor<f8e4m3, Chip, m![Kq4, H, Kd]>,
    pub wd_scale: &'a HbmTensor<f32, Chip, m![H]>,
    pub post_ff_rms_weight: &'a HbmTensor<bf16, Chip, m![Rc, H]>,
    /// `Rc` copies of the pre-FFN gamma / smoothing, times the rotation's signs (q4cal `fold_gamma` + `signed`).
    pub pre_signed: &'a HbmTensor<bf16, Chip, m![Rc, H]>,
    /// The packed odd 4-bit weight nibbles (two-flit layout, `twoflit::pack`).
    pub codes: &'a HbmTensor<i8, Chip, m![Lu, Hb]>,
    /// Row 0: the Up/Gate rows' half steps times the global scale; row 1: the column term `C'`.
    pub scales: &'a HbmTensor<f32, Chip, m![Xc2, Lu]>,
}

macro_rules! mid_fs_t1024 {
    ($name:ident, $body:ident) => {
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn $name(
            device: &mut Device,
            out_f: &HbmTensor<bf16, Chip, m![Rq, Hqf]>,
            rows_f: &HbmTensor<i32, Chip, m![Tok1024, Hh]>,
            rope_s: &HbmTensor<i32, Chip, m![Tok1024]>,
            rows_s: &HbmTensor<i32, Chip, m![Tok1024, Hh]>,
            stage_s: &HbmTensor<i32, Chip, m![Tok1024]>,
            x: &mut HbmTensor<bf16, Chip, m![Tok1024, H]>,
            consts: Int8Consts<'_>,
            o: FullOInt8<'_>,
            ffn: Q4FfnWeights<'_>,
            layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
            next: Int8SlidingQkvWeights<'_>,
            qin_s: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
            k_n: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
            v_n: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
        ) {
            crate::device::tokenwise_int8::segments_full_i8::$body::<ffn_fast::GlOf<Tok1024>, Tok1024, 256, 512, crate::device::tokenwise_int8::passes::TwoPasses>(
                device, out_f, rows_f, rope_s, rows_s, stage_s, x, &consts, &o, &ffn, layer_scalar, &next, qin_s, k_n, v_n,
            );
        }
    };
}

mid_fs_t1024!(mid_fs_q4_t1024_body, mid_fs_q4_fp8down_kv8_v9_2_xchg);

// Tok1024 RSS shares the Q4 FFN and QKV stages across both O schedules.
macro_rules! mid_ss_t1024 {
    ($name:ident, $projection:ident) => {
        pub(crate) fn $name<F: ffn_fast::DownForm>(
            device: &mut Device,
            out_s: &HbmTensor<bf16, Chip, m![Rq, Hqs]>,
            rows_s: &HbmTensor<i32, Chip, m![Tok1024, Hh]>,
            stage_s: &HbmTensor<i32, Chip, m![Tok1024]>,
            publish_s: &HbmTensor<i32, Chip, m![Tok1024]>,
            rope_s: &HbmTensor<i32, Chip, m![Tok1024]>,
            x: &mut HbmTensor<bf16, Chip, m![Tok1024, H]>,
            consts: Int8Consts<'_>,
            o: crate::ops_tokenwise_int8::SlidingOStaticF8<'_>,
            ffn: Q4FfnWeights<'_>,
            layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
            next: crate::ops_tokenwise_int8::StaticF8SlidingQkvWeights<'_>,
            k_l: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
            v_l: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
            qin_s: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
            k_n: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
            v_n: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
        ) {
            let (xa0, xa1) = pf_o::fp8c::pair::$projection::<Tok1024, 256, 512, m![1], m![H / 64 # 64], m![H / 16 % 4], 64>(
                device, o.sigma, out_s, rows_s, x, o.wo, o.ws, o.post_attn_rms_weight,
            );
            moves::publish_kv8_cl::<Tok1024, m![Tok1024 / 512], m![Tok1024 % 128, 1 # 2], m![Tok1024 / 128 % 4, Ns, Ds]>(device, k_l, stage_s, publish_s);
            moves::publish_kv8_cl::<Tok1024, m![Tok1024 / 512], m![Tok1024 % 128, 1 # 2], m![Tok1024 / 128 % 4, Ns, Ds]>(device, v_l, stage_s, publish_s);
            let (xf0, xf1) = ffn_fast::ug4::serve::stage_dm2::<Tok1024, 256, 512, m![1], m![Lu / 1024 # 32, Lu / 256 % 2], m![Lu / 64 % 4], m![1], m![H / 64 # 64], m![H / 16 % 4], m![Tok1024 / 64], m![Tok1024 % 64], 64, 64, 16>(
                device, ffn.sigma, ffn.sigma_down,
                (&xa0, &xa1),
                x,
                consts.eye,
                ffn.pre_ff_rms_weight,
                ffn.wug,
                ffn.wug_scale,
                ffn.wd,
                ffn.wd_scale,
                ffn.post_ff_rms_weight,
                layer_scalar,
                ffn.pre_signed,
                ffn.codes,
                ffn.scales,
                consts.h64,
                consts.h60,
            );
            pf_qkv::fp8_xchg::stage::<Tok1024, 256, 512, TwoPasses, m![1], m![Nq / 64 % 64], m![Nq / 16 % 4], m![Tok1024 / 64], m![Tok1024 % 64], 64>(
                device, next.sigma,
                (&xf0, &xf1),
                next.input_rms_weight,
                consts.eye,
                next.wq,
                next.ws,
                next.q_rms_weight,
                next.k_rms_weight,
                rope_s,
                next.cos,
                next.sin,
                rows_s,
                stage_s,
                qin_s,
                k_n,
                v_n,
            );
        }
    };
}

mid_ss_t1024!(mid_ss_q4_port_t1024_body, stage2_s1);

// G12/P941 RSS pipelines the O epilogue, inter-slice transpose and row DMA per token pass.
#[cfg(feature = "compile-carried")]
mid_ss_t1024!(mid_ss_q4_port_t1024_f1_body, stage2_s1_pp);

// G12/P941 RFS uses token-cluster integer Full O.
#[cfg(feature = "compile-carried")]
mid_fs_t1024!(mid_fs_q4_t1024_nat_body, mid_fs_q4_fp8down_kv8_v9_2_xchg_nat);

#[cfg(feature = "compile-carried")]
use crate::device::tokenwise_int8::passes::ThreePasses;
#[cfg(feature = "compile-carried")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn mid_ss_q4_port_t1536_body<F: ffn_fast::DownForm>(
    device: &mut Device,
    out_s: &HbmTensor<bf16, Chip, m![Rq, Hqs]>,
    rows_s: &HbmTensor<i32, Chip, m![Tok1536, Hh]>,
    stage_s: &HbmTensor<i32, Chip, m![Tok1536]>,
    publish_s: &HbmTensor<i32, Chip, m![Tok1536]>,
    rope_s: &HbmTensor<i32, Chip, m![Tok1536]>,
    x: &mut HbmTensor<bf16, Chip, m![Tok1536, H]>,
    consts: Int8Consts<'_>,
    o: crate::ops_tokenwise_int8::SlidingOStaticF8<'_>,
    ffn: Q4FfnWeights<'_>,
    layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
    next: crate::ops_tokenwise_int8::StaticF8SlidingQkvWeights<'_>,
    k_l: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
    v_l: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
    qin_s: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
    k_n: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
    v_n: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
) {
    let (xa0, xa1, xa2) = pf_o::fp8c::pair::stage3_s1::<Tok1536, 256, 512, m![1], m![H / 64 # 64], m![H / 16 % 4], 64>(
        device, o.sigma, out_s, rows_s, x, o.wo, o.ws, o.post_attn_rms_weight,
    );
    moves::publish_kv8_cl::<Tok1536, m![Tok1536 / 768], m![Tok1536 % 128, 1 # 2], m![Tok1536 / 128 % 6, Ns, Ds]>(device, k_l, stage_s, publish_s);
    moves::publish_kv8_cl::<Tok1536, m![Tok1536 / 768], m![Tok1536 % 128, 1 # 2], m![Tok1536 / 128 % 6, Ns, Ds]>(device, v_l, stage_s, publish_s);
    let (xf0, xf1, xf2) = ffn_fast::ug4::serve::stage_dm3::<Tok1536, 256, 512, m![1], m![Lu / 1024 # 32, Lu / 256 % 2], m![Lu / 64 % 4], m![1], m![H / 64 # 64], m![H / 16 % 4], m![Tok1536 / 64], m![Tok1536 % 64], 64, 64, 16>(
        device, ffn.sigma, ffn.sigma_down,
        (&xa0, &xa1, &xa2),
        x,
        consts.eye,
        ffn.pre_ff_rms_weight,
        ffn.wug,
        ffn.wug_scale,
        ffn.wd,
        ffn.wd_scale,
        ffn.post_ff_rms_weight,
        layer_scalar,
        ffn.pre_signed,
        ffn.codes,
        ffn.scales,
        consts.h64,
        consts.h60,
    );
    pf_qkv::fp8_xchg::stage3::<Tok1536, 256, 512, ThreePasses, m![1], m![Nq / 64 % 64], m![Nq / 16 % 4], m![Tok1536 / 64], m![Tok1536 % 64], 64>(
        device, next.sigma,
        (&xf0, &xf1, &xf2),
        next.input_rms_weight,
        consts.eye,
        next.wq,
        next.ws,
        next.q_rms_weight,
        next.k_rms_weight,
        rope_s,
        next.cos,
        next.sin,
        rows_s,
        stage_s,
        qin_s,
        k_n,
        v_n,
    );
}
#[cfg(feature = "compile-carried")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn mid_fs_q4_t1536_body(
    device: &mut Device,
    out_f: &HbmTensor<bf16, Chip, m![Rq, Hqf]>,
    rows_f: &HbmTensor<i32, Chip, m![Tok1536, Hh]>,
    rope_s: &HbmTensor<i32, Chip, m![Tok1536]>,
    rows_s: &HbmTensor<i32, Chip, m![Tok1536, Hh]>,
    stage_s: &HbmTensor<i32, Chip, m![Tok1536]>,
    x: &mut HbmTensor<bf16, Chip, m![Tok1536, H]>,
    consts: Int8Consts<'_>,
    o: FullOInt8<'_>,
    ffn: Q4FfnWeights<'_>,
    layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
    next: Int8SlidingQkvWeights<'_>,
    qin_s: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
    k_n: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
    v_n: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
) {
    crate::device::tokenwise_int8::segments_full_i8::mid_fs_q4_fp8down_kv8_v9_3_xchg::<ffn_fast::GlOf<Tok1536>, Tok1536, 256, 512, crate::device::tokenwise_int8::passes::ThreePasses>(
        device, out_f, rows_f, rope_s, rows_s, stage_s, x, &consts, &o, &ffn, layer_scalar, &next, qin_s, k_n, v_n,
    );
}

#[cfg(feature = "compile-carried")]
macro_rules! standalone_ss {
    ($name:ident, $tok:ident, $body:ident) => {
        #[device(chip = 1)]
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            device: &mut Device,
            out_s: &HbmTensor<bf16, Chip, m![Rq, Hqs]>,
            rows_s: &HbmTensor<i32, Chip, m![$tok, Hh]>,
            stage_s: &HbmTensor<i32, Chip, m![$tok]>,
            publish_s: &HbmTensor<i32, Chip, m![$tok]>,
            rope_s: &HbmTensor<i32, Chip, m![$tok]>,
            x: &mut HbmTensor<bf16, Chip, m![$tok, H]>,
            consts: Int8Consts<'_>,
            o: crate::ops_tokenwise_int8::SlidingOStaticF8<'_>,
            ffn: Q4FfnWeights<'_>,
            layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
            next: crate::ops_tokenwise_int8::StaticF8SlidingQkvWeights<'_>,
            k_l: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
            v_l: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
            qin_s: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
            k_n: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
            v_n: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
        ) {
            $body::<ffn_fast::GlOf<$tok>>(device, out_s, rows_s, stage_s, publish_s, rope_s,
                x, consts, o, ffn, layer_scalar, next, k_l, v_l, qin_s, k_n, v_n);
        }
    };
}
#[cfg(feature = "compile-carried")]
standalone_ss!(mid_ss_t1536, Tok1536, mid_ss_q4_port_t1536_body);

#[cfg(feature = "compile-carried")]
macro_rules! standalone_fs {
    ($name:ident, $tok:ident, $body:ident) => {
        #[device(chip = 1)]
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            device: &mut Device,
            out_f: &HbmTensor<bf16, Chip, m![Rq, Hqf]>,
            rows_f: &HbmTensor<i32, Chip, m![$tok, Hh]>,
            rope_s: &HbmTensor<i32, Chip, m![$tok]>,
            rows_s: &HbmTensor<i32, Chip, m![$tok, Hh]>,
            stage_s: &HbmTensor<i32, Chip, m![$tok]>,
            x: &mut HbmTensor<bf16, Chip, m![$tok, H]>,
            consts: Int8Consts<'_>,
            o: FullOInt8<'_>,
            ffn: Q4FfnWeights<'_>,
            layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
            next: Int8SlidingQkvWeights<'_>,
            qin_s: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
            k_n: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
            v_n: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
        ) {
            $body(device, out_f, rows_f, rope_s, rows_s, stage_s, x, consts, o, ffn,
                layer_scalar, next, qin_s, k_n, v_n);
        }
    };
}
#[cfg(feature = "compile-carried")]
standalone_fs!(mid_fs_t1536, Tok1536, mid_fs_q4_t1536_body);
