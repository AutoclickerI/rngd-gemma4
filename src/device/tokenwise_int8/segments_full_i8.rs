//! INT8 transitions across Full attention. Sliding-to-Full publishes the current
//! layer's staged E4M3 K/V and projects the next Full Q/K/V. Full-to-Sliding projects
//! the next Sliding Q/K/V and stages its E4M3 K/V for publication.

use furiosa_opt_std::prelude::*;

use super::core8_full;
use super::ffn_fast::{self, FastPasses};
use super::ffn::Lu;
use super::passes::Passes;
use super::qkv::Nq;
use super::rotate2;
use crate::Chip;
use crate::axes::{Ds, H, Hh, Hqf, Hqs, Ns, Rq, Rs};
use crate::device::proj_fast::heads::HeadsRowsEarlyPasses8;
use crate::device::proj_fast::qkv as pf_qkv;
use crate::ops_core::{FullOInt8, Int8Consts, Int8SlidingQkvWeights};

// Both Full O mappings hand the same two DM passes to the Q4 FFN and Sliding QKV.
macro_rules! mid_fs_dm2 {
    ($name:ident, $projection:ident) => {
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn $name<F: ffn_fast::DownForm, T: AxisName, const HALF: usize, const PS2: usize, PP: Passes + FastPasses + HeadsRowsEarlyPasses8 + core8_full::FullORowsPasses>(
            device: &mut Device,
            out_f: &HbmTensor<bf16, Chip, m![Rq, Hqf]>,
            rows_f: &HbmTensor<i32, Chip, m![T, Hh]>,
            rope_s: &HbmTensor<i32, Chip, m![T]>,
            rows_s: &HbmTensor<i32, Chip, m![T, Hh]>,
            stage_s: &HbmTensor<i32, Chip, m![T]>,
            x: &mut HbmTensor<bf16, Chip, m![T, H]>,
            consts: &Int8Consts<'_>,
            o: &FullOInt8<'_>,
            ffn: &crate::ops_core::q4serve::Q4FfnWeights<'_>,
            layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
            next: &Int8SlidingQkvWeights<'_>,
            qin_s: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
            k_n: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
            v_n: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
        ) {
            let dc = rotate2::dm_consts::<T, HALF>(device, consts.h64, consts.h60);
            let (xa0, xa1) = core8_full::$projection::<T, HALF, PS2, PP, m![1], m![H / 64 # 64], m![H / 16 % 4], m![T / 64], m![T % 64], 64>(
                device, out_f, rows_f, x, consts.eye, consts.h64, consts.h8, o.o_signs, o.wo, o.ws, o.post_attn_rms_weight,
            );
            // The Q4 Up/Gate and FP8 Down stage passes both token groups through DM.
            let (xf0, xf1) = ffn_fast::ug4::serve::stage_dm2::<T, HALF, PS2, m![1], m![Lu / 1024 # 32, Lu / 256 % 2], m![Lu / 64 % 4], m![1], m![H / 64 # 64], m![H / 16 % 4], m![T / 64], m![T % 64], 64, 64, 16>(
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
            pf_qkv::int8s_xchg::stage::<T, HALF, PS2, PP, m![1], m![Nq / 64 % 64], m![Nq / 16 % 4], m![T / 64], m![T % 64], 64>(
                device,
                (&xf0, &xf1),
                next.input_rms_weight_signed,
                &dc,
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

mid_fs_dm2!(mid_fs_q4_fp8down_kv8_v9_2_xchg, full_o_stage_i8_rows_dm2);

// G12/P941 reads natural-order rows in the token-cluster O contraction.
#[cfg(feature = "compile-carried")]
mid_fs_dm2!(mid_fs_q4_fp8down_kv8_v9_2_xchg_nat, full_o_stage_i8_rows_s256_dm2);

#[cfg(feature = "compile-carried")]
pub(crate) fn mid_fs_q4_fp8down_kv8_v9_3_xchg<F: ffn_fast::DownForm, T: AxisName, const HALF: usize, const PS2: usize, PP: Passes + FastPasses + HeadsRowsEarlyPasses8 + core8_full::FullORowsPasses>(
    device: &mut Device,
    out_f: &HbmTensor<bf16, Chip, m![Rq, Hqf]>,
    rows_f: &HbmTensor<i32, Chip, m![T, Hh]>,
    rope_s: &HbmTensor<i32, Chip, m![T]>,
    rows_s: &HbmTensor<i32, Chip, m![T, Hh]>,
    stage_s: &HbmTensor<i32, Chip, m![T]>,
    x: &mut HbmTensor<bf16, Chip, m![T, H]>,
    consts: &Int8Consts<'_>,
    o: &FullOInt8<'_>,
    ffn: &crate::ops_core::q4serve::Q4FfnWeights<'_>,
    layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
    next: &Int8SlidingQkvWeights<'_>,
    qin_s: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
    k_n: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
    v_n: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
) {
    let dc = rotate2::dm_consts::<T, HALF>(device, consts.h64, consts.h60);
    let (xa0, xa1, xa2) = core8_full::full_o_stage_i8_rows_dm3::<T, HALF, PS2, PP, m![1], m![H / 64 # 64], m![H / 16 % 4], m![T / 64], m![T % 64], 64>(
        device, out_f, rows_f, x, consts.eye, consts.h64, consts.h8, o.o_signs, o.wo, o.ws, o.post_attn_rms_weight,
    );
    let (xf0, xf1, xf2) = ffn_fast::ug4::serve::stage_dm3::<T, HALF, PS2, m![1], m![Lu / 1024 # 32, Lu / 256 % 2], m![Lu / 64 % 4], m![1], m![H / 64 # 64], m![H / 16 % 4], m![T / 64], m![T % 64], 64, 64, 16>(
        device, ffn.sigma, ffn.sigma_down, (&xa0, &xa1, &xa2), x, consts.eye, ffn.pre_ff_rms_weight, ffn.wug, ffn.wug_scale, ffn.wd, ffn.wd_scale,
        ffn.post_ff_rms_weight, layer_scalar, ffn.pre_signed, ffn.codes, ffn.scales, consts.h64, consts.h60,
    );
    pf_qkv::int8s_xchg::stage3::<T, HALF, PS2, PP, m![1], m![Nq / 64 % 64], m![Nq / 16 % 4], m![T / 64], m![T % 64], 64>(
        device,
        (&xf0, &xf1, &xf2),
        next.input_rms_weight_signed,
        &dc,
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
