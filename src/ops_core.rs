//! BF16 and INT8 decoder segments on the stationary-weight core.

// Carried kernels expand calls to the slicesO helpers by name.
#![allow(non_snake_case)]

use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;
use crate::device::layout::Cluster;
use crate::device::tokenwise_int8::ffn::Lu;
use crate::device::tokenwise_int8::ffn_fast;
use crate::device::proj_fast::{o as pf_o, qkv as pf_qkv};
use crate::device::tokenwise_int8::passes::{OnePass, TwoPasses};
#[cfg(feature = "compile-carried")]
use crate::device::tokenwise_int8::passes::ThreePasses;
use crate::device::tokenwise_int8::qkv::Nq;
use crate::device::tokenwise_int8::rotate::Rc;

/// One Full layer's O-stage weights for the BF16 core.
#[derive(DeviceSend)]
pub struct FullOBf16<'a> {
    pub o_weight: &'a HbmTensor<f8e4m3, Chip, m![H, Qf]>,
    pub o_weight_scale: &'a HbmTensor<bf16, Chip, m![H]>,
    pub post_attn_rms_weight: &'a HbmTensor<bf16, Chip, m![Rc, H]>,
}

/// One layer's FFN-stage weights for the BF16 core (NVFP4 packed weights, E4M3 block scales, f32
/// global scales as supplied; the norms as 16 copies). The layer scalar is a separate argument
/// (a padded mapping inside a parameter struct's field is an internal compiler error).
#[derive(DeviceSend)]
pub struct FfnBf16<'a> {
    pub pre_ff_rms_weight: &'a HbmTensor<bf16, Chip, m![Rc, H]>,
    pub up_weight_packed: &'a HbmTensor<f4e2m1, Chip, m![L, H]>,
    pub gate_weight_packed: &'a HbmTensor<f4e2m1, Chip, m![L, H]>,
    pub down_weight_packed: &'a HbmTensor<f4e2m1, Chip, m![H, L]>,
    pub up_weight_scale: &'a HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    pub gate_weight_scale: &'a HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    pub down_weight_scale: &'a HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    pub up_global_scale: &'a HbmTensor<f32, Chip, m![1]>,
    pub gate_global_scale: &'a HbmTensor<f32, Chip, m![1]>,
    pub down_global_scale: &'a HbmTensor<f32, Chip, m![1]>,
    pub post_ff_rms_weight: &'a HbmTensor<bf16, Chip, m![Rc, H]>,
}

pub mod ends;
pub mod prep;
pub mod full_i8;

pub(crate) mod moves;
/// Resident Sliding decode and INT8 layer transitions with FP8 KV.
/// Resident Sliding layer programs (`mid_ss` and the next layer's decode groups).
pub mod resident;
/// The resident Full -> Sliding programs (`mid_fs` and the next layer's whole Sliding attention).
#[cfg(feature = "compile-carried")]
pub mod resident_fs;
/// The first launch cleans the PE data cache over the firmware's shared fields.
pub mod spm_clean;
/// The Full-side resident programs (INT8-Full `mid_sf` and the Full layer's decode groups).
pub mod full_resident;
/// Q4 Gate/Up weights and bodies for the selected RSS and RFS layers.
pub(crate) mod q4serve;

// ------------------------------------------------------------------------------- INT8 carrier

pub use crate::ops_tokenwise_int8::{Int8Consts, Int8SlidingQkvWeights};

/// One Full layer's O-stage weights for the INT8 carrier: the prepared per-row codes
/// (columns in `rotate_heads` order for W = 512), their f32 scales, the O-input signs (16 copies)
/// and the norm.
#[derive(DeviceSend)]
pub struct FullOInt8<'a> {
    pub wo: &'a HbmTensor<i8, Chip, m![H, Qf]>,
    pub ws: &'a HbmTensor<f32, Chip, m![H]>,
    pub o_signs: &'a HbmTensor<bf16, Chip, m![Rc, Qf]>,
    pub post_attn_rms_weight: &'a HbmTensor<bf16, Chip, m![Rc, H]>,
}

// The 128-row projections use Sub fills to leave Main available for contractions.

macro_rules! segment_int8_fp8kv_entries {
    (
        $tok:ident, $g:literal, $qhalf:literal, $ps2:literal, $pp:ty,
        [$qkvm:ident],
        [$qring:ty, $qxs:ty, $qys:ty, $qtt:ty, $qtr:ty, $qts:literal, $ph:literal, $ps:literal, $p:literal],
        [$hs:ty, $he:ty, $ks:ty, $ke:ty],
        [$kft:ty, $kct:ty],
        $pre:ident $(, $pre_text:ident, [$embed_slice:ty, $embed_elem:ty])?
    ) => {
        /// INT8 `pre` with FP8 (E4M3) KV pools: layer 0's q into `qin_s` at `rows_s`, K/V into layer 0's pools at `stage_s`.
        #[device(chip = 1)]
        #[allow(clippy::too_many_arguments)]
        pub fn $pre(
            device: &mut Device,
            x: &HbmTensor<bf16, Chip, m![$tok, H]>,
            rope_s: &HbmTensor<i32, Chip, m![$tok]>,
            rows_s: &HbmTensor<i32, Chip, m![$tok, Hh]>,
            stage_s: &HbmTensor<i32, Chip, m![$tok]>,
            consts: Int8Consts<'_>,
            next: Int8SlidingQkvWeights<'_>,
            qin_s: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
            k0: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
            v0: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
        ) {
            let mut q = HbmTensor::<bf16, Chip, m![$tok, Qs]>::new();
            let mut k = HbmTensor::<bf16, Chip, m![$tok, Ps]>::new();
            let mut v = HbmTensor::<bf16, Chip, m![$tok, Ps]>::new();
            $qkvm::sliding_qkv_stage::<$tok, $g, $qhalf, $ps2, $pp, $qring, $qxs, $qys, $qtt, $qtr, $qts, $ph, $ps, $p>(
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
            moves::scatter_q::<$tok, $hs, $he>(device, q, rows_s, qin_s);
            moves::scatter_kv_e4m3::<$tok, $ks, $ke, $kft, $kct>(device, k, stage_s, k0);
            moves::scatter_kv_e4m3::<$tok, $ks, $ke, $kft, $kct>(device, v, stage_s, v0);
        }

        $(
            /// Text-only `pre`: gather the already-scaled embedding rows directly into
            /// the residual before running layer 0 QKV. Media steps use `$pre` so their
            /// encoder rows remain untouched by this gather.
            #[device(chip = 1)]
            #[allow(clippy::too_many_arguments)]
            pub fn $pre_text(
                device: &mut Device,
                embedding_table: &HbmTensor<bf16, Chip, m![W, H]>,
                index: &HbmTensor<i32, Chip, m![$tok]>,
                x: &mut HbmTensor<bf16, Chip, m![$tok, H]>,
                rope_s: &HbmTensor<i32, Chip, m![$tok]>,
                rows_s: &HbmTensor<i32, Chip, m![$tok, Hh]>,
                stage_s: &HbmTensor<i32, Chip, m![$tok]>,
                consts: Int8Consts<'_>,
                next: Int8SlidingQkvWeights<'_>,
                qin_s: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
                k0: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
                v0: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
            ) {
                let rows: DmTensor<bf16, Chip, Cluster, $embed_slice, $embed_elem> = embedding_table
                    .gather::<m![W], m![H]>()
                    .by_byte_offsets(index)
                    .to_dm(&mut device.tdma);
                rows.view().to_hbm_view(&mut device.tdma, x.view_mut());
                let mut q = HbmTensor::<bf16, Chip, m![$tok, Qs]>::new();
                let mut k = HbmTensor::<bf16, Chip, m![$tok, Ps]>::new();
                let mut v = HbmTensor::<bf16, Chip, m![$tok, Ps]>::new();
                $qkvm::sliding_qkv_stage::<$tok, $g, $qhalf, $ps2, $pp, $qring, $qxs, $qys, $qtt, $qtr, $qts, $ph, $ps, $p>(
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
                moves::scatter_q::<$tok, $hs, $he>(device, q, rows_s, qin_s);
                moves::scatter_kv_e4m3::<$tok, $ks, $ke, $kft, $kct>(device, k, stage_s, k0);
                moves::scatter_kv_e4m3::<$tok, $ks, $ke, $kft, $kct>(device, v, stage_s, v0);
            }
        )?

    };
}

segment_int8_fp8kv_entries!(
    Tok512, 64, 256, 512, OnePass,
    [pf_qkv],
    [m![1], m![Nq / 64 % 64], m![Nq / 16 % 4], m![Tok512 / 64], m![Tok512 % 64], 64, 32, 64, 8],
    [m![Tok512 % 128, Hh], m![Tok512 / 128, Hqs], m![Tok512 % 256], m![Tok512 / 256, Ns, Ds]],
    [m![Tok512 / 256, Ns, Ds / 16], m![Tok512 / 256, Ns, Ds / 8]],
    pre_int8_fp8kv_t512
);
segment_int8_fp8kv_entries!(
    Tok1024, 128, 256, 512, TwoPasses,
    [pf_qkv],
    [m![1], m![Nq / 64 % 64], m![Nq / 16 % 4], m![Tok1024 / 64], m![Tok1024 % 64], 64, 32, 64, 16],
    [m![Tok1024 % 128, Hh], m![Tok1024 / 128, Hqs], m![Tok1024 % 128, 1 # 2], m![Tok1024 / 128, Ns, Ds]],
    [m![Tok1024 / 128, Ns, Ds / 16], m![Tok1024 / 128, Ns, Ds / 8]],
    pre_int8_fp8kv_t1024, pre_int8_fp8kv_text_t1024,
    [m![Tok1024 % 128, 1 # 2], m![Tok1024 / 128, H]]
);

/// Standalone Q4 transitions for a 1,024-row step.
pub mod cohort_q4;

/// Tok128 segments use the Tok512 weight layout.
pub mod smallbatch;

#[cfg(feature = "compile-carried")]
#[device(chip = 1)]
#[allow(clippy::too_many_arguments)]
pub fn pre_int8_fp8kv_t1536(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Tok1536, H]>,
    rope_s: &HbmTensor<i32, Chip, m![Tok1536]>,
    rows_s: &HbmTensor<i32, Chip, m![Tok1536, Hh]>,
    stage_s: &HbmTensor<i32, Chip, m![Tok1536]>,
    consts: Int8Consts<'_>,
    next: Int8SlidingQkvWeights<'_>,
    qin_s: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
    k0: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
    v0: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
) {
    pre_int8_fp8kv_t1536_body(device, x, rope_s, rows_s, stage_s, consts, next, qin_s, k0, v0);
}

#[cfg(feature = "compile-carried")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn pre_int8_fp8kv_t1536_body(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Tok1536, H]>,
    rope_s: &HbmTensor<i32, Chip, m![Tok1536]>,
    rows_s: &HbmTensor<i32, Chip, m![Tok1536, Hh]>,
    stage_s: &HbmTensor<i32, Chip, m![Tok1536]>,
    consts: Int8Consts<'_>,
    next: Int8SlidingQkvWeights<'_>,
    qin_s: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
    k0: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
    v0: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
) {
    let mut q = HbmTensor::<bf16, Chip, m![Tok1536, Qs]>::new();
    let mut k = HbmTensor::<bf16, Chip, m![Tok1536, Ps]>::new();
    let mut v = HbmTensor::<bf16, Chip, m![Tok1536, Ps]>::new();
    pf_qkv::sliding_qkv_stage::<Tok1536, 192, 256, 512, ThreePasses, m![1], m![Nq / 64 % 64], m![Nq / 16 % 4], m![Tok1536 / 64], m![Tok1536 % 64], 64, 32, 64, 24>(
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
    moves::scatter_q::<Tok1536, m![Tok1536 % 128, Hh], m![Tok1536 / 128, Hqs]>(device, q, rows_s, qin_s);
    moves::scatter_kv_e4m3::<Tok1536, m![Tok1536 % 128, 1 # 2], m![Tok1536 / 128, Ns, Ds], m![Tok1536 / 128, Ns, Ds / 16], m![Tok1536 / 128, Ns, Ds / 8]>(device, k, stage_s, k0);
    moves::scatter_kv_e4m3::<Tok1536, m![Tok1536 % 128, 1 # 2], m![Tok1536 / 128, Ns, Ds], m![Tok1536 / 128, Ns, Ds / 16], m![Tok1536 / 128, Ns, Ds / 8]>(device, v, stage_s, v0);
}
