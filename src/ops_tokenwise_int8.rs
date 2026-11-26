//! Rotated INT8 tokenwise kernels composed with BF16 stages.
//! See `device::tokenwise_int8` for the declared arithmetic.
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;
use crate::device::tokenwise_int8::ffn::Lu;
use crate::device::tokenwise_int8::rotate_ff::{K16, Kd, Kq4, Of};
use crate::device::tokenwise_int8::ffn_fast::fp8c::Lp;
use crate::device::tokenwise_int8::rotate_heads::O8;
use crate::device::tokenwise_int8::qkv::Nq;
use crate::device::tokenwise_int8::rotate::{H60Hbm, Kb, Ob, Rc, Sgi, Sl8};

/// One Sliding layer's INT8 QKV-stage weights (prepared at startup).
#[derive(DeviceSend)]
pub struct Int8SlidingQkvWeights<'a> {
    /// The input RMSNorm gamma times the rotation's signs.
    pub input_rms_weight_signed: &'a HbmTensor<bf16, Chip, m![Rc, H]>,
    /// Fused Q | K | V rows, rotated input order, per-row INT8.
    pub wq: &'a HbmTensor<i8, Chip, m![Nq, H]>,
    /// `f32(s_w)` per row.
    pub ws: &'a HbmTensor<f32, Chip, m![Nq]>,
    pub q_rms_weight: &'a HbmTensor<bf16, Chip, m![Ds]>,
    pub k_rms_weight: &'a HbmTensor<bf16, Chip, m![Ds]>,
    pub cos: &'a HbmTensor<bf16, Chip, m![E, Ds]>,
    pub sin: &'a HbmTensor<bf16, Chip, m![E, Ds]>,
}

/// The rotation constants every INT8 stage reads.
#[derive(DeviceSend)]
pub struct Int8Consts<'a> {
    pub h64: &'a HbmTensor<bf16, Chip, m![Rc, Ob, Kb]>,
    pub h60: &'a HbmTensor<bf16, Chip, m![Rc, { H60Hbm }]>,
    pub eye: &'a HbmTensor<f32, Chip, m![Sgi, Sl8]>,
    pub h4: &'a HbmTensor<bf16, Chip, m![Rc, Of, K16]>,
    /// Sylvester H8 `[O8, K16]` (rows zero-padded to 16): the Full O-input rotation's H_B.
    pub h8: &'a HbmTensor<bf16, Chip, m![Rc, O8, K16]>,
}

// Static FP8 projections use E4M3 rows in natural input order and plain norm gammas.
// Up/Gate includes 2,048 zero rows so every weight lane receives finite data. Preserve
// the field layout for carried kernels that share the argument ABI.

#[derive(DeviceSend)]
pub struct SlidingOStaticF8<'a> {
    pub sigma: &'a HbmTensor<f32, Chip, m![1 # 8]>,
    /// Per-row E4M3 codes (`amax / 256`), natural `Qs` order.
    pub wo: &'a HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    /// `f32(s_w)` per row.
    pub ws: &'a HbmTensor<f32, Chip, m![H]>,
    /// Unread (the INT8 O-input signs' slot).
    pub o_signs: &'a HbmTensor<bf16, Chip, m![Rc, Qs]>,
    pub post_attn_rms_weight: &'a HbmTensor<bf16, Chip, m![Rc, H]>,
}

#[derive(DeviceSend)]
pub struct StaticF8FfnWeights<'a> {
    pub sigma_down: &'a HbmTensor<f32, Chip, m![1 # 8]>,
    pub sigma: &'a HbmTensor<f32, Chip, m![1 # 8]>,
    /// `Rc` copies of the plain pre-FFN norm gamma.
    pub pre_ff_rms_weight: &'a HbmTensor<bf16, Chip, m![Rc, H]>,
    /// Up and Gate rows in the `Lu` order, natural `H` order, per-row E4M3, then 2,048 zero rows.
    pub wug: &'a HbmTensor<f8e4m3, Chip, m![Lp, H]>,
    /// `f32(s_w) * f32(g)` per row.
    pub wug_scale: &'a HbmTensor<f32, Chip, m![Lu]>,
    /// Down, natural input order, quarter-major, per-row E4M3.
    pub wd: &'a HbmTensor<f8e4m3, Chip, m![Kq4, H, Kd]>,
    pub wd_scale: &'a HbmTensor<f32, Chip, m![H]>,
    /// `Rc` copies of the post-FFN norm gamma.
    pub post_ff_rms_weight: &'a HbmTensor<bf16, Chip, m![Rc, H]>,
}

#[derive(DeviceSend)]
pub struct StaticF8SlidingQkvWeights<'a> {
    pub sigma: &'a HbmTensor<f32, Chip, m![1 # 8]>,
    /// `Rc` copies of the plain input RMSNorm gamma.
    pub input_rms_weight: &'a HbmTensor<bf16, Chip, m![Rc, H]>,
    /// Fused Q | K | V rows, natural input order, per-row E4M3.
    pub wq: &'a HbmTensor<f8e4m3, Chip, m![Nq, H]>,
    /// `f32(s_w)` per row.
    pub ws: &'a HbmTensor<f32, Chip, m![Nq]>,
    pub q_rms_weight: &'a HbmTensor<bf16, Chip, m![Ds]>,
    pub k_rms_weight: &'a HbmTensor<bf16, Chip, m![Ds]>,
    pub cos: &'a HbmTensor<bf16, Chip, m![E, Ds]>,
    pub sin: &'a HbmTensor<bf16, Chip, m![E, Ds]>,
}
