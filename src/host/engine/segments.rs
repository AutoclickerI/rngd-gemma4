//! Host operands for decoder segments. `pre` projects layer 0 Q/K/V, each `mid` runs
//! O and FFN for its layer and projects the next layer's Q/K/V, and `last` serves
//! the emitting rows. Image and audio tokens enter the same residual after encoding.

use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;
use crate::host::engine::RowInput;
use crate::host::engine::buckets::{Bucket, Dense};
use crate::host::load::{FullLayer, MlpWeights, Weights};
use crate::host::runtime;
use crate::host::engine::core::CoreLayer;
use crate::ops_core;

type BoxError = Box<dyn std::error::Error>;

/// Rotary cosine and sine tables for Sliding and Full heads.
pub(crate) struct Rope {
    pub cos_s: HbmTensor<bf16, Chip, m![E, Ds]>,
    pub sin_s: HbmTensor<bf16, Chip, m![E, Ds]>,
    pub cos_f: HbmTensor<bf16, Chip, m![E, Df]>,
    pub sin_f: HbmTensor<bf16, Chip, m![E, Df]>,
}

impl Rope {
    pub async fn new(device: &mut Device) -> Result<Self, BoxError> {
        let (cos_s, sin_s) = runtime::upload_rope(device, Ds::SIZE, 10_000.0, Ds::SIZE).await?;
        let (cos_f, sin_f) = runtime::upload_rope(device, Df::SIZE, 1_000_000.0, 128).await?;
        Ok(Self { cos_s, sin_s, cos_f, sin_f })
    }
}

// Stationary-weight BF16 segment arguments use the supplied
// tensors, the norms as the `[Rc, H]` copies and a Sliding layer's Q/K/V as the concatenated
// `[Nq, H]` codes prepared at startup (`core::CoreLayer`) ----

pub(crate) fn core_full_o<'a>(l: &'a FullLayer, c: &'a CoreLayer) -> ops_core::FullOBf16<'a> {
    ops_core::FullOBf16 { o_weight: &l.o_weight, o_weight_scale: &l.o_weight_scale, post_attn_rms_weight: &c.post_attention_norm }
}

pub(crate) fn core_ffn<'a>(mlp: &'a MlpWeights, c: &'a CoreLayer) -> ops_core::FfnBf16<'a> {
    ops_core::FfnBf16 {
        pre_ff_rms_weight: &c.pre_feedforward_norm,
        up_weight_packed: &mlp.up_weight_packed,
        gate_weight_packed: &mlp.gate_weight_packed,
        down_weight_packed: &mlp.down_weight_packed,
        up_weight_scale: &mlp.up_weight_scale,
        gate_weight_scale: &mlp.gate_weight_scale,
        down_weight_scale: &mlp.down_weight_scale,
        up_global_scale: &mlp.up_global_scale,
        gate_global_scale: &mlp.gate_global_scale,
        down_global_scale: &mlp.down_global_scale,
        post_ff_rms_weight: &c.post_feedforward_norm,
    }
}

/// Media preparation at admission and gathering during a step.
pub(crate) struct MediaRows {
    encoder: crate::host::media::MediaEncoder,
}

impl MediaRows {

    pub async fn new(device: &mut Device, _rows: usize) -> Result<Self, Error> {
        Ok(Self { encoder: crate::host::media::MediaEncoder::new(device).await? })
    }

    pub async fn prepare(&mut self, device: &mut Device, weights:&Weights, requests: &[crate::host::media::MediaRequest<'_>]) -> Result<(), BoxError> {
        self.encoder.prepare(device, weights, requests).await
    }

    pub fn ready(&self, spans: &[crate::host::engine::Span<'_>]) -> bool { self.encoder.ready(spans) }
    pub fn begin(&mut self, spans: &[crate::host::engine::Span<'_>]) {
        self.encoder.begin(spans);
    }
    pub async fn embed_resident<B: Bucket>(
        &mut self, device: &mut Device, weights: &Weights, dense: &mut Dense<B::T>,
        spans: &[crate::host::engine::Span<'_>], logical_to_physical: &[usize],
    ) -> Result<(), BoxError> {
        self.encoder.embed_resident::<B>(device, weights, spans, dense, logical_to_physical).await
    }

    /// Every media row of `inputs` (step rows in order) into its row of `dense.x`.
    pub async fn embed<B: Bucket>(
        &mut self,
        device: &mut Device,
        weights: &Weights,
        dense: &mut Dense<B::T>,
        inputs: &[RowInput<'_>], logical_to_physical: &[usize],
    ) -> Result<(), BoxError> {
        self.encoder.embed::<B>(device, weights, inputs, dense, Some(logical_to_physical)).await
    }
}
