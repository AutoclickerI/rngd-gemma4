//! Host bindings for the stationary-weight segments. Decoder segments gather attention
//! output and publish K/V directly. The final segment gathers layer 47's attention output
//! for the emitting rows.

use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;
use crate::host::engine::LayerPool;
use crate::host::engine::attention::AttentionState;
use crate::host::engine::buckets::{Bucket, Dense};
use crate::host::engine::carriers::Carriers;
use crate::host::engine::segments::Rope;
use crate::host::load::{Layer, Weights};

type BoxError = Box<dyn std::error::Error>;

/// Replicated norm gammas.
pub(crate) struct CoreLayer {
    pub post_attention_norm: HbmTensor<bf16, Chip, m![Rc, H]>,
    pub pre_feedforward_norm: HbmTensor<bf16, Chip, m![Rc, H]>,
    pub post_feedforward_norm: HbmTensor<bf16, Chip, m![Rc, H]>,
}

/// The core's prepared weights for every layer.
pub(crate) struct CoreWeights {
    pub layers: Vec<CoreLayer>,
    pub w4_head: super::head4::Head,
}

impl CoreWeights {
    /// Replicate the normalization vectors and prepare the W4 head.
    pub async fn prepare(device: &mut Device, weights: &mut Weights, checkpoint: std::sync::Arc<memmap2::Mmap>) -> Result<Self, BoxError> {
        // Quantization uses only checkpoint bytes. Start it while device uploads are in flight.
        let w4_preparation = super::head4::Preparation::start(checkpoint)?;
        let mut layers = Vec::with_capacity(weights.layers.len());
        for layer in weights.layers.iter_mut() {
            let (post_attention, pre_ff, post_ff) = match layer {
                Layer::Sliding(l) => (&l.post_attention_norm, &l.pre_feedforward_norm, &l.post_feedforward_norm),
                Layer::Full(l) => (&l.post_attention_norm, &l.pre_feedforward_norm, &l.post_feedforward_norm),
            };
            layers.push(CoreLayer {
                post_attention_norm: copies(device, post_attention).await?,
                pre_feedforward_norm: copies(device, pre_ff).await?,
                post_feedforward_norm: copies(device, post_ff).await?,
            });
        }
        let w4_head = super::head4::Head::upload(device, w4_preparation.finish()?, &weights.final_norm).await?;
        Ok(Self { layers, w4_head })
    }
}

/// `[H]` -> `[Rc, H]`, every copy equal.
async fn copies(device: &mut Device, vector: &HbmTensor<bf16, Chip, m![H]>) -> Result<HbmTensor<bf16, Chip, m![Rc, H]>, Error> {
    let host: HostTensor<bf16, m![H]> = vector.to_host(&mut device.pdma).await?;
    crate::host::load::upload(device, super::int8::copies(&host.into_vec())).await
}

/// `ops_core::mid_{ss,sf,fs}_{c}_tN`: layer `l`'s output halves gathered from its row space, a
/// Sliding layer's staged K/V published into the rings, O and FFN of `l`, Q/K/V of `l + 1` with
/// the query halves into the next family's `qin` and K/V into `next_pool`. `Core` launches the
/// bucket's kernel (`Bucket::core_mid`); `Int8` the bucket's INT8 `mid_ss` (`Bucket::int8_mid_ss`)
/// with layer `l`'s prepared INT8 weights (the hybrid Full transitions at `mid_sf`/`mid_fs`).
#[allow(clippy::too_many_arguments)]
pub(crate) async fn mid<B: Bucket>(
    device: &mut Device,
    carriers: &Carriers,
    l: usize,
    dense: &mut Dense<B::T>,
    layer: &Layer,
    next: &Layer,
    rope: &Rope,
    official: &mut AttentionState,
    pool: &mut LayerPool,
    next_pool: &mut LayerPool,
) -> Result<(), BoxError> {
    let (i8w, c) = (&carriers.int8, &carriers.core);
    Ok(match (layer, next, pool, next_pool) {
        (Layer::Sliding(this), Layer::Sliding(n), LayerPool::Sliding { k: k_l, v: v_l }, LayerPool::Sliding { k: k_n, v: v_n }) => {
            B::int8_mid_ss(device, dense, &this.layer_scalar, i8w.mid_ss(l, n, c, rope)?, official, k_l, v_l, k_n, v_n).await?
        }
        (Layer::Sliding(this), Layer::Full(n), LayerPool::Sliding { k: k_l, v: v_l }, LayerPool::Full { k: k_n, v: v_n }) => {
            B::int8_mid_sf(device, dense, &this.layer_scalar, i8w.mid_sf(l, n, c, rope)?, official, k_l, v_l, k_n, v_n).await?
        }
        (Layer::Full(this), Layer::Sliding(n), LayerPool::Full { .. }, LayerPool::Sliding { k: k_n, v: v_n }) => {
            B::int8_mid_fs(device, dense, &this.layer_scalar, i8w.mid_fs(l, n, c, rope)?, official, k_n, v_n).await?
        }
        _ => return Err(format!("INT8 mid({l}): layer and pool families disagree, or two consecutive Full layers").into()),
    })
}
