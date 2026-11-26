//! The server's prepared INT8 decoder and W4 output head.
use furiosa_opt_std::prelude::*;

use super::core::CoreWeights;
use super::int8::Int8Weights;
use crate::host::load::{Layer, Weights};

type BoxError = Box<dyn std::error::Error>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Pre,
    MidSs,
    MidSf,
    MidFs,
}

impl Kind {

    pub fn name(self) -> &'static str {
        match self {
            Kind::Pre => "pre",
            Kind::MidSs => "mid_ss",
            Kind::MidSf => "mid_sf",
            Kind::MidFs => "mid_fs",
        }
    }
}

pub(crate) struct Carriers {
    pub core: CoreWeights,
    pub int8: Int8Weights,
}

impl Carriers {
    pub async fn prepare(device: &mut Device, weights: &mut Weights, checkpoint: std::sync::Arc<memmap2::Mmap>, q4_prefetch: super::int8::q4serve::Prefetch) -> Result<Self, BoxError> {
        let core = CoreWeights::prepare(device, weights, checkpoint).await?;
        let mut plan = vec![(0, Kind::Pre)];
        for (l, pair) in weights.layers.windows(2).enumerate() {
            let kind = match (&pair[0], &pair[1]) {
                (Layer::Sliding(_), Layer::Sliding(_)) => Kind::MidSs,
                (Layer::Sliding(_), Layer::Full(_)) => Kind::MidSf,
                (Layer::Full(_), Layer::Sliding(_)) => Kind::MidFs,
                (Layer::Full(_), Layer::Full(_)) => return Err("two consecutive Full layers have no segment".into()),
            };
            plan.push((l, kind));
        }
        let int8 = Int8Weights::prepare(device, weights, &plan, q4_prefetch).await?;
        Ok(Self { core, int8 })
    }

    /// Validate each argument constructor against the prepared model.
    pub fn check_arguments(&self, weights: &Weights, rope: &super::segments::Rope) -> Result<(), BoxError> {
        let w = &self.int8;
        for &(l, kind) in &w.plan {
            match (kind, &weights.layers[if kind == Kind::Pre { 0 } else { l + 1 }]) {
                (Kind::Pre, Layer::Sliding(next)) => { w.pre(next, rope)?; }
                (Kind::MidSs, Layer::Sliding(next)) => { w.mid_ss(l, next, &self.core, rope)?; }
                (Kind::MidSf, Layer::Full(next)) => { w.mid_sf(l, next, &self.core, rope)?; }
                (Kind::MidFs, Layer::Sliding(next)) => { w.mid_fs(l, next, &self.core, rope)?; }
                _ => return Err(format!("invalid prepared layer {l} for {kind:?}").into()),
            }
        }
        Ok(())
    }
}
