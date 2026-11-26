//! Weight preparation for the INT8 carrier. The engine uses natural and static-FP8
//! segment weights, with Q4 Gate/Up prepared for the selected layers.
//!
//! A middle segment runs O and FFN for its layer and Q/K/V for the next layer.
//! Prepare the tensors in that order so the startup loader can release each source
//! tensor once its last prepared representation is ready.
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{H, L, Pf, Qf, Qs, Rc};
use crate::device::tokenwise_int8::ffn::Lu;
use crate::device::tokenwise_int8::qkv::Nq;
use crate::device::tokenwise_int8::rotate::{H60Hbm, Kb, Ob, Sgi, Sl8};
use crate::device::tokenwise_int8::rotate_ff::{K16, Kd, Kq4, Of};
use crate::device::tokenwise_int8::rotate_heads::O8;
use crate::host::engine::carriers::Kind;
use crate::host::engine::core::CoreWeights;
use crate::host::engine::segments::Rope;
use crate::host::load::{FullLayer, Layer, MlpWeights, SlidingLayer, Weights, upload as up};
use crate::ops_core::prep::WugRows;
use crate::ops_core::{FullOInt8, Int8Consts, Int8SlidingQkvWeights};

type BoxError = Box<dyn std::error::Error>;

pub(crate) mod natural;
mod fp8_scales;
mod fp8_row_mse;
pub(crate) mod q4cal;
pub(crate) mod q4serve;

/// INT8 QKV on each Full layer in `mid_sf` and INT8 O on layers 5..41 in `mid_fs`.
/// Layer 47's O and FFN stay on the core's `last`.
macro_rules! int8_full_kernels {
    ({ $($on:tt)* }) => {
        { $($on)* }
    };
}
pub(crate) use int8_full_kernels;

/// Sliding KV uses raw E4M3. Segment kernels publish 2,048-byte rows to
/// the pools for the DSL attention kernels.
pub type SlidingKv = f8e4m3;

/// Rotation sites: the sign seed of layer `l` and site `s` is `l * 8 + s`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Site {
    Mlp = 2,
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The signs `D` of `n` elements at (`layer`, `site`): element `i` is bit `i mod 64` of the
/// SplitMix64 stream's word `i div 64` (seed `layer * 8 + site`), bit 0 -> +1, bit 1 -> -1.
pub(crate) fn signs(layer: usize, site: Site, n: usize) -> Vec<f32> {
    signs_from_seed((layer * 8 + site as usize) as u64, n)
}

pub(super) fn signs_from_seed(mut state: u64, n: usize) -> Vec<f32> {
    (0..n.div_ceil(64))
        .flat_map(|_| {
            let word = splitmix64(&mut state);
            (0..64).map(move |i| if (word >> i) & 1 == 0 { 1.0 } else { -1.0 })
        })
        .take(n)
        .collect()
}

/// Sylvester Hadamard of a power-of-two order: `h[i][j] = (-1)^popcount(i & j)`.
pub(super) fn sylvester(n: usize) -> Vec<f32> {
    (0..n * n).map(|ij| if ((ij / n) & (ij % n)).count_ones() % 2 == 0 { 1.0 } else { -1.0 }).collect()
}

/// Paley construction I of order `q + 1` (`q = 59`).
pub(super) fn paley(n: usize) -> Vec<f32> {
    let q = n - 1;
    let residues: std::collections::HashSet<usize> = (1..q).map(|i| (i * i) % q).collect();
    let chi = |a: usize| -> f32 {
        if a == 0 {
            0.0
        } else if residues.contains(&a) {
            1.0
        } else {
            -1.0
        }
    };
    let mut h = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let core = if i == 0 && j == 0 {
                0.0
            } else if i == 0 {
                1.0
            } else if j == 0 {
                -1.0
            } else {
                chi((j + q - i) % q)
            };
            h[i * n + j] = core + if i == j { 1.0 } else { 0.0 };
        }
    }
    h
}

/// In-place Sylvester-ordered Walsh-Hadamard transform of 64 values (`X[k] = sum_j (-1)^popcount(k & j) x[j]`).
pub(super) fn fwht64(x: &mut [f32]) {
    let mut len = 1;
    while len < 64 {
        for i in (0..64).step_by(2 * len) {
            for j in i..i + len {
                let (a, b) = (x[j], x[j + len]);
                x[j] = a + b;
                x[j + len] = a - b;
            }
        }
        len *= 2;
    }
}
/// H60 in the kernels' TRF order `[Oa / 32][Oa % 8][Ka][Oa / 8 % 4]`, rows 60..63 zero.
pub(super) fn h60_trf_order(h60: &[f32]) -> Vec<bf16> {
    let mut v = vec![bf16::from_f32(0.0); 64 * 60];
    for half in 0..2 {
        for lane in 0..8 {
            for a in 0..60 {
                for grp in 0..4 {
                    let ao = half * 32 + grp * 8 + lane;
                    let val = if ao < 60 { h60[ao * 60 + a] } else { 0.0 };
                    v[((half * 8 + lane) * 60 + a) * 4 + grp] = bf16::from_f32(val);
                }
            }
        }
    }
    v
}

/// A `[b, b]` Hadamard as `[b, 16]` rows zero-padded to 16.
fn padded16(h: &[f32], b: usize) -> Vec<bf16> {
    (0..b).flat_map(|o| (0..16).map(move |k| bf16::from_f32(if k < b { h[o * b + k] } else { 0.0 }))).collect()
}

/// `Rc` copies of one vector.
pub(super) fn copies<D: Copy>(v: &[D]) -> Vec<D> {
    (0..Rc::SIZE).flat_map(|_| v.iter().copied()).collect()
}

/// `gamma * D` (exact: a sign flip).
fn signed(gamma: &[bf16], d: &[f32]) -> Vec<bf16> {
    gamma.iter().zip(d).map(|(g, s)| if *s < 0.0 { bf16::from_bits(g.to_bits() ^ 0x8000) } else { *g }).collect()
}

pub(super) fn eye() -> impl Iterator<Item = f32> {
    (0..64).map(|i| if i / 8 == i % 8 { 1.0 } else { 0.0 })
}

/// Shared transform constants and unit row factors for FP8 weight preparation.
pub(crate) struct Consts {
    h64: HbmTensor<bf16, Chip, m![Rc, Ob, Kb]>,
    h60: HbmTensor<bf16, Chip, m![Rc, { H60Hbm }]>,
    eye: HbmTensor<f32, Chip, m![Sgi, Sl8]>,
    h4: HbmTensor<bf16, Chip, m![Rc, Of, K16]>,
    h8: HbmTensor<bf16, Chip, m![Rc, O8, K16]>,
    ones: HbmTensor<f32, Chip, m![L]>,
}

impl Consts {
    async fn new(device: &mut Device) -> Result<Self, Error> {
        let h64: Vec<bf16> = sylvester(64).iter().map(|v| bf16::from_f32(*v)).collect();
        Ok(Self {
            h64: up(device, copies(&h64)).await?,
            h60: up(device, copies(&h60_trf_order(&paley(60)))).await?,
            eye: up(device, eye()).await?,
            h4: up(device, copies(&padded16(&sylvester(4), 4))).await?,
            h8: up(device, copies(&padded16(&sylvester(8), 8))).await?,
            ones: up(device, vec![1.0f32; L::SIZE]).await?,
        })
    }

    fn args(&self) -> Int8Consts<'_> {
        Int8Consts { h64: &self.h64, h60: &self.h60, eye: &self.eye, h4: &self.h4, h8: &self.h8 }
    }
}

// ---- per-layer prepared tensors ----

/// A Full layer's per-row INT8 O weights and per-channel inverse smoothing.
pub(crate) struct OFullInt8 {
    wo: HbmTensor<i8, Chip, m![H, Qf]>,
    ws: HbmTensor<f32, Chip, m![H]>,
    d_o: HbmTensor<bf16, Chip, m![Rc, Qf]>,
}

/// A Sliding layer's Q/K/V as an INT8 segment produces them: the signed input gamma and `wq`
/// (`[Nq, H]`, rows Q | K | V) with its scales.
pub(crate) struct QkvInt8 {
    input_signed: HbmTensor<bf16, Chip, m![Rc, H]>,
    wq: HbmTensor<i8, Chip, m![Nq, H]>,
    ws: HbmTensor<f32, Chip, m![Nq]>,
}

/// The arguments of one `ops_core::pre_int8_tN` launch besides the step tensors and pools.
pub(crate) struct PreInt8<'a> {
    pub consts: Int8Consts<'a>,
    pub next: Int8SlidingQkvWeights<'a>,
}

/// The arguments of one `ops_core::mid_ss_int8_tN` launch besides the step tensors and pools.
pub(crate) struct MidSsInt8<'a> {
    pub consts: Int8Consts<'a>,
    pub o: crate::ops_tokenwise_int8::SlidingOStaticF8<'a>,
    pub next: crate::ops_tokenwise_int8::StaticF8SlidingQkvWeights<'a>,
    /// Q4 FFN arguments shared by stock and resident segments.
    pub q4: crate::ops_core::q4serve::Q4FfnWeights<'a>,
}
/// The arguments of one `ops_core::full_i8::mid_sf_int8_tN` launch (layer `l` Sliding, `l + 1` Full).
pub(crate) struct MidSfInt8<'a> {
    pub consts: Int8Consts<'a>,
    /// Static FP8 FFN arguments for the Sliding-to-Full transition.
    pub ffn_f8: crate::ops_tokenwise_int8::StaticF8FfnWeights<'a>,
    /// Static FP8 Sliding O and Full Q/K.
    pub o_f8: crate::ops_tokenwise_int8::SlidingOStaticF8<'a>,
    pub next_f8: crate::ops_core::full_resident::FullQkvStaticF8<'a>,
}

/// The arguments of one `ops_core::full_i8::mid_fs_int8_tN` launch (layer `l` Full, `l + 1` Sliding).
pub(crate) struct MidFsInt8<'a> {
    pub consts: Int8Consts<'a>,
    pub o: FullOInt8<'a>,
    pub next: Int8SlidingQkvWeights<'a>,
    /// Q4 Gate/Up and static FP8 Down.
    pub q4: crate::ops_core::q4serve::Q4FfnWeights<'a>,
}

/// The INT8 carrier's prepared weights for every layer.
pub(crate) struct Int8Weights {
    consts: Consts,
    natural_layers: Vec<natural::Layer>,
    fp8_slot_layer: usize,
    /// The INT8 segments in layer order.
    pub plan: Vec<(usize, Kind)>,
}

/// Layer `l`'s MLP and pre-FFN norm, either family.
fn mlp_of(weights: &Weights, l: usize) -> (&MlpWeights, &HbmTensor<bf16, Chip, m![H]>) {
    match &weights.layers[l] {
        Layer::Sliding(s) => (&s.mlp, &s.pre_feedforward_norm),
        Layer::Full(f) => (&f.mlp, &f.pre_feedforward_norm),
    }
}

fn mlp_of_mut(weights: &mut Weights, l: usize) -> &mut MlpWeights {
    match &mut weights.layers[l] {
        Layer::Sliding(s) => &mut s.mlp,
        Layer::Full(f) => &mut f.mlp,
    }
}

impl Int8Weights {

    /// Prepare O, FFN and Q/K/V in separate passes so each source tensor can be freed
    /// after its final use.
    pub async fn prepare(device: &mut Device, weights: &mut Weights, plan: &[(usize, Kind)], q4_prefetch: q4serve::Prefetch) -> Result<Self, BoxError> {
        let row_mse = fp8_row_mse::RowMse::load()?;
        let static_scales = fp8_scales::load()?;
        let n = weights.layers.len();
        for &(l, kind) in plan {
            let fam = |i: usize| matches!(weights.layers.get(i), Some(Layer::Full(_)));
            let ok = match kind {
                Kind::MidSs => l + 1 < n && !fam(l) && !fam(l + 1),
                Kind::MidSf => l + 1 < n && !fam(l) && fam(l + 1),
                Kind::MidFs => l + 1 < n && fam(l) && !fam(l + 1),
                Kind::Pre => l == 0 && !fam(0),
            };
            if !ok {
                return Err(format!("INT8 plan: {} at layer {l} does not match the layer families", kind.name()).into());
            }
        }
        let consts = Consts::new(device).await?;
        let factors = natural::Factors::load(&weights.layers)?;
        let mut natural_layers: Vec<natural::Layer> = (0..n).map(|_|natural::Layer::default()).collect();
        // Validate that the segment plan matches the calibration's layer set.
        let q4_layers: Vec<_> = plan.iter().filter_map(|&(l, kind)| {
            matches!(kind, Kind::MidSs | Kind::MidFs).then_some(l)
        }).collect();
        // O of l.
        for &(l, kind) in plan {
            match (&mut weights.layers[l], kind) {
                (_, Kind::Pre) => continue,
                (Layer::Sliding(s), Kind::MidSs | Kind::MidSf) => {
                    natural_layers[l].fp8_o = Some(natural::prep_fp8_o(device, s, static_scales[l][1]).await?);
                    s.o_weight_scale = HbmTensor::new();
                }
                (Layer::Full(f), Kind::MidFs) => {
                    natural_layers[l].o_full = Some(natural::prep_o_full(device, &consts, f, factors[l].o).await?);
                    f.o_weight = HbmTensor::new();
                    f.o_weight_scale = HbmTensor::new();
                }
                _ => unreachable!("checked above"),
            }
        }
        let mut q4_host = q4_prefetch.finish(&q4_layers)?.into_iter();
        let fp8_slot_layer = plan.iter().find_map(|&(l, kind)| (kind == Kind::MidSf).then_some(l))
            .ok_or("missing Sliding-to-Full segment for FP8 argument slots")?;
        for &(l, kind) in plan.iter().filter(|(_, kind)| *kind != Kind::Pre) {
            let (mlp, pre) = mlp_of(weights, l);
            let ratios = row_mse.layer(l)?;
            let prepared = &mut natural_layers[l];
            if kind == Kind::MidSf {
                prepared.fp8_gate_up = Some(natural::prep_fp8_gate_up(
                    device, &consts, mlp, pre, static_scales[l][2], &ratios,
                ).await?);
            }
            prepared.fp8_down = Some(natural::prep_fp8_down(
                device, &consts, mlp, static_scales[l][3], ratios.down,
            ).await?);
            if kind != Kind::MidSf {
                let host = q4_host.next().ok_or("missing Q4 host weights")?;
                prepared.q4 = Some(q4serve::upload(device, l, host, pre).await?);
            }
            release_mlp(mlp_of_mut(weights, l));
        }
        // Q/K/V of l + 1.
        for &(l, kind) in plan {
            match kind {
                Kind::MidSs | Kind::MidFs | Kind::Pre => {
                    // `pre` produces layer 0's Q/K/V, a `mid(l)` layer `l + 1`'s.
                    let q = if kind == Kind::Pre { l } else { l + 1 };
                    let Layer::Sliding(next) = &mut weights.layers[q] else { unreachable!("checked above") };
                    if kind == Kind::MidSs {
                        natural_layers[q].fp8_qkv = Some(natural::prep_fp8_qkv(device, next, static_scales[q][0]).await?);
                    } else {
                        natural_layers[q].qkv = Some(natural::prep_qkv(
                            device, &consts, next, factors[q].qkv,
                        ).await?);
                        next.qkv_weight = HbmTensor::new();
                    }
                    next.qkv_weight_scale = HbmTensor::new();
                }
                Kind::MidSf => {
                    let Layer::Full(next) = &mut weights.layers[l + 1] else { unreachable!("checked above") };
                    natural_layers[l + 1].fp8_qk_full = Some(natural::prep_fp8_qk_full(device, next).await?);
                }
            }
        }
        Ok(Self { consts, natural_layers, fp8_slot_layer, plan: plan.to_vec() })
    }

    fn unprepared(what: &'static str) -> Error {
        Error::Uncompiled { path: what, key: vec![] }
    }

    /// Static FP8 FFN arguments for a Sliding-to-Full segment.
    fn ffn_f8_args<'a>(&'a self, l: usize, core: &'a CoreWeights) -> Result<crate::ops_tokenwise_int8::StaticF8FfnWeights<'a>, Error> {
        self.ffn_args(l, l, core)
    }

    fn ffn_args<'a>(&'a self, l: usize, slots: usize, core: &'a CoreWeights) -> Result<crate::ops_tokenwise_int8::StaticF8FfnWeights<'a>, Error> {
        let down = self.natural_layers[l].fp8_down.as_ref().ok_or_else(|| Self::unprepared("FP8 Down missing"))?;
        let gate_up = self.natural_layers[slots].fp8_gate_up.as_ref().ok_or_else(|| Self::unprepared("FP8 Gate/Up missing"))?;
        Ok(down.args(&core.layers[l].post_feedforward_norm, gate_up))
    }

    fn q4_args<'a>(&'a self, l: usize, core: &'a CoreWeights) -> Result<crate::ops_core::q4serve::Q4FfnWeights<'a>, Error> {
        // Q4 images retain the FP8 Gate/Up slots in their ABI. Bind an already
        // resident FP8 Gate/Up projection to those unread slots.
        let args = self.ffn_args(l, self.fp8_slot_layer, core)?;
        Ok(self.natural_layers[l].q4.as_ref().ok_or_else(|| Self::unprepared("Q4 Gate/Up missing"))?.args(&args))
    }

    fn o_full_args<'a>(&'a self, l: usize, core: &'a CoreWeights) -> Result<FullOInt8<'a>, Error> {
        let layer = &self.natural_layers[l].o_full;
        let o = layer.as_ref().ok_or_else(|| Self::unprepared("int8: this layer's INT8 Full O was not prepared at engine start"))?;
        Ok(FullOInt8 { wo: &o.wo, ws: &o.ws, o_signs: &o.d_o, post_attn_rms_weight: &core.layers[l].post_attention_norm })
    }

    fn qkv_args<'a>(&'a self, n: usize, next: &'a SlidingLayer, rope: &'a Rope) -> Result<Int8SlidingQkvWeights<'a>, Error> {
        let q = self.natural_layers.get(n).and_then(|x| x.qkv.as_ref());
        let q = q.ok_or_else(|| Self::unprepared("int8: this layer's INT8 Sliding Q/K/V was not prepared at engine start"))?;
        Ok(Int8SlidingQkvWeights {
            input_rms_weight_signed: &q.input_signed,
            wq: &q.wq,
            ws: &q.ws,
            q_rms_weight: &next.q_norm,
            k_rms_weight: &next.k_norm,
            cos: &rope.cos_s,
            sin: &rope.sin_s,
        })
    }

    /// The INT8 `pre` launch arguments: layer 0's Q/K/V.
    pub fn pre<'a>(&'a self, layer0: &'a SlidingLayer, rope: &'a Rope) -> Result<PreInt8<'a>, Error> {
        Ok(PreInt8 { consts: self.consts.args(), next: self.qkv_args(0, layer0, rope)? })
    }

    /// The INT8 `mid_ss(l)` launch arguments: layer `l`'s O and FFN, `next` (layer `l + 1`)'s
    /// Q/K/V; the post-attention and post-FFN norms are the core's `[Rc, H]` copies of `l`.
    pub fn mid_ss<'a>(&'a self, l: usize, next: &'a SlidingLayer, core: &'a CoreWeights, rope: &'a Rope) -> Result<MidSsInt8<'a>, Error> {
        let own = &self.natural_layers[l];
        let following = &self.natural_layers[l + 1];
        Ok(MidSsInt8 {
            consts: self.consts.args(),
            o: own.fp8_o.as_ref().ok_or_else(|| Self::unprepared("FP8 SS O missing"))?.args(&core.layers[l].post_attention_norm),
            q4: self.q4_args(l, core)?,
            next: following.fp8_qkv.as_ref().ok_or_else(|| Self::unprepared("FP8 SS QKV missing"))?.args(next, rope),
        })
    }

    /// The INT8 `mid_sf(l)` launch arguments: Sliding layer `l`'s O and FFN, Full `next`'s Q and K.
    pub fn mid_sf<'a>(&'a self, l: usize, next: &'a FullLayer, core: &'a CoreWeights, rope: &'a Rope) -> Result<MidSfInt8<'a>, Error> {
        let qk = self.natural_layers[l + 1].fp8_qk_full.as_ref().ok_or_else(|| Self::unprepared("RSF FP8 Full Q/K missing"))?;
        let next_f8 = qk.args(next, rope);
        let o_f8 = self.natural_layers[l].fp8_o.as_ref().ok_or_else(|| Self::unprepared("RSF FP8 Sliding O missing"))?.args(&core.layers[l].post_attention_norm);
        let ffn_f8 = self.ffn_f8_args(l, core)?;
        Ok(MidSfInt8 { consts: self.consts.args(), ffn_f8, o_f8, next_f8 })
    }

    /// The INT8 `mid_fs(l)` launch arguments: Full layer `l`'s O and FFN, Sliding `next`'s Q/K/V.
    pub fn mid_fs<'a>(&'a self, l: usize, next: &'a SlidingLayer, core: &'a CoreWeights, rope: &'a Rope) -> Result<MidFsInt8<'a>, Error> {
        let q4 = self.q4_args(l, core)?;
        Ok(MidFsInt8 { q4, consts: self.consts.args(), o: self.o_full_args(l, core)?, next: self.qkv_args(l + 1, next, rope)? })
    }

}

fn release_mlp(m: &mut MlpWeights) {
    m.up_weight_packed = HbmTensor::new();
    m.gate_weight_packed = HbmTensor::new();
    m.down_weight_packed = HbmTensor::new();
    m.up_weight_scale = HbmTensor::new();
    m.gate_weight_scale = HbmTensor::new();
    m.down_weight_scale = HbmTensor::new();
    m.up_global_scale = HbmTensor::new();
    m.gate_global_scale = HbmTensor::new();
    m.down_global_scale = HbmTensor::new();
}
