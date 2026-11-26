//! Startup calibration for Q4 Gate/Up. The artifact stores power-of-two smoothing
//! and per-row scale ratios for the Q4 layer set.
use crate::host::load::{checked_tensor, reciprocal_global_scale};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use serde::Deserialize;
use std::{collections::BTreeMap, sync::OnceLock};
#[path = "q4cal_sha.rs"]
pub(crate) mod sha256;
const H: usize = 3840;
const L: usize = 15360;
const DATA: &[u8] = include_bytes!("q4cal_artifacts/startup.bin.lzma");
const INDEX: &str = include_str!("q4cal_artifacts/startup.json");
const STORAGE: &str = include_str!("q4cal_artifacts/storage.json");

#[derive(Deserialize)]
struct Storage { schema: usize, file: String, encoding: String, sha256: String,
    bytes: usize, raw_sha256: String, raw_bytes: usize }
#[derive(Deserialize)]
struct Tensor { offset: usize, bytes: usize, shape: Vec<usize>, dtype: String }
#[derive(Deserialize)]
struct Index { schema: usize, endianness: String, tensors: BTreeMap<String, Tensor>, raw_sha256: String }
struct Artifact { raw: Vec<u8>, index: Index }

fn artifact() -> Result<&'static Artifact, String> {
    static VALUE: OnceLock<Result<Artifact, String>> = OnceLock::new();
    VALUE.get_or_init(|| {
        let s: Storage = serde_json::from_str(STORAGE).map_err(|e| e.to_string())?;
        let index: Index = serde_json::from_str(INDEX).map_err(|e| e.to_string())?;
        if s.schema != 1 || index.schema != 1 || s.file != "startup.bin.lzma"
            || s.encoding != "lzma-alone" || index.endianness != "little"
            || s.bytes != DATA.len() || s.raw_bytes != super::q4serve::LAYERS.len() * (H + 4 * L)
            || sha256::hex(DATA) != s.sha256 || index.raw_sha256 != s.raw_sha256 {
            return Err("q4cal: storage/index identity mismatch".into());
        }
        let raw = crate::host::registry::lzma::decode_alone(DATA, s.raw_bytes)
            .map_err(|e| format!("q4cal: LZMA: {e}"))?;
        if raw.len() != s.raw_bytes || sha256::hex(&raw) != s.raw_sha256 {
            return Err("q4cal: decompressed SHA/length mismatch".into());
        }
        let a = Artifact { raw, index };
        let mut spans = Vec::new();
        for &l in super::q4serve::LAYERS {
            let mut expected = vec![(format!("L{l}.smooth_log2"), vec![H], "torch.int8", 1)];
            for site in ["up_proj", "gate_proj"] {
                expected.push((format!("L{l}.{site}.ratio"), vec![L, 1], "torch.float16", 2));
            }
            for (key, shape, dtype, width) in expected {
                let t = a.index.tensors.get(&key).ok_or_else(|| format!("q4stack missing {key}"))?;
                if t.shape != shape || t.dtype != dtype || t.bytes != shape.iter().product::<usize>() * width
                    || t.offset.checked_add(t.bytes).is_none_or(|end| end > a.raw.len()) {
                    return Err(format!("q4stack malformed {key}"));
                }
                spans.push((t.offset, t.offset + t.bytes));
            }
            for site in ["up_proj", "gate_proj"] {
                if a.ratios(l, site)?.iter().any(|s| !s.is_finite() || *s <= 0.) {
                    return Err(format!("q4stack invalid ratio L{l}.{site}"));
                }
            }
            if a.exponents(l)?.iter().any(|&e| !(-16..=16).contains(&e)) {
                return Err(format!("q4stack invalid smoothing L{l}"));
            }
        }
        spans.sort_unstable();
        if spans.len() != a.index.tensors.len() || spans[0].0 != 0
            || spans.last().unwrap().1 != a.raw.len() || spans.windows(2).any(|x| x[0].1 != x[1].0) {
            return Err("q4cal unaccounted/overlapping payload".into());
        }
        Ok(a)
    }).as_ref().map_err(Clone::clone)
}
impl Artifact {
    fn bytes(&self, key: &str) -> Result<&[u8], String> {
        let t = self.index.tensors.get(key).ok_or_else(|| format!("q4stack missing {key}"))?;
        self.raw.get(t.offset..t.offset+t.bytes).ok_or_else(|| "q4stack bounds".into())
    }
    fn ratios(&self, layer: usize, site: &str) -> Result<Vec<f32>, String> {
        Ok(self.bytes(&format!("L{layer}.{site}.ratio"))?.chunks_exact(2).map(|x| {
            let bits = u16::from_le_bytes(x.try_into().unwrap());
            let e = (bits >> 10) & 31; let m = bits & 1023;
            let value = if e == 0 { f32::from(m) * 2f32.powi(-24) }
                else if e == 31 { f32::NAN }
                else { (1. + f32::from(m)/1024.) * 2f32.powi(i32::from(e)-15) };
            if bits & 32768 == 0 { value } else { -value }
        }).collect())
    }
    fn exponents(&self, layer: usize) -> Result<Vec<i8>, String> {
        Ok(self.bytes(&format!("L{layer}.smooth_log2"))?.iter().map(|&v| v as i8).collect())
    }
}

/// Fold only the Q4 gamma. The separately prepared FP8 gamma keeps its original values.
pub(super) fn fold_gamma(layer: usize, gamma: &[super::bf16]) -> Result<Vec<super::bf16>, String> {
    if gamma.len() != H { return Err("q4stack gamma shape".into()); }
    let exponents = artifact()?.exponents(layer)?;
    Ok(gamma.iter().zip(exponents).map(|(g,e)|
        super::bf16::from_f32(g.to_f32() * 2f32.powi(-i32::from(e)))).collect())
}

/// Odd grid with ties assigned to the upper level.
pub(super) fn code(w: f32, scale: f32) -> i8 {
    (2 * ((w / (2. * scale)).floor().clamp(-8., 7.) as i32) + 1) as i8
}

/// Apply H60 first, then the H64 butterfly, with final normalization.
pub(super) fn rotate(x: &[f32; H], h60: &[f32], out: &mut [f32; H]) {
    for r in 0..60 {
        let dst = &mut out[r*64..(r+1)*64]; dst.fill(0.);
        for k in 0..60 {
            let sign = h60[r*60+k];
            for b in 0..64 { dst[b] += sign * x[k*64+b]; }
        }
        super::fwht64(dst);
        for v in dst { *v *= 0.016137430609197570_f64 as f32; }
    }
}

pub(super) struct Projection<'a> {
    packed: TensorView<'a>,
    blocks: TensorView<'a>,
    pub(super) global: f32,
    pub(super) ratios: Vec<f32>,
}

const E2M1: [f32; 16] = [0., 0.5, 1., 1.5, 2., 3., 4., 6., -0., -0.5, -1., -1.5, -2., -3., -4., -6.];

impl<'a> Projection<'a> {
    pub(super) fn load(tensors: &SafeTensors<'a>, layer: usize, site: &str) -> Result<Self, String> {
        let prefix = format!("model.language_model.layers.{layer}.mlp.{site}");
        let tensor = |field, dtype, shape: &[usize]| {
            checked_tensor(tensors, &format!("{prefix}.{field}"), dtype, shape).map_err(|e| e.to_string())
        };
        let packed = tensor("weight_packed", Dtype::U8, &[L, H / 2])?;
        let blocks = tensor("weight_scale", Dtype::F8_E4M3, &[L, H / 16])?;
        if blocks.data().iter().any(|&v| v & 127 == 127) {
            return Err(format!("nonfinite FP8 block scale: {prefix}"));
        }
        let global = reciprocal_global_scale(tensors, &format!("{prefix}.weight_global_scale"))
            .map_err(|e| e.to_string())?;
        if !global.is_finite() || global <= 0. {
            return Err(format!("invalid Q4 global scale: {prefix}"));
        }
        Ok(Self { packed, blocks, global, ratios: artifact()?.ratios(layer, site)? })
    }

    pub(super) fn expand(&self, row: usize, lut: &[f32; 256], smooth: &[f32], signs: &[f32], out: &mut [f32; H]) {
        let packed = &self.packed.data()[row * (H / 2)..][..H / 2];
        let scales = &self.blocks.data()[row * (H / 16)..][..H / 16];
        for k in 0..H {
            let byte = packed[k / 2];
            let code = if k % 2 == 0 { byte & 15 } else { byte >> 4 };
            out[k] = E2M1[usize::from(code)] * lut[usize::from(scales[k / 16])] * smooth[k] * signs[k];
        }
    }
}

pub(super) fn smooth(layer: usize) -> Result<Vec<f32>, String> {
    Ok(artifact()?.exponents(layer)?.into_iter().map(|e| 2f32.powi(i32::from(e))).collect())
}
