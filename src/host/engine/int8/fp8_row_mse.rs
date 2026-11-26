//! Per-row scale adjustments for FP8 weight preparation.
use super::*;

const BLOB: &[u8] = include_bytes!("fp8_row_mse.lz4");
const ROWS: usize = 47 * (2 * 15360 + 3840);
const PER_LAYER: usize = 2 * 15360 + 3840;

pub(super) struct RowMse { indices: Vec<u8> }
pub(super) struct Ratios<'a> { pub gate: &'a [u8], pub up: &'a [u8], pub down: &'a [u8] }

impl RowMse {
    pub(super) fn load() -> Result<Self, BoxError> {
        let meta: serde_json::Value = serde_json::from_str(include_str!("fp8_row_mse.json"))?;
        let sites = meta["sites"].as_array().ok_or("FP8 row-MSE site metadata missing")?;
        if sites.len() != 141 { return Err("FP8 row-MSE must bind 141 projections".into()); }
        let mut offset = 0;
        for layer in 0..47 {
            for (site, (name, rows)) in [("gate_proj",15360), ("up_proj",15360), ("down_proj",3840)].iter().enumerate() {
                let entry = &sites[3 * layer + site];
                if entry["label"] != format!("L{layer}.{name}") || entry["offset"] != offset || entry["rows"] != *rows {
                    return Err("FP8 row-MSE site order/offset mismatch".into());
                }
                offset += rows;
            }
        }
        let mut indices = vec![0; ROWS];
        crate::host::envelope_lookup::unpack_into(BLOB, &mut indices);
        let digest = super::q4cal::sha256::hex(&indices);
        if meta["raw_sha256"].as_str() != Some(digest.as_str()) {
            return Err("FP8 row-MSE decoded SHA-256 mismatch".into());
        }
        if indices.iter().any(|&v| v > 40) { return Err("FP8 row-MSE index outside screened ratio grid".into()); }
        Ok(Self { indices })
    }

    pub(super) fn layer(&self, layer: usize) -> Result<Ratios<'_>, BoxError> {
        if layer >= 47 { return Err("FP8 row-MSE layer outside 0..46".into()); }
        let r = &self.indices[layer * PER_LAYER..(layer + 1) * PER_LAYER];
        Ok(Ratios { gate: &r[..15360], up: &r[15360..30720], down: &r[30720..] })
    }
}
