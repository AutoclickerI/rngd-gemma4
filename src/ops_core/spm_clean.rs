//! The engine's first launch cleans the PE data cache over shared fields.
//!
//! The PE firmware's Initialize zeroes its shared fields (SPM 0xEB000..0x100000) with CPU stores. The MMU maps only
//! the DMA descriptor arena (0xEC000..0xFC000) non-cacheable, so the lead PE's D-cache keeps dirty zero lines over the
//! rest. That includes the gather index lists `dma_indirect_indices`, which DMA writes before each gather. A dirty line
//! evicted after an index load zeroes 16 offsets of that gather. The carried image of [`spm_clean`],
//! built by `tools/native/fopt/spm_clean.py`, runs `dc civac` over those
//! cacheable lines at the start of each cluster's task, before any DMA of the process has written SPM. Nothing makes
//! them dirty again: task code and libpe never store to the index lists.
//! The Rust body supplies a 16 KiB copy for the image's buffer layout.
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Dummy2, Dummy8, Dummy256};

/// The launch's operand: one 32-byte row per slice of both clusters.
pub type Rows = m![Dummy2, Dummy256, Dummy8];

/// `out = x`; the carried image first cleans the shared fields' cache lines (module docs).
#[device(chip = 1)]
pub fn spm_clean(device: &mut Device, x: &HbmTensor<f32, Chip, Rows>, out: &mut HbmTensor<f32, Chip, Rows>) {
    let d: DmTensor<f32, Chip, m![Dummy2], m![Dummy256], m![Dummy8]> = x.to_dm(&mut device.tdma);
    d.view().to_hbm_view(&mut device.tdma, out.view_mut());
}
