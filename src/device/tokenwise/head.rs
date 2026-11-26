//! Normalize each emitting row, contract all 3,840 hidden columns into each vocabulary
//! logit, and apply the softcap. Stream the vocabulary in bands of 16,384 rows with
//! 32 rows per slice while eight output tokens share the TRF lanes.

use furiosa_opt_std::prelude::*;

use crate::Chip;

/// Rows `rows` (indices into the step) of a step tensor `[S, D]`, gathered into `[T, D]`.
pub(crate) fn gather_rows<S: AxisName, D: M, T: AxisName>(
    device: &mut Device,
    step: &HbmTensor<bf16, Chip, m![S, { D }]>,
    rows: &HbmTensor<i32, Chip, m![T]>,
    out: &mut HbmTensor<bf16, Chip, m![T, { D }]>,
) {
    // Two indices per slice (the index list's in-slice extent must be a whole 8 bytes), staged into PE 0's
    // scratchpad, where the indirect DMA reads a position index (Opt 0.9.0; earlier Opt read it from DM).
    let idx: DmTensor<i32, Chip, m![1 # 2], m![T / 2 # 256], m![T % 2]> = rows.to_dm(&mut device.tdma);
    let idx: SpmTensor<i32, Chip, m![1 # 2], m![1 # 4], m![T]> = idx.to_spm(&mut device.tdma);
    let r: DmTensor<bf16, Chip, m![1 # 2], m![T / 2 # 256], m![T % 2, { D }]> =
        step.gather::<m![S], m![{ D }]>().by_positions::<m![T]>(&idx).to_dm(&mut device.tdma);
    r.view().to_hbm_view(&mut device.tdma, out.view_mut());
}
