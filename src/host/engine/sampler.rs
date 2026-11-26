//! Output buffers and cached ban masks for the device sampler.
use crate::Chip;
use crate::axes::*;
use furiosa_opt_std::prelude::*;
use std::collections::HashSet;

pub(super) struct OutputRows {
    pub(super) exact: super::pipeline::Exact,
    pub logits: HbmTensor<bf16, Chip, m![Out128, Nseg, Lseg]>,
    pub rows: HbmTensor<i32, Chip, m![Out128]>,
    /// Byte offsets of the two attention output halves per row.
    pub a_rows: HbmTensor<i32, Chip, m![Out128, Hh]>,
    pub masks: Vec<(Vec<usize>, HbmTensor<i32, Chip, m![Nseg, Lseg]>)>,
}

impl OutputRows {
    pub async fn new(device: &mut Device) -> Result<Self, Error> {
        let logits = HostTensor::<bf16, m![Out128, Nseg, Lseg]>::zero().to_hbm(&mut device.pdma).await?;
        Ok(Self {
            exact: super::pipeline::Exact::new(device, &logits).await?,
            logits,
            rows: HostTensor::<i32, m![Out128]>::zero().to_hbm(&mut device.pdma).await?,
            a_rows: HostTensor::<i32, m![Out128, Hh]>::zero().to_hbm(&mut device.pdma).await?,
            masks: Vec::new(),
        })
    }

    /// The index in `masks` of `banned`'s device mask, uploaded on first use (the server sees two
    /// sets: the suppressed ids, and those with EOS and the channel ids under `ignore_eos`).
    pub async fn mask(&mut self, device: &mut Device, banned: &HashSet<usize>) -> Result<usize, Error> {
        let mut set: Vec<usize> = banned.iter().copied().collect();
        set.sort_unstable();
        if let Some(i) = self.masks.iter().position(|(s, _)| *s == set) {
            return Ok(i);
        }
        if self.masks.len() == 4 {
            self.masks.remove(0);
        }
        let entries = (0..Nseg::SIZE * Lseg::SIZE)
            .map(|id| if banned.contains(&id) { i32::MAX } else { (id % Lseg::SIZE) as i32 });
        let mask = HostTensor::<i32, m![Nseg, Lseg]>::from_vec(entries).to_hbm(&mut device.pdma).await?;
        self.masks.push((set, mask));
        Ok(self.masks.len() - 1)
    }
}
