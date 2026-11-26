//! Normalize audio frames and project them into text embeddings. An inter-slice
//! reduction computes each frame's mean square across eight slices. Divide by
//! `sqrt(mean_square + EPS)` in f32 and round to BF16 before the projection.
//! Accumulate the projection's K shards in f32, round once per output element
//! and scatter into the prompt residual.

use furiosa_opt_std::prelude::*;

use crate::axes::{Aa, Dummy8, H, Tok1024};
use crate::device::layout::Cluster;
use crate::{Chip, EPS};

pub(crate) mod group_index {
    macro_rules! direct {
        ($group:expr) => { $group };
    }
    // Opt requires a literal zero to index the untiled, one-group layout.
    macro_rules! zero {
        ($group:expr) => { 0usize };
    }
    pub(crate) use {direct, zero};
}
use group_index::direct as row_group_index;

axes![AudioRows = 864];
// Staged encoders use 144-row tiles. Resident encoders use two tiles per request.
const LIN_B2: usize = 144;
include!("ops_audio_rows_geometry.rs");

#[device(chip = 1)]
pub fn audio_rows(
    device: &mut Device,
    frames: &HbmTensor<bf16, Chip, m![AudioRows, Aa]>,
    weight: &HbmTensor<bf16, Chip, m![H, Aa]>,
    destination: &HbmTensor<i32, Chip, m![AudioRows]>,
    out: &mut HbmTensor<bf16, Chip, m![Tok1024, H]>,
) {
    audio_rows_impl::<Tok1024>(device, frames, weight, destination, out);
}

/// Host-side assembly of one launch's inputs: `n` real frames padded to `AudioRows`
/// with copies of the last frame, so every padded frame computes the same bits as the
/// last real frame and scattering it is a harmless rewrite.
pub struct AudioBatch {
    pub frames: Vec<bf16>,
    pub destination: Vec<i32>,
}

impl AudioBatch {
    /// `rows` yields each real frame's samples and destination byte offset.
    pub fn assemble<'a, I>(rows: I) -> Result<Self, &'static str>
    where
        I: ExactSizeIterator<Item = (&'a [bf16], i32)>,
    {
        let n = rows.len();
        if n == 0 || n > AudioRows::SIZE {
            return Err("audio rows per launch must be 1..=AudioRows");
        }
        let mut frames = Vec::with_capacity(AudioRows::SIZE * Aa::SIZE);
        let mut destination = Vec::with_capacity(AudioRows::SIZE);
        for (f, d) in rows {
            if f.len() != Aa::SIZE {
                return Err("audio frame width mismatch");
            }
            frames.extend_from_slice(f);
            destination.push(d);
        }
        for _ in n..AudioRows::SIZE {
            frames.extend_from_within((n - 1) * Aa::SIZE..n * Aa::SIZE);
            destination.push(destination[n - 1]);
        }
        Ok(Self { frames, destination })
    }
}

macro_rules! sized {
    ($index:ident; $($module:ident = $rows:literal),+ $(,)?) => {$(
        pub mod $module {
            use furiosa_opt_std::prelude::*;
            use super::{Aa, Chip, Cluster, Dummy8, EPS, H};
            use crate::axes::MediaPool;
            use super::group_index::$index as row_group_index;

            axes![AudioRows = $rows];
            const LIN_B2: usize = AudioRows::SIZE / 2;
            include!("ops_audio_rows_geometry.rs");

            #[device(chip = 1)]
            pub fn encode(
                device: &mut Device,
                frames: &HbmTensor<bf16, Chip, m![AudioRows, Aa]>,
                weight: &HbmTensor<bf16, Chip, m![H, Aa]>,
                destination: &HbmTensor<i32, Chip, m![AudioRows]>,
                out: &mut HbmTensor<bf16, Chip, m![MediaPool, H]>,
            ) {
                audio_rows_impl::<MediaPool>(device, frames, weight, destination, out);
            }
        }
    )+};
}

sized!(zero; sized32 = 32);
sized!(direct;
    sized64 = 64, sized96 = 96, sized128 = 128, sized160 = 160, sized192 = 192,
    sized224 = 224, sized256 = 256, sized288 = 288, sized320 = 320, sized352 = 352, sized384 = 384,
    sized416 = 416, sized448 = 448, sized480 = 480, sized512 = 512, sized544 = 544, sized576 = 576,
    sized608 = 608, sized640 = 640, sized672 = 672, sized704 = 704, sized736 = 736, sized768 = 768,
);
