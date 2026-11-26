//! Encode image patches and scatter their text embeddings into the prompt residual.
//! Process 32 patches per row group, with each row sharded over eight slices for
//! LayerNorm. Linear stages accumulate K shards in f32.
//!
//! The position table has two rows per position. Gather column row `2c` and vertical
//! row `2r + 1`, with each position clamped to `Pv - 1`. Add the two rows in f32
//! and round to BF16. The bias and each norm also round their output to BF16.

use furiosa_opt_std::prelude::*;

use crate::axes::{Dummy8, H, MediaPool, Mv, Ov, Pv, Rv, Tok1024};
use crate::device::layout::Cluster;
use crate::ops_audio_rows::group_index::direct as row_group_index;
use crate::{Chip, EPS};

axes![VisionRows = 288, PvRows = 2240];
// Staged encoders use 144-row tiles. Resident encoders use two tiles per request.
const LIN_B2: usize = 144;
type Pixel = bf16;
include!("ops_vision_rows_geometry.rs");

fn load_pixels(
    device: &mut Device,
    pixels: &HbmTensor<Pixel, Chip, m![VisionRows, Rv]>,
    group: usize,
) -> Group<Rv, 864> {
    load_group::<Rv, 864>(device, pixels, group)
}

#[device(chip = 1)]
#[allow(clippy::too_many_arguments)]
pub fn vision_rows(
    device: &mut Device,
    pixels: &HbmTensor<bf16, Chip, m![VisionRows, Rv]>,
    pos_table: &HbmTensor<bf16, Chip, m![PvRows, Mv]>,
    col_offsets: &HbmTensor<i32, Chip, m![VisionRows]>,
    row_offsets: &HbmTensor<i32, Chip, m![VisionRows]>,
    ln1_weight: &HbmTensor<bf16, Chip, m![Rv]>,
    ln1_bias: &HbmTensor<bf16, Chip, m![Rv]>,
    dense_weight: &HbmTensor<bf16, Chip, m![Mv, Rv]>,
    dense_bias: &HbmTensor<bf16, Chip, m![Mv]>,
    ln2_weight: &HbmTensor<bf16, Chip, m![Mv]>,
    ln2_bias: &HbmTensor<bf16, Chip, m![Mv]>,
    ln3_weight: &HbmTensor<bf16, Chip, m![Mv]>,
    ln3_bias: &HbmTensor<bf16, Chip, m![Mv]>,
    projection: &HbmTensor<bf16, Chip, m![H, Ov]>,
    destination: &HbmTensor<i32, Chip, m![VisionRows]>,
    out: &mut HbmTensor<bf16, Chip, m![Tok1024, H]>,
) {
    vision_rows_impl::<Tok1024>(
        device, pixels, pos_table, col_offsets, row_offsets, ln1_weight, ln1_bias, dense_weight,
        dense_bias, ln2_weight, ln2_bias, ln3_weight, ln3_bias, projection, destination, out,
    );
}

/// Byte offsets of a patch's two position-embedding rows in the `[PvRows, Mv]` table
/// (column row `2c`, row row `2r + 1`) with `runtime::gather_pos_embedding`'s clamps.
pub fn position_offsets(col: usize, row: usize) -> (i32, i32) {
    let row_bytes = Mv::SIZE * 2;
    let col = col.min(Pv::SIZE - 1);
    let row = row.min(Pv::SIZE - 1);
    ((2 * col * row_bytes) as i32, ((2 * row + 1) * row_bytes) as i32)
}

/// Host-side assembly of one launch's inputs: `n` real rows padded to `VisionRows` with
/// copies of the last row (same pixels, position and destination), so every padded row
/// computes the same bits as the last real row and scattering it is a harmless rewrite.
pub struct VisionBatch {
    pub pixels: Vec<bf16>,
    pub col_offsets: Vec<i32>,
    pub row_offsets: Vec<i32>,
    pub destination: Vec<i32>,
}

impl VisionBatch {
    /// `rows` yields each real row's pixels, `(col, row)` patch position and destination
    /// byte offset.
    pub fn assemble<'a, I>(rows: I) -> Result<Self, &'static str>
    where
        I: ExactSizeIterator<Item = (&'a [bf16], (usize, usize), i32)>,
    {
        let n = rows.len();
        if n == 0 || n > VisionRows::SIZE {
            return Err("vision rows per launch must be 1..=VisionRows");
        }
        let mut pixels = Vec::with_capacity(VisionRows::SIZE * Rv::SIZE);
        let mut col_offsets = Vec::with_capacity(VisionRows::SIZE);
        let mut row_offsets = Vec::with_capacity(VisionRows::SIZE);
        let mut destination = Vec::with_capacity(VisionRows::SIZE);
        for (p, (col, row), d) in rows {
            if p.len() != Rv::SIZE {
                return Err("vision row width mismatch");
            }
            pixels.extend_from_slice(p);
            let (c, r) = position_offsets(col, row);
            col_offsets.push(c);
            row_offsets.push(r);
            destination.push(d);
        }
        for _ in n..VisionRows::SIZE {
            pixels.extend_from_within((n - 1) * Rv::SIZE..n * Rv::SIZE);
            col_offsets.push(col_offsets[n - 1]);
            row_offsets.push(row_offsets[n - 1]);
            destination.push(destination[n - 1]);
        }
        Ok(Self { pixels, col_offsets, row_offsets, destination })
    }
}

macro_rules! sized {
    ($index:ident; $($module:ident),+ $(,)?) => {$(
        pub mod $module {
            use furiosa_opt_std::prelude::*;
            use super::{Chip, Cluster, Dummy8, EPS, H, MediaPool, Mv, Ov, Pv, PvRows, Rv};
            use crate::ops_audio_rows::group_index::$index as row_group_index;

            type VisionRows = crate::ops_audio_rows::$module::AudioRows;
            type Pixel = i8;
            const LIN_B2: usize = VisionRows::SIZE / 2;
            include!("ops_vision_rows_geometry.rs");

            fn load_pixels(
                device: &mut Device,
                pixels: &HbmTensor<Pixel, Chip, m![VisionRows, Rv]>,
                _group: usize,
            ) -> Group<Rv, 864> {
                let x: DmTensor<i8, Chip, Cluster, RowShards<Rv, 864>, m![Rv % 864]> = pixels
                    .view()
                    .tile::<m![VisionRows / 32], 1, m![VisionRows / 32 = 1 # {GROUPS}, VisionRows % 32, Rv]>(self::row_group_index!(_group))
                    .to_dm(&mut device.tdma);
                device.main
                    .begin(x.view())
                    .fetch::<m![Rv % 864 / 32], m![Rv % 864 % 32]>()
                    .fetch_cast::<i32>()
                    .collect::<m![Rv % 864 / 8], m![Rv % 864 % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_logic(LogicBinaryOpI32::BitAnd, 255)
                    .vector_fxp_to_fp(31)
                    .vector_narrow_split::<m![Rv % 864 / 4], m![Rv % 864 % 4]>()
                    .vector_fp_div(255.0f32)
                    .vector_widen_concat::<m![Rv % 864 / 8], m![Rv % 864 % 8]>()
                    .vector_final()
                    .cast::<bf16, m![Rv % 864 % 8 # 16]>()
                    .commit_trim::<m![Rv % 864 % 8]>()
                    .commit()
            }

            #[device(chip = 1)]
            pub fn encode_u8(
                device: &mut Device,
                pixels: &HbmTensor<i8, Chip, m![VisionRows, Rv]>,
                pos_table: &HbmTensor<bf16, Chip, m![PvRows, Mv]>,
                col_offsets: &HbmTensor<i32, Chip, m![VisionRows]>,
                row_offsets: &HbmTensor<i32, Chip, m![VisionRows]>,
                ln1_weight: &HbmTensor<bf16, Chip, m![Rv]>,
                ln1_bias: &HbmTensor<bf16, Chip, m![Rv]>,
                dense_weight: &HbmTensor<bf16, Chip, m![Mv, Rv]>,
                dense_bias: &HbmTensor<bf16, Chip, m![Mv]>,
                ln2_weight: &HbmTensor<bf16, Chip, m![Mv]>,
                ln2_bias: &HbmTensor<bf16, Chip, m![Mv]>,
                ln3_weight: &HbmTensor<bf16, Chip, m![Mv]>,
                ln3_bias: &HbmTensor<bf16, Chip, m![Mv]>,
                projection: &HbmTensor<bf16, Chip, m![H, Ov]>,
                destination: &HbmTensor<i32, Chip, m![VisionRows]>,
                out: &mut HbmTensor<bf16, Chip, m![MediaPool, H]>,
            ) {
                // Preserve allocation order in the compiled RGB8 encoder.
                let _unused_bf16: HbmTensor<bf16, Chip, m![VisionRows, Rv]> = HbmTensor::new();
                vision_rows_impl::<MediaPool>(
                    device, pixels, pos_table, col_offsets, row_offsets, ln1_weight, ln1_bias,
                    dense_weight, dense_bias, ln2_weight, ln2_bias, ln3_weight, ln3_bias,
                    projection, destination, out,
                );
            }
        }
    )+};
}

sized!(zero; sized32);
sized!(direct;
    sized64, sized96, sized128, sized160, sized192, sized224, sized256, sized288,
);
