//! Step glue of the token-packed engine (`host::engine`) around the core segments (`ops_core`):
//!
//! - `embed_rows_tN`: the step's embedded input `x0 [TokN, H]`: each row is gathered
//!   from the load-time BF16-pre-scaled table. Media rows are overwritten afterwards
//!   with their encoder outputs (`host::media`, `ops_media::put_rows`).
//! - `widen_tN`: the step's residual `x [TokN, H]` scattered into the first rows of
//!   `[Tok1024, H]` for steps in a smaller bucket. The core's `last` (`ops_core::ends`) reads
//!   the output rows of `x` there and gathers attention directly from the Full row space.
//!   Plain DMA. (A tiled view of the Tok1024 tensor cannot take the TokN rows: the axes differ, and the CPU
//!   interpreter rejects the store with `carve:... Unconsumed([Tok8 # 16])`.)
use furiosa_opt_std::prelude::*;

use crate::axes::*;
use crate::device::layout::Cluster;
use crate::Chip;
use crate::ops_media::PutRows;

#[cfg(feature = "compile-carried")]
pub mod wide_output;

macro_rules! embed_rows {
    ($name:ident, $tok:ident, $slice:ty, $elem:ty, [$($outer:tt)*]) => {
        /// The step's embedded input: rows `index` (byte offsets) of the load-time
        /// BF16-pre-scaled table.
        #[device(chip = 1)]
        pub fn $name(
            device: &mut Device,
            embedding_table: &HbmTensor<bf16, Chip, m![W, H]>,
            index: &HbmTensor<i32, Chip, m![$tok]>,
            x: &mut HbmTensor<bf16, Chip, m![$tok, H]>,
        ) {
            let rows: DmTensor<bf16, Chip, Cluster, $slice, $elem> = embedding_table.gather::<m![W], m![H]>().by_byte_offsets(index).to_dm(&mut device.tdma);
            rows.view().to_hbm_view(&mut device.tdma, x.view_mut());
        }
    };
}
macro_rules! embed_media_rows {
    ($name:ident, $tok:ident, $slice:ty, $elem:ty, [$($outer:tt)*]) => {
        #[device(chip = 1)]
        pub fn $name(
            device: &mut Device,
            embedding_table: &HbmTensor<bf16, Chip, m![W, H]>,
            index: &HbmTensor<i32, Chip, m![$tok]>,
            media: &HbmTensor<bf16, Chip, m![MediaPool, H]>,
            source: &HbmTensor<i32, Chip, m![PutRows]>,
            destination: &HbmTensor<i32, Chip, m![PutRows]>,
            x: &mut HbmTensor<bf16, Chip, m![$tok, H]>,
        ) {
            let rows: DmTensor<bf16, Chip, Cluster, $slice, $elem> = embedding_table.gather::<m![W], m![H]>().by_byte_offsets(index).to_dm(&mut device.tdma);
            rows.view().to_hbm_view(&mut device.tdma, x.view_mut());
            let soft: DmTensor<bf16, Chip, Cluster, m![PutRows % 256], m![PutRows / 256, H]> =
                media.gather::<m![MediaPool], m![H]>().by_byte_offsets(source).to_dm(&mut device.tdma);
            x.view_mut().scatter::<m![$tok], m![H]>().by_byte_offsets(destination.view()).from_dm_view(&mut device.tdma, soft.view());
        }
    };
}

embed_media_rows!(embed_media_t1024, Tok1024, m![Tok1024 % 128, 1 # 2], m![Tok1024 / 128, H], [Tok1024 / 128,]);
#[cfg(feature = "compile-carried")]
embed_media_rows!(embed_media_t1536, Tok1536, m![Tok1536 % 128, 1 # 2], m![Tok1536 / 128, H], [Tok1536 / 128,]);

embed_rows!(embed_rows_t128, Tok128, m![Tok128, 1 # 2], m![H], []);
embed_rows!(embed_rows_t1024, Tok1024, m![Tok1024 % 128, 1 # 2], m![Tok1024 / 128, H], [Tok1024 / 128,]);
#[cfg(feature = "compile-carried")]
embed_rows!(embed_rows_t1536, Tok1536, m![Tok1536 % 128, 1 # 2], m![Tok1536 / 128, H], [Tok1536 / 128,]);

macro_rules! widen {
    ($name:ident, $tok:ident, $slice:ty, $elem_x:ty) => {
        /// The step's residual into rows `0..N` of the Tok1024 tensor `last` reads.
        /// `index_x` holds row `i`'s byte offset there, `i x H x 2`.
        /// Attention stays in the Full row space consumed directly by `last`.
        #[device(chip = 1)]
        pub fn $name(
            device: &mut Device,
            x: &HbmTensor<bf16, Chip, m![$tok, H]>,
            index_x: &HbmTensor<i32, Chip, m![$tok]>,
            wide_x: &mut HbmTensor<bf16, Chip, m![Tok1024, H]>,
        ) {
            let rows: DmTensor<bf16, Chip, Cluster, $slice, $elem_x> = x.to_dm(&mut device.tdma);
            wide_x.view_mut().scatter::<m![Tok1024], m![H]>().by_byte_offsets(index_x.view()).from_dm_view(&mut device.tdma, rows.view());
        }
    };
}

widen!(widen_t128, Tok128, m![Tok128, 1 # 2], m![H]);
