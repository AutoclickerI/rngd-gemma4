//! The segments that cross a Full layer, composed on the BF16 core .
//! `mid_sf` (a Sliding layer's O and FFN, then the next Full layer's QKV) and `mid_fs` (a Full
//! layer's O and FFN, then the next Sliding layer's QKV). The segments read and write the
//! official attention's row spaces and KV pools directly, so the `ops_kv_publish` prep/finish
//! launches go away:
//!
//! 1. `a` is gathered from the family's `out` row space at `rows` (two kernel rows of eight
//!    heads per step row; `[T, Hh, Hq]` is `[T, Q]` byte for byte);
//! 2. (Sliding only) the layer's staged K/V rows move from `stage` to the owners' ring rows
//!    `publish`, in row order (a padding row rewrites its own staging row);
//! 3. O, FFN, the next layer's QKV;
//! 4. the next layer's q halves go into its `qin` at `rows`, K/V into its pools (Sliding at
//!    `stage`, Full in place at `publish`).
//!
//! Every move is a plain DMA (the same descriptors as `ops_kv_publish`), so values are unchanged
//! bit for bit.
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Ds, Ns, Rs};
use crate::ops_core::moves;

// ------------------------------------------------------------------------------ FP8 (E4M3) Sliding KV

/// `publish_sliding` on E4M3 Sliding pools: a byte copy of the staged rows (`ops_core::moves::publish_kv8`).
pub(crate) fn publish_sliding_kv8<T: AxisName, SS: M, SE: M>(
    device: &mut Device,
    stage: &HbmTensor<i32, Chip, m![T]>,
    publish: &HbmTensor<i32, Chip, m![T]>,
    k_pool: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
    v_pool: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
) {
    moves::publish_kv8::<T, SS, SE>(device, k_pool, stage, publish);
    moves::publish_kv8::<T, SS, SE>(device, v_pool, stage, publish);
}
