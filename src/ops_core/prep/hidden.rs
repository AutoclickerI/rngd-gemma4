use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::H;
use crate::device::tokenwise_int8::rotate::{TCl, TSl};

pub(crate) type Row<D, T, const HALF: usize, E> = DmTensor<D, Chip, TCl<T, HALF>, TSl<T, HALF>, E>;

axes![Hw = 512];

/// Rows of `H` (3,840 = 15 x 256, which no 512-row pass tiles) in passes of two 256-row blocks
/// `b, b + 1`, one per cluster, typed by the pass-local `Hw` (256 rows per cluster, every slice
/// live). The DMAs go through this form, whose cluster is the two-block window of `H / 256`.
pub(crate) type Blk<D, E> = DmTensor<D, Chip, m![H / 256 = 2], m![H % 256], E>;

/// `H` blocks `b` and `b + 1` of `[H, E]` into clusters 0 and 1.
pub(crate) fn load_blocks<D: MaterializableScalar, E: M>(device: &mut Device, src: &HbmTensor<D, Chip, m![H, { E }]>, b: usize) -> Row<D, Hw, 256, E> {
    let x: Blk<D, E> = src.view().tile::<m![H / 256], 2, m![H / 256 = 2 # 15, H % 256, { E }]>(b).to_dm(&mut device.tdma);
    unsafe { x.reshape() }
}

/// Per-row f32 values compacted into slice 0 of each cluster (`rotate::store_scales`' form) to
/// `H` blocks `b` and `b + 1` of `[H]`.
pub(crate) fn store_scale_blocks(device: &mut Device, w: DmTensor<f32, Chip, m![H / 256 = 2], m![1 # 256], m![H % 256]>, ws: &mut HbmTensor<f32, Chip, m![H]>, b: usize) {
    w.view().to_hbm_view(&mut device.tdma, ws.view_mut().tile::<m![H / 256], 2, m![H / 256 = 2 #{!} 15, H % 256]>(b));
}
