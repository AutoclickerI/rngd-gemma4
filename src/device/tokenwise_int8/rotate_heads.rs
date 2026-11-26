//! Online per-head rotation and per-row INT8 quantization of the O input, one token per slice
//! (`T <= 512` per pass; layout of `rotate`).
//!
//! `precision_policy` rotates the O input per query head: `R = blockdiag_16(H_W) D / sqrt(W)` with
//! independent seeded signs over the whole row (`W = 256` Sliding, `512` Full). Sylvester
//! `H_W = H_B (x) H64` with `b = 64 b1 + b0` (`B = 4` or `8`), so the row is `[head, b1, b0]` and
//! the rotation runs as two contraction stages on exact bf16 operands:
//!
//! 0. the signs, `x * d` in f32 (exact), as bf16;
//! 1. H64 over `b0`, its f32 output split into two bf16 terms (`trunc16(u)` and
//!    `bf16(u - trunc16(u))`, `rotate`'s split);
//! 2. H_B over `b1` on the two terms (summed in the f32 accumulator),
//!
//! then `y = bf16(v * f32(1 / sqrt(W)))` and the per-row INT8 of `rotate` from `y`. The output
//! order is `[head, b0' / 32, b1', b0' % 32]`; the O weight's columns are prepared in that order.
use furiosa_opt_std::prelude::*;

use super::rotate::{H64Trf, Kb, Ob, Rc, TCl, TSl};
use super::rotate_ff::K16;
use crate::Chip;

axes![K8 = 8, O8 = 8];

type Row<D, T, const HALF: usize, E> = DmTensor<D, Chip, TCl<T, HALF>, TSl<T, HALF>, E>;

/// H_B in the TRF: lane `O1`, one entry read per cycle (rows of 16, zeros past `B1`).
pub(crate) type HbTrf<T, const HALF: usize, B1, O1> = TrfTensor<bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![O1], m![B1 # 16]>;

/// H_B `[O1, K16]` (zeros past `B1`) from its `Rc` copies into every slice's TRF.
pub(crate) fn hb<T: AxisName, const HALF: usize, B1: AxisName, O1: AxisName, PP: super::passes::Passes>(
    device: &mut Device,
    h: &HbmTensor<bf16, Chip, m![Rc, O1, K16]>,
) -> HbTrf<T, HALF, B1, O1> {
    let d: Row<bf16, T, HALF, m![O1, K16]> = PP::table::<T, HALF, m![O1, K16]>(device, h.view());
    let d: Row<bf16, T, HALF, m![O1, B1 # 16]> = unsafe { d.reshape() };
    device.sub.begin(d.view()).fetch::<m![O1], m![B1 # 16]>().collect::<m![O1], m![B1 # 16]>().to_trf()
}

/// The sign row `[Q]` from its `Rc` copies into every slice's DM.
pub(crate) fn signs<T: AxisName, const HALF: usize, Q: AxisName, PP: super::passes::Passes>(device: &mut Device, d: &HbmTensor<bf16, Chip, m![Rc, Q]>) -> Row<bf16, T, HALF, m![Q]> {
    PP::table::<T, HALF, m![Q]>(device, d.view())
}

/// The H64 and H_B tables of `rotate_heads_pass` in the TRF, loaded once for every token pass, the
/// H64 factor from every slice's DM in the all-gather order (`rotate2::D64`, as
/// `proj_fast::rot_heads`' v9 cores take it).
pub(crate) fn heads_tables_d64<T: AxisName, const HALF: usize, B1: AxisName, O1: AxisName, LP: super::passes::Passes>(
    device: &mut Device,
    d64: &Row<bf16, T, HALF, super::rotate2::D64>,
    hbh: &HbmTensor<bf16, Chip, m![Rc, O1, K16]>,
) -> (H64Trf<T, HALF>, HbTrf<T, HALF, B1, O1>) {
    let h64t: H64Trf<T, HALF> = device
        .sub
        .begin(d64.view())
        .fetch::<m![Ob % 8, Ob / 8, Kb / 16], m![Kb % 16]>()
        .collect::<m![Ob % 8, Ob / 8, Kb / 16], m![Kb % 16]>()
        .to_trf();
    let hbt = hb::<T, HALF, B1, O1, LP>(device, hbh);
    (h64t, hbt)
}
