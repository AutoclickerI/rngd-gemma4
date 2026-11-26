//! Per-head Sliding O rotation. Load H64 and H_B from HBM once per stage and
//! reuse the DM copies across token passes.

use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Hh, Hqs, Qs, Rq};
use crate::device::tokenwise_int8::rotate::{TCl, TSl};

type Row<D, T, const HALF: usize, E> = DmTensor<D, Chip, TCl<T, HALF>, TSl<T, HALF>, E>;

/// The O input's rotation with its rows gathered from the attention row space, over all passes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gather_pass_rows<T: AxisName, const HALF: usize, const PS: usize, const P: usize, const PI: usize>(
    device: &mut Device,
    out: &HbmTensor<bf16, Chip, m![Rq, Hqs]>,
    rows: &HbmTensor<i32, Chip, m![T, Hh]>,
) -> Row<bf16, T, HALF, m![Qs]> {
    let mut prow = HbmTensor::<i32, Chip, m![T % { PS }, Hh]>::new();
    rows.view()
        .tile::<m![T / { PS }], 1, m![T / { PS } = 1 # { P }, T % { PS }, Hh]>(PI)
        .to_hbm_view(&mut device.tdma, prow.view_mut());
    let ad: Row<bf16, T, HALF, m![Hh, Hqs]> = out.gather::<m![Rq], m![Hqs]>().by_byte_offsets(&prow).to_dm(&mut device.tdma);
    unsafe { ad.reshape() }
}
