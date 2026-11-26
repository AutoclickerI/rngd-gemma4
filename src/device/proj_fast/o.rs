//! Rotated INT8 Sliding O. Each cluster contracts the token rows it produced against
//! the full weight matrix. Each slice holds 16 rows of 4,096 codes in its TRF.
//! Token ownership keeps the exact i32 sums local to their cluster.

use furiosa_opt_std::prelude::*;
use super::rot_heads::gather_pass_rows;

use crate::Chip;
use crate::axes::{H, Hh, Hqs, Qs, Rq};
use crate::device::tokenwise_int8::gemm::{self};
use crate::device::tokenwise_int8::rotate::{self, Kh, Rc, Sgi, Sl8, TCl};
use crate::device::tokenwise_int8::rotate2;

// The O input columns `Qs` in two halves of `Ko = 2,048` (256-byte aligned weight runs).
axes![Ko2 = 2, Ko = 2048];

/// Load the scales for every token row owned by this cluster into each slice's VRF.
fn row_scales<T: AxisName, const HALF: usize, const PS: usize, SL: M>(
    device: &mut Device,
    sx: &HbmTensor<f32, Chip, m![T]>,
) -> VrfTensor<f32, Chip, TCl<T, HALF>, SL, m![T / { PS }, T % { HALF }]> {
    let s: DmTensor<f32, Chip, TCl<T, HALF>, SL, m![T / { PS }, T % { HALF }]> = sx.to_dm(&mut device.tdma);
    device
        .sub
        .begin(s.view())
        .fetch::<m![T / { PS }, T % { HALF } / 8], m![T % 8]>()
        .collect::<m![T / { PS }, T % { HALF } / 8], m![T % 8]>()
        .to_vrf()
}

#[path = "o_fp8c.rs"]
pub(crate) mod fp8c;

// The same K contractions, with token-tile count expressed for Tok128.
#[path = "o_fp8c_drain.rs"]
pub(crate) mod fp8c_drain;
