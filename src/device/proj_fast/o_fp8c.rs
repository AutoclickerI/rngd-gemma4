//! Sliding O with E4M3 activations and weights.
//! Quantize each activation row using `s = max(amax / 256, 1e-30)` and round to E4M3.
//! The two K halves contract in f32. Their epilogue computes `bf16(((p0 + p1) * sx) * ws)`.

use furiosa_opt_std::prelude::*;

use super::*;
use crate::device::tokenwise_int8::ffn_fast::fp8c;
use crate::device::tokenwise_int8::rotate::TSl;

axes![Pf8 = 2];

/// The weight half `i` (`Qs / 2048 = i`) as E4M3 rows in DM (`wo_half0`'s layout).
fn wo_half8<T: AxisName, const HALF: usize, RING: M, XS: M, YS: M>(
    device: &mut Device,
    wo: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    i: usize,
) -> DmTensor<f8e4m3, Chip, TCl<T, HALF>, m![{ RING }, { XS }, { YS }], m![H % 16, Qs % 2048]> {
    wo.view().tile::<m![Qs / 2048], 1, m![H, Qs / 2048 = 1 # 2, Qs % 2048]>(i).to_dm(&mut device.tdma)
}

fn quantize_static_pass8<T: AxisName, const HALF: usize, const PS: usize, const P: usize, const PI: usize>(
    device: &mut Device,
    sigma: &HbmTensor<f32, Chip, m![1 # 8]>,
    ad: &DmTensor<bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![Qs]>,
    onehot: &VrfTensor<f32, Chip, TCl<T, HALF>, m![1 # 256], m![Sgi, Sl8]>,
    codes: &mut HbmTensor<f8e4m3, Chip, m![T, Qs]>,
    scales: &mut HbmTensor<f32, Chip, m![T]>,
) {
    let (s, q) = rotate2::row_quantize_static_f8::<T, HALF, Qs>(device, sigma, ad);
    q.view().to_hbm_view(
        &mut device.tdma,
        codes.view_mut().tile::<m![T / { PS }], 1, m![T / { PS } = 1 #{!} { P }, T % { PS }, Qs]>(PI),
    );
    rotate::store_scales::<T, HALF, m![T / { PS } = 1 #{!} { P }, T % { PS }]>(
        device,
        s,
        onehot,
        scales.view_mut().tile::<m![T / { PS }], 1, m![T / { PS } = 1 #{!} { P }, T % { PS }]>(PI),
    );
}

#[path = "o_pair.rs"]
pub(crate) mod pair;
