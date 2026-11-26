//! Stationary-weight head primitives. Stream the vocabulary in 32 chunks of
//! 8,192 rows, holding one chunk in DM and filling the TRF for each K half.
//! Activation rows traverse the switch ring and the f32 partials are added before
//! BF16 rounding. Logits use `bf16(tanh(f32(logit) / 30) * 30)` for softcapping.

use furiosa_opt_std::prelude::*;

use crate::axes::{H, W};
use crate::device::tokenwise::common::{self, NormRows};
use crate::device::tokenwise_int8::gemm16::{Acc, Tile};
use crate::{Chip, LOGIT_SOFTCAP};

/// Vocabulary chunk `W / 8192`: the cluster holds 4,096 rows, 16 per slice.
type VCl = m![W / 4096 % 2];

/// The final RMS norm of every row of `x` into `h`.
pub(crate) fn final_norm<T: AxisName, const G: usize>(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![T, H]>,
    final_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    h: &mut HbmTensor<bf16, Chip, m![T, H]>,
) {
    let gamma = common::norm_weight::<T>(device, final_rms_weight);
    for g in 0..G {
        let xg: NormRows<T> = common::group_view::<T, m![H], G>(x, g).to_dm(&mut device.tdma);
        let n = common::normalize::<T>(device, &xg, &gamma);
        n.view().to_hbm_view(&mut device.tdma, common::group_view_mut::<T, m![H], G>(h, g));
    }
}

/// `bf16(tanh(f32(l) / 30) * 30)` of a chunk's logits.
fn softcap<SL: M, TT: M, TR: M>(device: &mut Device, l: &Tile<VCl, SL, TT, TR, W>) -> Tile<VCl, SL, TT, TR, W> {
    let scaled: Acc<VCl, SL, TT, TR, W> = device
        .main
        .begin(l.view())
        .fetch::<m![{ TT }, { TR }, W / 8 % 2], m![W % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![{ TT }, { TR }, W / 8 % 2], m![W % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{ TT }, { TR }, W / 8 % 2, W / 4 % 2], m![W % 4]>()
        .vector_fp_div(LOGIT_SOFTCAP)
        .vector_widen_concat::<m![{ TT }, { TR }, W / 8 % 2], m![W % 8]>()
        .vector_final()
        .commit_trim::<m![W % 8]>()
        .commit();
    device
        .main
        .begin(scaled.view())
        .fetch::<m![{ TT }, { TR }, W / 8 % 2], m![W % 8]>()
        .collect::<m![{ TT }, { TR }, W / 8 % 2], m![W % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{ TT }, { TR }, W / 8 % 2, W / 4 % 2], m![W % 4]>()
        .vector_fp_unary(FpUnaryOp::Tanh)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), LOGIT_SOFTCAP)
        .vector_widen_concat::<m![{ TT }, { TR }, W / 8 % 2], m![W % 8]>()
        .vector_final()
        .cast::<bf16, m![W % 8 # 16]>()
        .commit_trim::<m![W % 8]>()
        .commit()
}

/// Every chunk's per-row scales, resident in DM: slice `W / 16 % 256` holds its 16 rows of each
/// of the 32 chunks (1 KB per slice), loaded once.
pub(crate) type Scales8<RING, XS, YS> = DmTensor<bf16, Chip, VCl, m![{ RING }, { XS }, { YS }], m![W / 8192, W / 8 % 2, W % 8]>;

/// All 262,144 row scales in one DMA, issued before the prefix with the first chunk.
pub(crate) fn load_scales8<RING: M, XS: M, YS: M>(device: &mut Device, scale: &HbmTensor<bf16, Chip, m![W]>) -> Scales8<RING, XS, YS> {
    scale.to_dm(&mut device.tdma)
}

/// The softcap and the transposed store of chunk `c`'s logits tile into the `[T, W]` chunk view.
#[allow(clippy::type_complexity)]
pub(super) fn store_chunk<T: AxisName, RING: M, XS: M, YS: M, TT: M, TR: M>(
    device: &mut Device,
    dot: &Tile<VCl, m![{ RING }, { XS }, { YS }], TT, TR, W>,
    c: usize,
    out: &mut HbmTensor<bf16, Chip, m![T, W]>,
) {
    {
        let l = softcap::<m![{ RING }, { XS }, { YS }], TT, TR>(device, dot);
        // Eight-slice transposes: slice `t % 8` of each ring holds token `t`'s 128 consecutive
        // logits of the chunk (a direct 16-column store per token runs the DMA at 0.3%).
        let l: DmTensor<bf16, Chip, VCl, m![W / 128 % 32, W / 16 % 8], m![T, W % 16]> = unsafe { l.reshape() };
        let l: DmTensor<bf16, Chip, VCl, m![W / 128 % 32, T % 8], m![T / 8, W / 16 % 8, W % 16]> = device
            .main
            .begin(l.view())
            .fetch::<m![T / 8, T % 8], m![W % 16]>()
            .switch::<m![W / 128 % 32, T % 8], m![T / 8, W / 16 % 8]>(SwitchConfig::InterTranspose {
                slice1: 8,
                slice0: 1,
                time0: 1,
            })
            .collect::<m![T / 8, W / 16 % 8], m![W % 16]>()
            .commit_trim::<m![W % 16]>()
            .commit();
        l.view().to_hbm_view(
            &mut device.tdma,
            out.view_mut().tile::<m![W / 8192], 1, m![T, W / 8192 = 1 #{!} 32, W % 8192]>(c),
        );
    }
}
