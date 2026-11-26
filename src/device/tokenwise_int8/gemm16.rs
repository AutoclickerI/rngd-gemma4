//! BF16 contraction with stationary weights: the INT8 core's structure (`gemm`) at the BF16 rate.
//!
//! Every slice holds 16 output rows in its TRF (8 lanes `N % 8` x 2 rows `N / 8 % 2` broadcast
//! in time) for a K portion of `64 F` elements (61,440 bytes at `F = 30`), and the bf16
//! activations of all tokens stream once per pass through the switch (Broadcast01 all-gather over
//! a ring of `TS` token slots x 4 K chunks of `CH = 16 F` elements). The contraction packet is
//! `[K / CH % 2, K % 16]` (32 bf16 from two chunks), so the weight's TRF order is
//! `[N / 8 % 2, K / 16 % F, K / CH2 % 2, K / CH % 2, K % 16]` with `CH2 = 2 CH`.
//!
//! A pass's f32 sums leave the contraction unrounded; a K split over passes adds them in a fixed
//! order (pass 0 + pass 1,...), the same in every bucket, so a row's bits do not depend on the
//! bucket or its position (the batch-invariance contract).
use furiosa_opt_std::prelude::*;

use super::gemm::Pi;
use crate::Chip;

/// Activations in each cluster: slice `(RING, TR, K / CH % 4)` holds that K chunk of the tokens
/// with ring position `TR`.
pub(crate) type XRing16<CL, RING, TT, TR, K, const CH: usize> =
    DmTensor<bf16, Chip, CL, m![{ RING }, { TR }, K / { CH } % 4], m![{ TT }, K % { CH }]>;
/// 16 weight rows per slice in the TRF, in the stream's K order.
pub(crate) type WTrf16<CL, SL, N, K, const F: usize, const CH: usize, const CH2: usize> =
    TrfTensor<bf16, Chip, CL, SL, m![N % 8], m![N / 8 % 2, K / 16 % { F }, K / { CH2 } % 2, K / { CH } % 2, K % 16]>;
/// A pass's f32 sums (or a summed tile) of the slice's 16 rows for all tokens.
pub(crate) type Acc<CL, SL, TT, TR, N> = DmTensor<f32, Chip, CL, SL, m![{ TT }, { TR }, N / 8 % 2, N % 8]>;
/// The same tile in bf16.
pub(crate) type Tile<CL, SL, TT, TR, N> = DmTensor<bf16, Chip, CL, SL, m![{ TT }, { TR }, N / 8 % 2, N % 8]>;

/// The slice's 16 decoded weight rows (`w` in DM, element `[N % 16, K portion]`) into the TRF.
pub(crate) fn to_trf<CL: M, SL: M, N: AxisName, K: AxisName, const F: usize, const CH: usize, const CH2: usize, WE: M>(
    device: &mut Device,
    w: &DmTensor<bf16, Chip, CL, SL, WE>,
) -> WTrf16<CL, SL, N, K, F, CH, CH2> {
    device
        .sub
        .begin(w.view())
        .fetch::<m![N % 8, N / 8 % 2, K / 16 % { F }, K / { CH2 } % 2, K / { CH } % 2], m![K % 16]>()
        .collect::<m![N % 8, N / 8 % 2, K / 16 % { F }, K / { CH2 } % 2, K / { CH } % 2], m![K % 16]>()
        .to_trf()
}

/// One pass: the f32 sums of the slice's 16 rows for all tokens.
#[allow(clippy::type_complexity)]
pub(crate) fn contract<
    CL: M,
    RING: M,
    XS: M,
    YS: M,
    TT: M,
    TR: M,
    N: AxisName,
    K: AxisName,
    const TS: usize,
    const F: usize,
    const CH: usize,
    const CH2: usize,
>(
    device: &mut Device,
    x: &XRing16<CL, RING, TT, TR, K, CH>,
    w: &WTrf16<CL, m![{ RING }, { XS }, { YS }], N, K, F, CH, CH2>,
) -> Acc<CL, m![{ RING }, { XS }, { YS }], TT, TR, N> {
    device
        .main
        .begin(x.view())
        .fetch::<m![{ TT }, K / 16 % { F }], m![K % 16]>()
        .switch::<m![{ RING }, { XS }, { YS }], m![{ TT }, { TR }, K / 16 % { F }, K / { CH } % 4]>(SwitchConfig::Broadcast01 {
            slice1: TS,
            slice0: 4,
            time0: F,
        })
        .collect::<m![{ TT }, { TR }, K / 16 % { F }, K / { CH } % 4], m![K % 16]>()
        .contract_outer::<m![{ TT }, { TR }, K / 16 % { F }, K / { CH2 } % 2, N / 8 % 2], m![K / { CH } % 2, K % 16], _, _, _>(w)
        .contract_packet::<m![1]>()
        .contract_time::<m![{ TT }, { TR }, N / 8 % 2]>()
        .contract_lane::<m![{ TT }, { TR }, N / 8 % 2], m![N % 8]>(LaneMode::Interleaved)
        .commit_trim::<m![N % 8]>()
        .commit()
}

/// `a + b` in f32 (pass order fixed by the caller). The pair is read in unzip groups of 8 tokens
/// (128 elements, the tile's tokens `[TT, TR]` taken 8 at a time, so any `TT x TR` that is a
/// multiple of 8): one-flit groups ran this pass at 3.6 cycles per flit on the device against
/// 1.1-1.2 for `ffn_fast::add_partials`'s grouped i32 pair. The same
/// element pairs and the same add: the bits do not change.
pub(crate) fn add<CL: M, SL: M, TT: M, TR: M, N: AxisName>(
    device: &mut Device,
    a: &Acc<CL, SL, TT, TR, N>,
    b: &Acc<CL, SL, TT, TR, N>,
) -> Acc<CL, SL, TT, TR, N> {
    device
        .main
        .begin_interleaved::<Pi, _, _, _, _, _>(a.view(), b.view())
        .fetch::<m![({ TT }, { TR }) / 8, Pi, ({ TT }, { TR }) % 8, N / 8 % 2], m![N % 8]>()
        .collect::<m![({ TT }, { TR }) / 8, Pi, ({ TT }, { TR }) % 8, N / 8 % 2], m![N % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Pi, m![{ TT }, { TR }, N / 8 % 2]>()
        .vector_clip_zip(ClipBinaryOpF32::Add)
        .vector_final()
        .commit_trim::<m![N % 8]>()
        .commit()
}

/// `bf16(a + b)`: the last partial sum, rounded once to bf16. Unzip
/// groups of 8 tokens, as `add`.
pub(crate) fn add_round<CL: M, SL: M, TT: M, TR: M, N: AxisName>(
    device: &mut Device,
    a: &Acc<CL, SL, TT, TR, N>,
    b: &Acc<CL, SL, TT, TR, N>,
) -> Tile<CL, SL, TT, TR, N> {
    device
        .main
        .begin_interleaved::<Pi, _, _, _, _, _>(a.view(), b.view())
        .fetch::<m![({ TT }, { TR }) / 8, Pi, ({ TT }, { TR }) % 8, N / 8 % 2], m![N % 8]>()
        .collect::<m![({ TT }, { TR }) / 8, Pi, ({ TT }, { TR }) % 8, N / 8 % 2], m![N % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Pi, m![{ TT }, { TR }, N / 8 % 2]>()
        .vector_clip_zip(ClipBinaryOpF32::Add)
        .vector_final()
        .cast::<bf16, m![N % 8 # 16]>()
        .commit_trim::<m![N % 8]>()
        .commit()
}

/// Apply the FP8 projection's per-output-row scale as `bf16(f32(y) * ws[n])`.
pub(crate) fn scale_cols<CL: M, SL: M, TT: M, TR: M, N: AxisName>(
    device: &mut Device,
    y: &Tile<CL, SL, TT, TR, N>,
    ws: &VrfTensor<f32, Chip, CL, SL, m![N / 8 % 2, N % 8]>,
) -> Tile<CL, SL, TT, TR, N> {
    device
        .main
        .begin(y.view())
        .fetch::<m![{ TT }, { TR }, N / 8 % 2], m![N % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![{ TT }, { TR }, N / 8 % 2], m![N % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{ TT }, { TR }, N / 8 % 2, N / 4 % 2], m![N % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), ws)
        .vector_widen_concat::<m![{ TT }, { TR }, N / 8 % 2], m![N % 8]>()
        .vector_final()
        .cast::<bf16, m![N % 8 # 16]>()
        .commit_trim::<m![N % 8]>()
        .commit()
}

/// `bf16(f32(y) * g)` with one scalar `g` (the NVFP4 global scale).
pub(crate) fn scale_all<CL: M, SL: M, TT: M, TR: M, N: AxisName>(
    device: &mut Device,
    y: &Tile<CL, SL, TT, TR, N>,
    g: &VrfTensor<f32, Chip, CL, SL, m![1 # 8]>,
) -> Tile<CL, SL, TT, TR, N> {
    device
        .main
        .begin(y.view())
        .fetch::<m![{ TT }, { TR }, N / 8 % 2], m![N % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![{ TT }, { TR }, N / 8 % 2], m![N % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{ TT }, { TR }, N / 8 % 2, N / 4 % 2], m![N % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), g)
        .vector_widen_concat::<m![{ TT }, { TR }, N / 8 % 2], m![N % 8]>()
        .vector_final()
        .cast::<bf16, m![N % 8 # 16]>()
        .commit_trim::<m![N % 8]>()
        .commit()
}

/// The output-row scales `ws [N % 16]` (bf16 in HBM) of the slice's 16 rows as f32 in its VRF.
pub(crate) fn col_scales<CL: M, SL: M, N: AxisName, WE: M>(
    device: &mut Device,
    ws: HbmTensorView<'_, bf16, Chip, WE>,
) -> VrfTensor<f32, Chip, CL, SL, m![N / 8 % 2, N % 8]> {
    let ws: DmTensor<bf16, Chip, CL, SL, m![N / 8 % 2, N % 8]> = ws.to_dm(&mut device.tdma);
    device
        .sub
        .begin(ws.view())
        .fetch::<m![N / 8 % 2], m![N % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![N / 8 % 2], m![N % 8]>()
        .to_vrf()
}
