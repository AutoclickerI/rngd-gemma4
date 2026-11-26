//! INT8 contraction with stationary weights: every slice holds 16 output rows of the weight in
//! its TRF (8 lanes `N % 8` x 2 rows `N / 8 % 2` broadcast in time, so each streamed 64-byte activation
//! packet feeds 1,024 MACs), and the activation codes of all tokens stream once per pass through
//! the switch (Broadcast01 all-gather over a ring of `TS` token slots x 4 K chunks) into every
//! slice's contraction engine. The i32 accumulator is exact, so the streamed K order may be any
//! permutation as long as the weight's TRF layout uses the same one, and K may be split across
//! passes or clusters with the partial sums added in i32.
//!
//! Layout parameters (mapping types, chosen per bucket by the caller):
//! - `CL`: the cluster axis (an output-row digit for row-split GEMMs, a K digit for K-split ones);
//!   `RING`, `XS`, `YS`: the slice axes, `RING` indexing the `256 / (TS * 4)` independent rings and
//!   `XS` (`TS` slices), `YS` (4 slices) the positions inside a ring;
//! - `TT`, `TR`: the token axis split as `[T / TS, T % TS]` (`TR` the ring position of a token);
//! - `K`, `F`, `CH = 32 F`, `CH2 = 64 F`: the cluster's K portion is `4 CH` bytes of axis `K`,
//!   chunk `K / CH % 4` on ring position `YS`; the stream order is `[K / 32 % F, K / CH % 4, K % 32]`.
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::Dummy256;

/// Activation codes in each cluster: slice `(RING, TR, K / CH % 4)` holds that K chunk of the
/// tokens with ring position `TR`.
pub(crate) type XRing<CL, RING, TT, TR, K, const CH: usize> =
    DmTensor<i8, Chip, CL, m![{ RING }, { TR }, K / { CH } % 4], m![{ TT }, K % { CH }]>;
/// 16 weight rows per slice in the TRF, in the stream's K order.
pub(crate) type WTrf<CL, SL, N, K, const F: usize, const CH: usize, const CH2: usize> =
    TrfTensor<i8, Chip, CL, SL, m![N % 8], m![N / 8 % 2, K / 32 % { F }, K / { CH2 } % 2, K / { CH } % 2, K % 32]>;

/// The codes of `T` tokens in every ring (`x` already the cluster's K portion; absent axes of
/// `CL`/`RING` replicate).
pub(crate) fn load_codes<CL: M, RING: M, TT: M, TR: M, K: AxisName, const CH: usize, XE: M>(
    device: &mut Device,
    x: HbmTensorView<'_, i8, Chip, XE>,
) -> XRing<CL, RING, TT, TR, K, CH> {
    x.to_dm(&mut device.tdma)
}

/// The slice's 16 weight rows (`w` in DM, element `[N % 16, K portion]`) into the TRF.
pub(crate) fn to_trf<CL: M, SL: M, N: AxisName, K: AxisName, const F: usize, const CH: usize, const CH2: usize, WE: M>(
    device: &mut Device,
    w: &DmTensor<i8, Chip, CL, SL, WE>,
) -> WTrf<CL, SL, N, K, F, CH, CH2> {
    device
        .sub
        .begin(w.view())
        .fetch::<m![N % 8, N / 8 % 2, K / 32 % { F }, K / { CH2 } % 2, K / { CH } % 2], m![K % 32]>()
        .collect::<m![N % 8, N / 8 % 2, K / 32 % { F }, K / { CH2 } % 2, K / { CH } % 2], m![K % 32]>()
        .to_trf()
}

/// `row_scales` for small buckets (`T % 8 == 0`, `T <= 128`): one read into slice 0 of each
/// cluster, then the switch broadcasts the `T` scales to the cluster's 256 slices.
/// Broadcasting avoids a padded DMA read for every slice.
pub(crate) fn row_scales_bcast<T: AxisName, CL: M, SL: M>(device: &mut Device, sx: &HbmTensor<f32, Chip, m![T]>) -> VrfTensor<f32, Chip, CL, SL, m![T]> {
    let one: DmTensor<f32, Chip, CL, m![1 # 256], m![T]> = sx.to_dm(&mut device.tdma);
    let rep: DmTensor<f32, Chip, CL, m![Dummy256], m![T]> = device
        .main
        .begin(one.view())
        .fetch::<m![T / 8], m![T % 8]>()
        .switch::<m![Dummy256], m![T / 8]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![T / 8], m![T % 8]>()
        .commit_trim::<m![T % 8]>()
        .commit();
    let rep: DmTensor<f32, Chip, CL, SL, m![T]> = unsafe { rep.reshape() };
    device.sub.begin(rep.view()).fetch::<m![T / 8], m![T % 8]>().collect::<m![T / 8], m![T % 8]>().to_vrf()
}

/// `to_trf` issued from the Main context: the same TRF tensor, the same bits. The static model
/// prices a 61 KiB fill at ~2.2k cycles from Main against ~7.9k from Sub, and in the static
/// schedule the Sub fill held the next explicit sync (and with it the DMA queue). For `mid_ss_int8_t8`, every
/// FFN fill from Main 662.2k device cycles against 692.8k, every QKV/O/FFN fill 646.3k; at Tok128
/// (Main busy with the contractions) neutral, 997.6k against 999.4k. The Tok512/Tok1024 segments
/// fill the O, Down and QKV weights from Main: -11.4 / -20.8 us per segment, bits equal.
pub(crate) fn to_trf_main<CL: M, SL: M, N: AxisName, K: AxisName, const F: usize, const CH: usize, const CH2: usize, WE: M>(
    device: &mut Device,
    w: &DmTensor<i8, Chip, CL, SL, WE>,
) -> WTrf<CL, SL, N, K, F, CH, CH2> {
    device
        .main
        .begin(w.view())
        .fetch::<m![N % 8, N / 8 % 2, K / 32 % { F }, K / { CH2 } % 2, K / { CH } % 2], m![K % 32]>()
        .collect::<m![N % 8, N / 8 % 2, K / 32 % { F }, K / { CH2 } % 2, K / { CH } % 2], m![K % 32]>()
        .to_trf()
}

/// The output-row scales `ws [N % 16]` of the slice's 16 rows in its VRF.
pub(crate) fn col_scales<CL: M, SL: M, N: AxisName, WE: M>(
    device: &mut Device,
    ws: HbmTensorView<'_, f32, Chip, WE>,
) -> VrfTensor<f32, Chip, CL, SL, m![N / 8 % 2, N % 8]> {
    let ws: DmTensor<f32, Chip, CL, SL, m![N / 8 % 2, N % 8]> = ws.to_dm(&mut device.tdma);
    device.sub.begin(ws.view()).fetch::<m![N / 8 % 2], m![N % 8]>().collect::<m![N / 8 % 2], m![N % 8]>().to_vrf()
}

/// One pass with the epilogue without the row scale: `y[t, n] = bf16(f32(sum_k xq[t, k] wq[n, k]) * ws[n])`
/// for the slice's 16 rows and all tokens. For outputs that an RMS norm consumes (QKV heads, O,
/// Down): a per-token factor cancels in the norm, so `sx` is not applied.
#[allow(clippy::type_complexity)]
pub(crate) fn contract_ws<
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
    x: &XRing<CL, RING, TT, TR, K, CH>,
    w: &WTrf<CL, m![{ RING }, { XS }, { YS }], N, K, F, CH, CH2>,
    ws: &VrfTensor<f32, Chip, CL, m![{ RING }, { XS }, { YS }], m![N / 8 % 2, N % 8]>,
) -> DmTensor<bf16, Chip, CL, m![{ RING }, { XS }, { YS }], m![{ TT }, { TR }, N / 8 % 2, N % 8]> {
    device
        .main
        .begin(x.view())
        .fetch::<m![{ TT }, K / 32 % { F }], m![K % 32]>()
        .switch::<m![{ RING }, { XS }, { YS }], m![{ TT }, { TR }, K / 32 % { F }, K / { CH } % 4]>(SwitchConfig::Broadcast01 {
            slice1: TS,
            slice0: 4,
            time0: F,
        })
        .collect::<m![{ TT }, { TR }, K / 32 % { F }, K / { CH } % 4], m![K % 32]>()
        .contract_outer::<m![{ TT }, { TR }, K / 32 % { F }, K / { CH2 } % 2, N / 8 % 2], m![K / { CH } % 2, K % 32], _, _, _>(w)
        .contract_packet::<m![1]>()
        .contract_time::<m![{ TT }, { TR }, N / 8 % 2]>()
        .contract_lane::<m![{ TT }, { TR }, N / 8 % 2], m![N % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp_to_fp(31)
        .vector_narrow_split::<m![{ TT }, { TR }, N / 8 % 2, N / 4 % 2], m![N % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), ws)
        .vector_widen_concat::<m![{ TT }, { TR }, N / 8 % 2], m![N % 8]>()
        .vector_final()
        .cast::<bf16, m![N % 8 # 16]>()
        .commit_trim::<m![N % 8]>()
        .commit()
}

/// One pass without the epilogue: the i32 partial sums of the slice's 16 rows.
#[allow(clippy::type_complexity)]
pub(crate) fn contract_i32<
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
    x: &XRing<CL, RING, TT, TR, K, CH>,
    w: &WTrf<CL, m![{ RING }, { XS }, { YS }], N, K, F, CH, CH2>,
) -> DmTensor<i32, Chip, CL, m![{ RING }, { XS }, { YS }], m![{ TT }, { TR }, N / 8 % 2, N % 8]> {
    device
        .main
        .begin(x.view())
        .fetch::<m![{ TT }, K / 32 % { F }], m![K % 32]>()
        .switch::<m![{ RING }, { XS }, { YS }], m![{ TT }, { TR }, K / 32 % { F }, K / { CH } % 4]>(SwitchConfig::Broadcast01 {
            slice1: TS,
            slice0: 4,
            time0: F,
        })
        .collect::<m![{ TT }, { TR }, K / 32 % { F }, K / { CH } % 4], m![K % 32]>()
        .contract_outer::<m![{ TT }, { TR }, K / 32 % { F }, K / { CH2 } % 2, N / 8 % 2], m![K / { CH } % 2, K % 32], _, _, _>(w)
        .contract_packet::<m![1]>()
        .contract_time::<m![{ TT }, { TR }, N / 8 % 2]>()
        .contract_lane::<m![{ TT }, { TR }, N / 8 % 2], m![N % 8]>(LaneMode::Interleaved)
        .commit_trim::<m![N % 8]>()
        .commit()
}

axes![Pi = 2];
