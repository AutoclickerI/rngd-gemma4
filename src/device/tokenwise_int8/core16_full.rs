//! The BF16 carrier's Full stages on the stationary-weight core (`gemm16`): the current
//! `ops_tokenwise` arithmetic (bf16 dot product, then the bf16 row scale) up to the f32
//! summation order, with the supplied FP8 codes decoded exactly to bf16 on the way into the TRF.
//!
//! - Full O: `K = Qf = 8,192` split across the clusters, four in-slice passes of 1,024 per
//!   cluster, summed as `(p0 + p1) + (p2 + p3)` on each cluster and once more across them.
//! - Full QKV: the 8,192 Q rows on all 512 slices (two K halves of 1,920 as the Sliding QKV),
//!   the 512 K rows on token-split rings (16 slices of 16 rows per ring, the rows replicated
//!   over the rings, each ring contracting its own tokens) with the same K tiling and the same
//!   `h0 + h1` order, so one host model covers both; V is the raw-K path of the head phase.
use furiosa_opt_std::prelude::*;

use super::core16::{fp8_rows, fp8_rows_half};
use super::gemm16::{self, Acc, Tile};
use super::passes::Passes;
use super::post;
use super::rotate::{self, Rc};
use crate::Chip;
use crate::axes::{H, Pf, Qf};

// ----------------------------------------------------------------------------------- Full O

/// The O cluster: the K half `Qf / 4096` (240 slices x 16 of the 3,840 rows each).
pub(crate) type OCl = m![Qf / 4096];

/// O pass `p`: the f32 sums over `Qf / 1024 % 4 = p` of the cluster's K half.
#[allow(clippy::type_complexity)]
fn o_pass<T: AxisName, RING: M, XS: M, YS: M, TT: M, TR: M, const TS: usize>(
    device: &mut Device,
    a: &HbmTensor<bf16, Chip, m![T, Qf]>,
    wo: &HbmTensor<f8e4m3, Chip, m![H, Qf]>,
    p: usize,
) -> Acc<OCl, m![{ RING }, { XS }, { YS }], TT, TR, H> {
    let at = a.view().tile::<m![Qf / 1024 % 4], 1, m![T, Qf / 4096, Qf / 1024 % 4 = 1 # 4, Qf % 1024]>(p);
    let xr: gemm16::XRing16<OCl, RING, TT, TR, Qf, 256> = at.to_dm(&mut device.tdma);
    let wt = wo.view().tile::<m![Qf / 1024 % 4], 1, m![H, Qf / 4096, Qf / 1024 % 4 = 1 # 4, Qf % 1024]>(p);
    let w8: DmTensor<f8e4m3, Chip, OCl, m![{ RING }, { XS }, { YS }], m![H % 16, Qf % 1024]> = wt.to_dm(&mut device.tdma);
    let wb = fp8_rows::<OCl, m![{ RING }, { XS }, { YS }], H, Qf, 1024>(device, &w8);
    let w = gemm16::to_trf::<OCl, m![{ RING }, { XS }, { YS }], H, Qf, 16, 256, 512, m![H % 16, Qf % 1024]>(device, &wb);
    gemm16::contract::<OCl, RING, XS, YS, TT, TR, H, Qf, TS, 16, 256, 512>(device, &xr, &w)
}

/// The BF16 Full O stage for `T` tokens (`o_stage` of `ops_tokenwise::full` on the stationary
/// core): `y = bf16(bf16(a. wo) * s)` with the dot summed as
/// `((q0 + q1) + (q2 + q3)) + ((q4 + q5) + (q6 + q7))` over the eight 1,024-wide K chunks
/// (chunks 0-3 on cluster 0, 4-7 on cluster 1), then the post-attention norm and residual add;
/// `x` is updated in place.
#[allow(clippy::too_many_arguments)]
pub(crate) fn full_o_stage<T: AxisName, const HALF: usize, const PS: usize, PP: Passes, RING: M, XS: M, YS: M, TT: M, TR: M, const TS: usize>(
    device: &mut Device,
    a: &HbmTensor<bf16, Chip, m![T, Qf]>,
    x: &mut HbmTensor<bf16, Chip, m![T, H]>,
    wo: &HbmTensor<f8e4m3, Chip, m![H, Qf]>,
    ws: &HbmTensor<bf16, Chip, m![H]>,
    post_attn_rms_weight: &HbmTensor<bf16, Chip, m![Rc, H]>,
) {
    let p0 = o_pass::<T, RING, XS, YS, TT, TR, TS>(device, a, wo, 0);
    let p1 = o_pass::<T, RING, XS, YS, TT, TR, TS>(device, a, wo, 1);
    let p01 = gemm16::add::<OCl, m![{ RING }, { XS }, { YS }], TT, TR, H>(device, &p0, &p1);
    let p2 = o_pass::<T, RING, XS, YS, TT, TR, TS>(device, a, wo, 2);
    let p3 = o_pass::<T, RING, XS, YS, TT, TR, TS>(device, a, wo, 3);
    let p23 = gemm16::add::<OCl, m![{ RING }, { XS }, { YS }], TT, TR, H>(device, &p2, &p3);
    let own = gemm16::add::<OCl, m![{ RING }, { XS }, { YS }], TT, TR, H>(device, &p01, &p23);
    let other: Acc<OCl, m![{ RING }, { XS }, { YS }], TT, TR, H> = own.view().cluster_swap().to_dm(&mut device.tdma);
    let dot: Tile<OCl, m![{ RING }, { XS }, { YS }], TT, TR, H> =
        gemm16::add_round::<OCl, m![{ RING }, { XS }, { YS }], TT, TR, H>(device, &own, &other);
    let wsv = gemm16::col_scales::<OCl, m![{ RING }, { XS }, { YS }], H, m![H]>(device, ws.view());
    let y = gemm16::scale_cols::<OCl, m![{ RING }, { XS }, { YS }], TT, TR, H>(device, &dot, &wsv);

    // Both clusters hold every row; cluster 0 stores them through eight-slice transposes (the
    // Sliding O store).
    let y: DmTensor<bf16, Chip, m![1 # 2], m![H / 128 # 32, H / 16 % 8], m![T, H % 16]> = unsafe { y.reshape() };
    let y: DmTensor<bf16, Chip, m![1 # 2], m![H / 128 # 32, T % 8], m![T / 8, H / 16 % 8, H % 16]> = device
        .main
        .begin(y.view())
        .fetch::<m![T / 8, T % 8], m![H % 16]>()
        .switch::<m![H / 128 # 32, T % 8], m![T / 8, H / 16 % 8]>(SwitchConfig::InterTranspose {
            slice1: 8,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![T / 8, H / 16 % 8], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();
    let mut o = HbmTensor::<bf16, Chip, m![T, H]>::new();
    y.view().to_hbm_view(&mut device.tdma, o.view_mut());

    let gamma = post::vector::<T, HALF, PP>(device, post_attn_rms_weight);
    PP::post_attention::<T, HALF, PS>(device, &o, &gamma, x);
}

// --------------------------------------------------------------------------------- Full QKV

/// The Q rows' cluster: `Qf / 4096` (256 slices x 16 rows each, no padding).
pub(crate) type QCl = m![Qf / 4096];
/// The K rows' cluster: `Pf / 256` (16 slices x 16 rows, one ring of `KTS` token slots x 4 K
/// chunks, replicated over the token-split rings `KR`).
pub(crate) type KCl = m![Pf / 256];
pub(crate) type KXs = m![Pf / 64 % 4];
pub(crate) type KYs = m![Pf / 16 % 4];
pub(crate) type KSl<KR> = m![{ KR }, { KXs }, { KYs }];
const KTS: usize = 4;

/// Q pass `p` (K half `H / 1920 = p`): the f32 sums of the slice's 16 Q rows for all tokens.
#[allow(clippy::type_complexity)]
fn q_pass<T: AxisName, RING: M, XS: M, YS: M, TT: M, TR: M, const TS: usize>(
    device: &mut Device,
    h: &HbmTensor<bf16, Chip, m![T, H]>,
    w8: &DmTensor<f8e4m3, Chip, QCl, m![{ RING }, { XS }, { YS }], m![Qf % 16, H]>,
    p: usize,
) -> Acc<QCl, m![{ RING }, { XS }, { YS }], TT, TR, Qf> {
    let ht = h.view().tile::<m![H / 1920], 1, m![T, H / 1920 = 1 # 2, H % 1920]>(p);
    let xr: gemm16::XRing16<QCl, RING, TT, TR, H, 480> = ht.to_dm(&mut device.tdma);
    let wb = fp8_rows_half::<QCl, m![{ RING }, { XS }, { YS }], Qf>(device, w8, p);
    let w = gemm16::to_trf::<QCl, m![{ RING }, { XS }, { YS }], Qf, H, 30, 480, 960, m![Qf % 16, H % 1920]>(device, &wb);
    gemm16::contract::<QCl, RING, XS, YS, TT, TR, Qf, H, TS, 30, 480, 960>(device, &xr, &w)
}

/// K pass `p` (K half `H / 1920 = p`): the f32 sums of the slice's 16 K rows for its ring's
/// tokens, in the Q pass's K order.
#[allow(clippy::type_complexity)]
fn k_pass<T: AxisName, KR: M, KTT: M, KTR: M>(
    device: &mut Device,
    h: &HbmTensor<bf16, Chip, m![T, H]>,
    w8: &DmTensor<f8e4m3, Chip, KCl, KSl<KR>, m![Pf % 16, H]>,
    p: usize,
) -> Acc<KCl, KSl<KR>, KTT, KTR, Pf> {
    let ht = h.view().tile::<m![H / 1920], 1, m![T, H / 1920 = 1 # 2, H % 1920]>(p);
    let xr: gemm16::XRing16<KCl, KR, KTT, KTR, H, 480> = ht.to_dm(&mut device.tdma);
    let wb = fp8_rows_half::<KCl, KSl<KR>, Pf>(device, w8, p);
    let w = gemm16::to_trf::<KCl, KSl<KR>, Pf, H, 30, 480, 960, m![Pf % 16, H % 1920]>(device, &wb);
    gemm16::contract::<KCl, KR, KXs, KYs, KTT, KTR, Pf, H, KTS, 30, 480, 960>(device, &xr, &w)
}

/// The Full head phase on the scaled projections `q_raw [T, Qf]`, `k_raw [T, Pf]` (one choice per
/// entry, as a type: a const-generic branch around two allocating phases does not lower).
/// `Rolled` is `heads_full` (one (row, query-head pair) per slice, `P` rolled passes of `PS`
/// rows, `PH` pairs); `OneRow` / `OneRow2` are `heads_full1` (one row per slice with all its
/// heads, `PS2 = 2 HALF` rows per pass, one or two const-generic passes). The arithmetic is the
/// same per element, so the outputs are bit-identical.
#[allow(clippy::too_many_arguments)]
/// The projection part of the Full QKV stage: input norm, then the scaled Q (`[T, Qf]`) and K
/// (`[T, Pf]`) dots, stored in token-major order for the head phase.
#[allow(clippy::too_many_arguments)]
pub(crate) fn full_qkv_gemm<
    T: AxisName,
    const HALF: usize,
    const PS2: usize,
    PP: Passes,
    RING: M,
    XS: M,
    YS: M,
    TT: M,
    TR: M,
    const TS: usize,
    KR: M,
    KTT: M,
    KTR: M,
    KT8: M,
    KT1: M,
>(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![T, H]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![Rc, H]>,
    q_weight: &HbmTensor<f8e4m3, Chip, m![Qf, H]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Pf, H]>,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qf]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Pf]>,
    q_raw: &mut HbmTensor<bf16, Chip, m![T, Qf]>,
    k_raw: &mut HbmTensor<bf16, Chip, m![T, Pf]>,
) {
    let g = rotate::gamma::<T, HALF, PP>(device, input_rms_weight);
    let mut h = HbmTensor::<bf16, Chip, m![T, H]>::new();
    PP::normalize_rows::<T, HALF, PS2>(device, x, &g, &mut h);

    // Q: every slice holds 16 rows, all tokens stream through each ring.
    // The Q and K codes as one DMA each of whole 3,840-byte rows (the half rows were misaligned to
    // the 256-byte unit); the passes decode their half from DM.
    let wq8: DmTensor<f8e4m3, Chip, QCl, m![{ RING }, { XS }, { YS }], m![Qf % 16, H]> = q_weight.to_dm(&mut device.tdma);
    let p0 = q_pass::<T, RING, XS, YS, TT, TR, TS>(device, &h, &wq8, 0);
    let p1 = q_pass::<T, RING, XS, YS, TT, TR, TS>(device, &h, &wq8, 1);
    let dot = gemm16::add_round::<QCl, m![{ RING }, { XS }, { YS }], TT, TR, Qf>(device, &p0, &p1);
    let wsq = gemm16::col_scales::<QCl, m![{ RING }, { XS }, { YS }], Qf, m![Qf]>(device, q_weight_scale.view());
    let y = gemm16::scale_cols::<QCl, m![{ RING }, { XS }, { YS }], TT, TR, Qf>(device, &dot, &wsq);
    // Eight-slice transposes give 256-byte runs per token.
    let y: DmTensor<bf16, Chip, QCl, m![Qf / 128 % 32, Qf / 16 % 8], m![T, Qf % 16]> = unsafe { y.reshape() };
    let y: DmTensor<bf16, Chip, QCl, m![Qf / 128 % 32, T % 8], m![T / 8, Qf / 16 % 8, Qf % 16]> = device
        .main
        .begin(y.view())
        .fetch::<m![T / 8, T % 8], m![Qf % 16]>()
        .switch::<m![Qf / 128 % 32, T % 8], m![T / 8, Qf / 16 % 8]>(SwitchConfig::InterTranspose {
            slice1: 8,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![T / 8, Qf / 16 % 8], m![Qf % 16]>()
        .commit_trim::<m![Qf % 16]>()
        .commit();
    y.view().to_hbm_view(&mut device.tdma, q_raw.view_mut());

    // K: each ring holds the cluster's 256 rows and contracts its own tokens.
    let wk8: DmTensor<f8e4m3, Chip, KCl, KSl<KR>, m![Pf % 16, H]> = k_weight.to_dm(&mut device.tdma);
    let k0 = k_pass::<T, KR, KTT, KTR>(device, &h, &wk8, 0);
    let k1 = k_pass::<T, KR, KTT, KTR>(device, &h, &wk8, 1);
    let dotk = gemm16::add_round::<KCl, KSl<KR>, KTT, KTR, Pf>(device, &k0, &k1);
    let wsk = gemm16::col_scales::<KCl, KSl<KR>, Pf, m![Pf]>(device, k_weight_scale.view());
    let yk = gemm16::scale_cols::<KCl, KSl<KR>, KTT, KTR, Pf>(device, &dotk, &wsk);
    // Eight-slice transposes give 256-byte runs per token (a direct store of the slice's 16
    // columns per token ran at 0.3% DMA utilization: 145k of 529k static cycles at Tok512). The
    // ring's tokens `[KTT, KTR]` are the same bits as `[KT8, KT1]` (`T / 8` within the ring,
    // `T % 8`).
    let yk: DmTensor<bf16, Chip, KCl, m![{ KR }, Pf / 128 % 2, Pf / 16 % 8], m![{ KT8 }, { KT1 }, Pf % 16]> = unsafe { yk.reshape() };
    let yk: DmTensor<bf16, Chip, KCl, m![{ KR }, Pf / 128 % 2, { KT1 }], m![{ KT8 }, Pf / 16 % 8, Pf % 16]> = device
        .main
        .begin(yk.view())
        .fetch::<m![{ KT8 }, { KT1 }], m![Pf % 16]>()
        .switch::<m![{ KR }, Pf / 128 % 2, { KT1 }], m![{ KT8 }, Pf / 16 % 8]>(SwitchConfig::InterTranspose {
            slice1: 8,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![{ KT8 }, Pf / 16 % 8], m![Pf % 16]>()
        .commit_trim::<m![Pf % 16]>()
        .commit();
    yk.view().to_hbm_view(&mut device.tdma, k_raw.view_mut());
}
