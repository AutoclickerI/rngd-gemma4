//! The INT8 carrier's Full stages (rotated per-row INT8, the declared arithmetic of `mod.rs`):
//!
//! - Full QKV: the signed input norm, the H60 (x) H64 rotation and per-row quantization of the
//!   hidden row (`Passes::norm_rotate_quantize`), the Q rows (8,192) on all 512 slices in one
//!   pass of K = 3,840, the K rows (512) on token-split rings (32 slices holding all 512 rows,
//!   replicated over the rings and the clusters, each ring contracting its own tokens), the
//!   epilogue `bf16((f32(acc) * sx) * ws)`, then the Full head phase (V from the raw K).
//! - Full O: the online per-head rotation of the attention output (signs, H8 (x) H64 per head,
//!   `1 / sqrt(512)`) and per-row quantization (`Passes::rotate_heads`), the INT8 O GEMM with
//!   K = 8,192 split across the clusters and two passes of 2,048 per cluster, the exact i32 sums
//!   added across the passes and the clusters, the epilogue, and the post-attention norm and
//!   residual add.
//!
//! The INT8 dot products are exact, so every K split gives the same bits; a row's outputs do not
//! depend on the bucket or its position.
use furiosa_opt_std::prelude::*;

use super::gemm::{self, Pi};
use super::passes::{OnePass, Passes, TwoPasses};
use super::post2;
use super::rotate::{self, Kb, Kh, Ob, Rc, Sgi, Sl8};
use super::rotate2::{self, D64};
use super::rotate_ff::K16;
use super::rotate_heads::{self, K8, O8};
use crate::Chip;
use crate::axes::{Dummy2, Dummy256, H, Hh, Hqf, Qf, Rq};

// --------------------------------------------------------------------------------- Full QKV

// ----------------------------------------------------------------------------------- Full O

/// The O cluster: the K half `Qf / 4096` (240 slices x 16 of the 3,840 rows each).
pub(crate) type OCl = m![Qf / 4096];

/// O pass `p`: the i32 sums over `Qf / 2048 % 2 = p` of the cluster's K half.
#[allow(clippy::type_complexity)]
fn o_pass_i8<T: AxisName, RING: M, XS: M, YS: M, TT: M, TR: M, const TS: usize>(
    device: &mut Device,
    codes: &HbmTensor<i8, Chip, m![T, Qf]>,
    wo: &HbmTensor<i8, Chip, m![H, Qf]>,
    p: usize,
) -> DmTensor<i32, Chip, OCl, m![{ RING }, { XS }, { YS }], m![{ TT }, { TR }, H / 8 % 2, H % 8]> {
    let ct = codes.view().tile::<m![Qf / 2048 % 2], 1, m![T, Qf / 4096, Qf / 2048 % 2 = 1 # 2, Qf % 2048]>(p);
    let xr: gemm::XRing<OCl, RING, TT, TR, Qf, 512> = ct.to_dm(&mut device.tdma);
    let wt = wo.view().tile::<m![Qf / 2048 % 2], 1, m![H, Qf / 4096, Qf / 2048 % 2 = 1 # 2, Qf % 2048]>(p);
    let wd: DmTensor<i8, Chip, OCl, m![{ RING }, { XS }, { YS }], m![H % 16, Qf % 2048]> = wt.to_dm(&mut device.tdma);
    let w = gemm::to_trf_main::<OCl, m![{ RING }, { XS }, { YS }], H, Qf, 16, 512, 1024, m![H % 16, Qf % 2048]>(device, &wd);
    gemm::contract_i32::<OCl, RING, XS, YS, TT, TR, H, Qf, TS, 16, 512, 1024>(device, &xr, &w)
}

/// An O pass's weights in DM: the cluster's K half, `Qf / 2048 % 2 = p` (`o_pass_i8`'s load).
pub(crate) type OWeights<RING, XS, YS> = DmTensor<i8, Chip, OCl, m![{ RING }, { XS }, { YS }], m![H % 16, Qf % 2048]>;

/// O pass `p`'s weights loaded on their own (`full_o_stage_i8_rows` issues the first pass's early).
fn o_weights_i8<RING: M, XS: M, YS: M>(device: &mut Device, wo: &HbmTensor<i8, Chip, m![H, Qf]>, p: usize) -> OWeights<RING, XS, YS> {
    let wt = wo.view().tile::<m![Qf / 2048 % 2], 1, m![H, Qf / 4096, Qf / 2048 % 2 = 1 # 2, Qf % 2048]>(p);
    wt.to_dm(&mut device.tdma)
}

/// `o_pass_i8` on weights already in DM (`o_weights_i8`).
#[allow(clippy::type_complexity)]
fn o_pass_i8_w<T: AxisName, RING: M, XS: M, YS: M, TT: M, TR: M, const TS: usize>(
    device: &mut Device,
    codes: &HbmTensor<i8, Chip, m![T, Qf]>,
    wd: &OWeights<RING, XS, YS>,
    p: usize,
) -> DmTensor<i32, Chip, OCl, m![{ RING }, { XS }, { YS }], m![{ TT }, { TR }, H / 8 % 2, H % 8]> {
    let ct = codes.view().tile::<m![Qf / 2048 % 2], 1, m![T, Qf / 4096, Qf / 2048 % 2 = 1 # 2, Qf % 2048]>(p);
    let xr: gemm::XRing<OCl, RING, TT, TR, Qf, 512> = ct.to_dm(&mut device.tdma);
    let w = gemm::to_trf_main::<OCl, m![{ RING }, { XS }, { YS }], H, Qf, 16, 512, 1024, m![H % 16, Qf % 2048]>(device, wd);
    gemm::contract_i32::<OCl, RING, XS, YS, TT, TR, H, Qf, TS, 16, 512, 1024>(device, &xr, &w)
}

/// `full_o_stage_i8_rows` up to the O passes: the rows' rotation and quantization
/// (`FullORowsPasses::rotate_heads_rows`) and both O passes' i32 partial sums, with the row scales.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn full_o_rows_partials<T: AxisName, const HALF: usize, const PS: usize, PP: Passes + FullORowsPasses, RING: M, XS: M, YS: M, TT: M, TR: M, const TS: usize>(
    device: &mut Device,
    out_f: &HbmTensor<bf16, Chip, m![Rq, Hqf]>,
    rows_f: &HbmTensor<i32, Chip, m![T, Hh]>,
    eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>,
    h64: &HbmTensor<bf16, Chip, m![Rc, Ob, Kb]>,
    h8: &HbmTensor<bf16, Chip, m![Rc, O8, K16]>,
    signs: &HbmTensor<bf16, Chip, m![Rc, Qf]>,
    wo: &HbmTensor<i8, Chip, m![H, Qf]>,
) -> (OPartial<RING, XS, YS, TT, TR>, OPartial<RING, XS, YS, TT, TR>, HbmTensor<f32, Chip, m![T]>) {
    let mut codes = HbmTensor::<i8, Chip, m![T, Qf]>::new();
    let mut scales = HbmTensor::<f32, Chip, m![T]>::new();
    let onehot = rotate::onehot::<T, HALF>(device, eye);
    let d = signs_ag_f::<T, HALF>(device, signs);
    let wd0 = PP::rotate_heads_rows::<T, HALF, PS, RING, XS, YS>(device, out_f, rows_f, wo, &d, h64, h8, &onehot, &mut codes, &mut scales);
    let p0 = o_pass_i8_w::<T, RING, XS, YS, TT, TR, TS>(device, &codes, &wd0, 0);
    let p1 = o_pass_i8::<T, RING, XS, YS, TT, TR, TS>(device, &codes, wo, 1);
    (p0, p1, scales)
}

/// Keep both passes' post-attention rows in DM.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn full_o_stage_i8_rows_dm2<T: AxisName, const HALF: usize, const PS: usize, PP: Passes + FullORowsPasses, RING: M, XS: M, YS: M, TT: M, TR: M, const TS: usize>(
    device: &mut Device,
    out_f: &HbmTensor<bf16, Chip, m![Rq, Hqf]>,
    rows_f: &HbmTensor<i32, Chip, m![T, Hh]>,
    x: &HbmTensor<bf16, Chip, m![T, H]>,
    eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>,
    h64: &HbmTensor<bf16, Chip, m![Rc, Ob, Kb]>,
    h8: &HbmTensor<bf16, Chip, m![Rc, O8, K16]>,
    signs: &HbmTensor<bf16, Chip, m![Rc, Qf]>,
    wo: &HbmTensor<i8, Chip, m![H, Qf]>,
    ws: &HbmTensor<f32, Chip, m![H]>,
    post_attn_rms_weight: &HbmTensor<bf16, Chip, m![Rc, H]>,
) -> (Row<bf16, T, HALF, m![Kh]>, Row<bf16, T, HALF, m![Kh]>) {
    let (p0, p1, scales) = full_o_rows_partials::<T, HALF, PS, PP, RING, XS, YS, TT, TR, TS>(device, out_f, rows_f, eye, h64, h8, signs, wo);
    let o = full_o_half_o::<T, RING, XS, YS, TT, TR, 512>(device, p0, p1, &scales, ws);
    let gamma = rotate2::gamma_nat_ag2::<T, HALF>(device, post_attn_rms_weight);
    let x0 = post2::post_attention_pass_dm::<T, HALF, PS, 2, 0>(device, &o, &gamma, x);
    let x1 = post2::post_attention_pass_dm::<T, HALF, PS, 2, 1>(device, &o, &gamma, x);
    (x0, x1)
}

type Row<D, T, const HALF: usize, E> = DmTensor<D, Chip, rotate::TCl<T, HALF>, rotate::TSl<T, HALF>, E>;
type Onehot<T, const HALF: usize> = VrfTensor<f32, Chip, rotate::TCl<T, HALF>, m![1 # 256], m![Sgi, Sl8]>;

/// The attention rows of token pass `PI` (of `P`) gathered from the Full row space into one token
/// per slice (`proj_fast::rot_heads::gather_pass_rows` on the Full rows: `[Hh, Hqf]` = `Qf`).
fn gather_pass_rows_f<T: AxisName, const HALF: usize, const PS: usize, const P: usize, const PI: usize>(
    device: &mut Device,
    out: &HbmTensor<bf16, Chip, m![Rq, Hqf]>,
    rows: &HbmTensor<i32, Chip, m![T, Hh]>,
) -> Row<bf16, T, HALF, m![Qf]> {
    let mut prow = HbmTensor::<i32, Chip, m![T % { PS }, Hh]>::new();
    rows.view()
        .tile::<m![T / { PS }], 1, m![T / { PS } = 1 # { P }, T % { PS }, Hh]>(PI)
        .to_hbm_view(&mut device.tdma, prow.view_mut());
    let ad: Row<bf16, T, HALF, m![Hh, Hqf]> = out.gather::<m![Rq], m![Hqf]>().by_byte_offsets(&prow).to_dm(&mut device.tdma);
    unsafe { ad.reshape() }
}

/// The Full O input's token passes on rows gathered from the row space (`full_o_stage_i8_rows`):
/// every pass's gather issued first, the H64 / H8 tables loaded once for both passes, then the
/// first O pass's weights (so they stream while the passes rotate), then per pass the operations
/// of `Passes::rotate_heads` (`rotate_heads::rotate_heads_pass_dm_q`) and its stores, the codes'
/// second K half (`Qf / 4096 = 1`, cluster 1's O input) at the other token half's rows
/// (`store_codes_swapped`). Returns the first O pass's weights. `half_finish` is the finish on
/// those rows (`full_o_half_finish` with the bucket's token half).
pub(crate) trait FullORowsPasses: Passes {
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn rotate_heads_rows<T: AxisName, const HALF: usize, const PS: usize, RING: M, XS: M, YS: M>(
        device: &mut Device,
        out_f: &HbmTensor<bf16, Chip, m![Rq, Hqf]>,
        rows_f: &HbmTensor<i32, Chip, m![T, Hh]>,
        wo: &HbmTensor<i8, Chip, m![H, Qf]>,
        d: &Row<bf16, T, HALF, m![Qf]>,
        h64: &HbmTensor<bf16, Chip, m![Rc, Ob, Kb]>,
        h8: &HbmTensor<bf16, Chip, m![Rc, O8, K16]>,
        onehot: &Onehot<T, HALF>,
        codes: &mut HbmTensor<i8, Chip, m![T, Qf]>,
        scales: &mut HbmTensor<f32, Chip, m![T]>,
    ) -> OWeights<RING, XS, YS>;

}

/// In one pass, each cluster stores its token half and exchanges the second K half.
impl FullORowsPasses for OnePass {
    fn rotate_heads_rows<T: AxisName, const HALF: usize, const PS: usize, RING: M, XS: M, YS: M>(
        device: &mut Device,
        out_f: &HbmTensor<bf16, Chip, m![Rq, Hqf]>,
        rows_f: &HbmTensor<i32, Chip, m![T, Hh]>,
        wo: &HbmTensor<i8, Chip, m![H, Qf]>,
        d: &Row<bf16, T, HALF, m![Qf]>,
        h64: &HbmTensor<bf16, Chip, m![Rc, Ob, Kb]>,
        h8: &HbmTensor<bf16, Chip, m![Rc, O8, K16]>,
        onehot: &Onehot<T, HALF>,
        codes: &mut HbmTensor<i8, Chip, m![T, Qf]>,
        scales: &mut HbmTensor<f32, Chip, m![T]>,
    ) -> OWeights<RING, XS, YS> {
        const { assert!(T::SIZE == 2 * HALF, "one pass, one token half per cluster") };
        let ad = gather_pass_rows_f::<T, HALF, PS, 1, 0>(device, out_f, rows_f);
        let d64 = h64_ag::<T, HALF>(device, h64);
        let (_h64t, _hbt) = rotate_heads::heads_tables_d64::<T, HALF, K8, O8, Self>(device, &d64, h8);
        let wd0 = o_weights_i8::<RING, XS, YS>(device, wo, 0);
        let (s, q) = {
            let ad = rotate2::scale_row_pow2::<T, HALF, Qf>(device, &ad, d);
            rotate2::row_quantize_i8::<T, HALF, Qf>(device, &ad)
        };
        q.view()
            .tile::<m![Qf / 4096], 1, m![Qf / 4096 = 1 # 2, Qf % 4096]>(0)
            .to_hbm_view(&mut device.tdma, codes.view_mut().tile::<m![Qf / 4096], 1, m![T, Qf / 4096 = 1 #{!} 2, Qf % 4096]>(0));
        // The second K half crosses to the other cluster, which stores it at its own token half.
        let q1: Row<i8, T, HALF, m![Qf % 4096]> = q.view().cluster_swap().cluster_slice::<m![Qf / 4096], m![Qf % 4096]>([1, 1]).to_dm(&mut device.tdma);
        q1.view().to_hbm_view(&mut device.tdma, codes.view_mut().tile::<m![Qf / 4096], 1, m![T, Qf / 4096 = 1 #{!} 2, Qf % 4096]>(1));
        rotate::store_scales::<T, HALF, m![T / { PS } = 1 #{!} 1, T % { PS }]>(
            device,
            s,
            onehot,
            scales.view_mut().tile::<m![T / { PS }], 1, m![T / { PS } = 1 #{!} 1, T % { PS }]>(0),
        );
        wd0
    }
}

/// Tok1024 (two passes of 512 rows; pass `PI` is token half `T / 512 = PI`): the pass's second
/// K half goes to token half `1 - PI`, one store per pass and K half.
impl FullORowsPasses for TwoPasses {
    fn rotate_heads_rows<T: AxisName, const HALF: usize, const PS: usize, RING: M, XS: M, YS: M>(
        device: &mut Device,
        out_f: &HbmTensor<bf16, Chip, m![Rq, Hqf]>,
        rows_f: &HbmTensor<i32, Chip, m![T, Hh]>,
        wo: &HbmTensor<i8, Chip, m![H, Qf]>,
        d: &Row<bf16, T, HALF, m![Qf]>,
        h64: &HbmTensor<bf16, Chip, m![Rc, Ob, Kb]>,
        h8: &HbmTensor<bf16, Chip, m![Rc, O8, K16]>,
        onehot: &Onehot<T, HALF>,
        codes: &mut HbmTensor<i8, Chip, m![T, Qf]>,
        scales: &mut HbmTensor<f32, Chip, m![T]>,
    ) -> OWeights<RING, XS, YS> {
        const { assert!(T::SIZE == 2 * PS, "the Tok1024 rows form: two passes, one token half each") };
        let ad0 = gather_pass_rows_f::<T, HALF, PS, 2, 0>(device, out_f, rows_f);
        let ad1 = gather_pass_rows_f::<T, HALF, PS, 2, 1>(device, out_f, rows_f);
        let d64 = h64_ag::<T, HALF>(device, h64);
        let (h64t, hbt) = rotate_heads::heads_tables_d64::<T, HALF, K8, O8, Self>(device, &d64, h8);
        let wd0 = o_weights_i8::<RING, XS, YS>(device, wo, 0);
        rotate_heads_pass_sw::<T, HALF, PS, 0>(device, &ad0, d, &h64t, &hbt, onehot, codes, scales);
        rotate_heads_pass_sw::<T, HALF, PS, 1>(device, &ad1, d, &h64t, &hbt, onehot, codes, scales);
        wd0
    }
}

/// Tok1024 pass `PI` (token half `T / PS = PI`) of the rows form: rotate and quantize, the
/// codes' first K half at the pass's rows and the second at the other half's, then the scales.
#[allow(clippy::too_many_arguments)]
fn rotate_heads_pass_sw<T: AxisName, const HALF: usize, const PS: usize, const PI: usize>(
    device: &mut Device,
    ad: &Row<bf16, T, HALF, m![Qf]>,
    d: &Row<bf16, T, HALF, m![Qf]>,
    _h64t: &rotate::H64Trf<T, HALF>,
    _hbt: &rotate_heads::HbTrf<T, HALF, K8, O8>,
    onehot: &Onehot<T, HALF>,
    codes: &mut HbmTensor<i8, Chip, m![T, Qf]>,
    scales: &mut HbmTensor<f32, Chip, m![T]>,
) {
    let (s, q) = {
        let ad = rotate2::scale_row_pow2::<T, HALF, Qf>(device, ad, d);
        rotate2::row_quantize_i8::<T, HALF, Qf>(device, &ad)
    };
    q.view().tile::<m![Qf / 4096], 1, m![Qf / 4096 = 1 # 2, Qf % 4096]>(0).to_hbm_view(
        &mut device.tdma,
        codes
            .view_mut()
            .tile::<m![T / { PS }], 1, m![T / { PS } = 1 #{!} 2, T % { PS }, Qf]>(PI)
            .tile::<m![Qf / 4096], 1, m![T / { PS } = 1 #{!} 2, T % { PS }, Qf / 4096 = 1 #{!} 2, Qf % 4096]>(0),
    );
    q.view().tile::<m![Qf / 4096], 1, m![Qf / 4096 = 1 # 2, Qf % 4096]>(1).to_hbm_view(
        &mut device.tdma,
        codes
            .view_mut()
            .tile::<m![T / { PS }], 1, m![T / { PS } = 1 #{!} 2, T % { PS }, Qf]>(1 - PI)
            .tile::<m![Qf / 4096], 1, m![T / { PS } = 1 #{!} 2, T % { PS }, Qf / 4096 = 1 #{!} 2, Qf % 4096]>(1),
    );
    rotate::store_scales::<T, HALF, m![T / { PS } = 1 #{!} 2, T % { PS }]>(
        device,
        s,
        onehot,
        scales.view_mut().tile::<m![T / { PS }], 1, m![T / { PS } = 1 #{!} 2, T % { PS }]>(PI),
    );
}

/// An O pass's i32 partial sums: the cluster's K half, every row, the slice's 16 outputs.
pub(crate) type OPartial<RING, XS, YS, TT, TR> = DmTensor<i32, Chip, OCl, m![{ RING }, { XS }, { YS }], m![{ TT }, { TR }, H / 8 % 2, H % 8]>;

/// One token half's (`T / HT`) i32 sums, `TT = T / 64`, `TR = T % 64` split at `HT`.
type HPart<T, const HT: usize, RING, XS, YS> = DmTensor<i32, Chip, OCl, m![{ RING }, { XS }, { YS }], m![T % { HT } / 64, T % 64, H / 8 % 2, H % 8]>;

/// `a + b` (exact i32) on row block `T / HT = h` of both passes' partials.
fn add_half<T: AxisName, const HT: usize, RING: M, XS: M, YS: M, TT: M, TR: M>(
    device: &mut Device,
    a: &OPartial<RING, XS, YS, TT, TR>,
    b: &OPartial<RING, XS, YS, TT, TR>,
    h: usize,
) -> HPart<T, HT, RING, XS, YS> {
    device
        .main
        .begin_interleaved::<Pi, _, _, _, _, _>(
            a.view().tile::<m![T / { HT }], 1, m![T / { HT } = 1 # 2, T % { HT } / 64, T % 64, H / 8 % 2, H % 8]>(h),
            b.view().tile::<m![T / { HT }], 1, m![T / { HT } = 1 # 2, T % { HT } / 64, T % 64, H / 8 % 2, H % 8]>(h),
        )
        .fetch::<m![T % { HT } / 64, T % 64, H / 8 % 2, Pi], m![H % 8]>()
        .collect::<m![T % { HT } / 64, T % 64, H / 8 % 2, Pi], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Pi, m![T % { HT } / 64, T % 64, H / 8 % 2]>()
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        .vector_final()
        .commit_trim::<m![H % 8]>()
        .commit()
}

/// `a + b` (exact i32) on one token half's sums.
fn add_h<T: AxisName, const HT: usize, RING: M, XS: M, YS: M>(
    device: &mut Device,
    a: &HPart<T, HT, RING, XS, YS>,
    b: &HPart<T, HT, RING, XS, YS>,
) -> HPart<T, HT, RING, XS, YS> {
    device
        .main
        .begin_interleaved::<Pi, _, _, _, _, _>(a.view(), b.view())
        .fetch::<m![T % { HT } / 64, T % 64, H / 8 % 2, Pi], m![H % 8]>()
        .collect::<m![T % { HT } / 64, T % 64, H / 8 % 2, Pi], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Pi, m![T % { HT } / 64, T % 64, H / 8 % 2]>()
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        .vector_final()
        .commit_trim::<m![H % 8]>()
        .commit()
}

/// Token half `T / HT = c`'s row scales in every slice of cluster `c` (`gemm::row_scales` on the half).
fn row_scales_half<T: AxisName, const HT: usize, SL: M>(device: &mut Device, sx: &HbmTensor<f32, Chip, m![T]>) -> VrfTensor<f32, Chip, OCl, SL, m![T % { HT }]> {
    let s: DmTensor<f32, Chip, m![T / { HT }], m![Dummy256], m![T % { HT }]> = sx.to_dm(&mut device.tdma);
    let s: DmTensor<f32, Chip, OCl, SL, m![T % { HT }]> = unsafe { s.reshape() };
    device.sub.begin(s.view()).fetch::<m![T % { HT } / 8], m![T % 8]>().collect::<m![T % { HT } / 8], m![T % 8]>().to_vrf()
}

/// `gemm::epilogue` on one token half's sums: `bf16((f32(acc) * sx[t]) * ws[n])`.
fn epilogue_h<T: AxisName, const HT: usize, RING: M, XS: M, YS: M>(
    device: &mut Device,
    acc: &HPart<T, HT, RING, XS, YS>,
    sx: &VrfTensor<f32, Chip, OCl, m![{ RING }, { XS }, { YS }], m![T % { HT }]>,
    ws: &VrfTensor<f32, Chip, OCl, m![{ RING }, { XS }, { YS }], m![H / 8 % 2, H % 8]>,
) -> DmTensor<bf16, Chip, OCl, m![{ RING }, { XS }, { YS }], m![T % { HT } / 64, T % 64, H / 8 % 2, H % 8]> {
    device
        .main
        .begin(acc.view())
        .fetch::<m![T % { HT } / 64, T % 64, H / 8 % 2], m![H % 8]>()
        .collect::<m![T % { HT } / 64, T % 64, H / 8 % 2], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp_to_fp(31)
        .vector_narrow_split::<m![T % { HT } / 64, T % 64, H / 8 % 2, H / 4 % 2], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), sx)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), ws)
        .vector_widen_concat::<m![T % { HT } / 64, T % 64, H / 8 % 2], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

/// `full_o_half_finish` up to the O rows `o [T, H]` in HBM (before the post-attention pass).
fn full_o_half_o<T: AxisName, RING: M, XS: M, YS: M, TT: M, TR: M, const HT: usize>(
    device: &mut Device,
    p0: OPartial<RING, XS, YS, TT, TR>,
    p1: OPartial<RING, XS, YS, TT, TR>,
    scales: &HbmTensor<f32, Chip, m![T]>,
    ws: &HbmTensor<f32, Chip, m![H]>,
) -> HbmTensor<bf16, Chip, m![T, H]> {
    const { assert!(T::SIZE == 2 * HT, "the half finish splits the rows in two") };
    let send = add_half::<T, HT, RING, XS, YS, TT, TR>(device, &p0, &p1, 1);
    let other: HPart<T, HT, RING, XS, YS> = send.view().cluster_swap().to_dm(&mut device.tdma);
    let own = add_half::<T, HT, RING, XS, YS, TT, TR>(device, &p0, &p1, 0);
    let acc = add_h::<T, HT, RING, XS, YS>(device, &own, &other);
    let sx = row_scales_half::<T, HT, m![{ RING }, { XS }, { YS }]>(device, scales);
    let wsv = gemm::col_scales::<OCl, m![{ RING }, { XS }, { YS }], H, m![H]>(device, ws.view());
    let y = epilogue_h::<T, HT, RING, XS, YS>(device, &acc, &sx, &wsv);

    // Cluster `c` holds token half `c`; it stores its rows through eight-slice transposes.
    let y: DmTensor<bf16, Chip, m![T / { HT }], m![H / 128 # 32, H / 16 % 8], m![T % { HT }, H % 16]> = unsafe { y.reshape() };
    let y: DmTensor<bf16, Chip, m![T / { HT }], m![H / 128 # 32, T % 8], m![T % { HT } / 8, H / 16 % 8, H % 16]> = device
        .main
        .begin(y.view())
        .fetch::<m![T % { HT } / 8, T % 8], m![H % 16]>()
        .switch::<m![H / 128 # 32, T % 8], m![T % { HT } / 8, H / 16 % 8]>(SwitchConfig::InterTranspose {
            slice1: 8,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![T % { HT } / 8, H / 16 % 8], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();
    let mut o = HbmTensor::<bf16, Chip, m![T, H]>::new();
    y.view().to_hbm_view(&mut device.tdma, o.view_mut());
    o
}

axes![Xf16 = 16];

/// The Full O signs `[Rc, Qf]` in every slice's DM as an all-gather (`rotate2::signs_ag2` on the
/// 8,192-wide row: slice `16 g + j` reads the 1 KB run `Qf / 512 = j` of copy `g`, then a
/// sixteen-slice `Broadcast1`) instead of a replicated load into all 512 slices (the shared-weight layout).
fn signs_ag_f<T: AxisName, const HALF: usize>(device: &mut Device, src: &HbmTensor<bf16, Chip, m![Rc, Qf]>) -> Row<bf16, T, HALF, m![Qf]> {
    let p: DmTensor<bf16, Chip, m![Dummy2], m![Rc, Qf / 512], m![Qf % 512]> = src.to_dm(&mut device.tdma);
    let p: DmTensor<bf16, Chip, rotate::TCl<T, HALF>, m![Rc, Qf / 512], m![Qf % 512]> = unsafe { p.reshape() };
    let d: DmTensor<bf16, Chip, rotate::TCl<T, HALF>, m![Rc, Xf16], m![Qf]> = device
        .main
        .begin(p.view())
        .fetch::<m![Qf % 512 / 16], m![Qf % 16]>()
        .switch::<m![Rc, Xf16], m![Qf % 512 / 16, Qf / 512]>(SwitchConfig::Broadcast1 { slice1: 16, slice0: 1 })
        .collect::<m![Qf % 512 / 16, Qf / 512], m![Qf % 16]>()
        .commit_trim::<m![Qf % 16]>()
        .commit();
    unsafe { d.reshape() }
}

/// The H64 factor in every slice's DM as an all-gather (`rotate2::dm_consts`'s `h64` half) instead
/// of a replicated load into all 512 slices (the shared-weight layout).
fn h64_ag<T: AxisName, const HALF: usize>(device: &mut Device, h64: &HbmTensor<bf16, Chip, m![Rc, Ob, Kb]>) -> Row<bf16, T, HALF, D64> {
    let p64: DmTensor<bf16, Chip, m![Dummy2], m![Rc, Ob / 4], m![Ob % 4, Kb]> = h64.to_dm(&mut device.tdma);
    let p64: DmTensor<bf16, Chip, rotate::TCl<T, HALF>, m![Rc, Ob / 4], m![Ob % 4, Kb]> = unsafe { p64.reshape() };
    let d64: DmTensor<bf16, Chip, rotate::TCl<T, HALF>, m![Rc, Xf16], D64> = device
        .main
        .begin(p64.view())
        .fetch::<m![Ob % 4, Kb / 16], m![Kb % 16]>()
        .switch::<m![Rc, Xf16], m![Ob % 4, Kb / 16, Ob / 4]>(SwitchConfig::Broadcast1 { slice1: 16, slice0: 1 })
        .collect::<m![Ob % 4, Kb / 16, Ob / 4], m![Kb % 16]>()
        .commit_trim::<m![Kb % 16]>()
        .commit();
    unsafe { d64.reshape() }
}

#[cfg(feature = "compile-carried")]
#[path = "core8_full_slices256.rs"]
mod slices256;

#[cfg(feature = "compile-carried")]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn full_o_stage_i8_rows_s256_dm2<T: AxisName, const HALF: usize, const PS: usize, PP: Passes + FullORowsPasses, RING: M, XS: M, YS: M, TT: M, TR: M, const TS: usize>(
    device: &mut Device,
    out_f: &HbmTensor<bf16, Chip, m![Rq, Hqf]>,
    rows_f: &HbmTensor<i32, Chip, m![T, Hh]>,
    x: &HbmTensor<bf16, Chip, m![T, H]>,
    eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>,
    _h64: &HbmTensor<bf16, Chip, m![Rc, Ob, Kb]>,
    _h8: &HbmTensor<bf16, Chip, m![Rc, O8, K16]>,
    signs: &HbmTensor<bf16, Chip, m![Rc, Qf]>,
    wo: &HbmTensor<i8, Chip, m![H, Qf]>,
    ws: &HbmTensor<f32, Chip, m![H]>,
    post_attn_rms_weight: &HbmTensor<bf16, Chip, m![Rc, H]>,
) -> (Row<bf16, T, HALF, m![Kh]>, Row<bf16, T, HALF, m![Kh]>) {
    const { assert!(T::SIZE == 1024 && HALF == 256 && PS == 512); }
    let mut codes = HbmTensor::<i8, Chip, m![T, Qf]>::new();
    let mut scales = HbmTensor::<f32, Chip, m![T]>::new();
    let onehot = rotate::onehot::<T, HALF>(device, eye);
    let d = signs_ag_f::<T, HALF>(device, signs);
    let ad0 = gather_pass_rows_f::<T, HALF, PS, 2, 0>(device, out_f, rows_f);
    let ad1 = gather_pass_rows_f::<T, HALF, PS, 2, 1>(device, out_f, rows_f);
    full_o_codes_natural_pass::<T, HALF, PS, 0>(device, &ad0, &d, &onehot, &mut codes, &mut scales);
    full_o_codes_natural_pass::<T, HALF, PS, 1>(device, &ad1, &d, &onehot, &mut codes, &mut scales);
    let wd0 = slices256::load_weight::<T, HALF>(device, wo, 0);
    let (y0, y1) = slices256::gemm::<T, HALF, PS>(device, &codes, &scales, wd0, wo, ws);
    let gamma = rotate2::gamma_nat_ag2::<T, HALF>(device, post_attn_rms_weight);
    let x0 = post2::post_attention_from_dm::<T, HALF, PS, 2, 0>(device, &y0, &gamma, x);
    let x1 = post2::post_attention_from_dm::<T, HALF, PS, 2, 1>(device, &y1, &gamma, x);
    (x0, x1)
}

// Apply the sign, power-of-two scale and INT8 quantizer, then store each full
// activation row in natural token order for the token-cluster GEMM.
#[cfg(feature = "compile-carried")]
#[allow(clippy::too_many_arguments)]
fn full_o_codes_natural_pass<T: AxisName, const HALF: usize, const PS: usize, const PI: usize>(
    device: &mut Device,
    ad: &Row<bf16, T, HALF, m![Qf]>,
    d: &Row<bf16, T, HALF, m![Qf]>,
    onehot: &Onehot<T, HALF>,
    codes: &mut HbmTensor<i8, Chip, m![T, Qf]>,
    scales: &mut HbmTensor<f32, Chip, m![T]>,
) {
    let (s, q) = {
        let ad = rotate2::scale_row_pow2::<T, HALF, Qf>(device, ad, d);
        rotate2::row_quantize_i8::<T, HALF, Qf>(device, &ad)
    };
    q.view().to_hbm_view(&mut device.tdma, codes.view_mut().tile::<m![T / { PS }], 1,
        m![T / { PS } = 1 #{!} 2, T % { PS }, Qf]>(PI));
    rotate::store_scales::<T, HALF, m![T / { PS } = 1 #{!} 2, T % { PS }]>(
        device, s, onehot,
        scales.view_mut().tile::<m![T / { PS }], 1, m![T / { PS } = 1 #{!} 2, T % { PS }]>(PI),
    );
}

/// Same per-row Full-O arithmetic with the token-half size explicit.
#[cfg(feature = "compile-carried")]
pub(crate) fn full_o_stage_i8_rows_dm1_half<T: AxisName, const HALF: usize, const PS: usize, PP: Passes + FullORowsPasses, RING: M, XS: M, YS: M, TT: M, TR: M, const TS: usize, const HT: usize>(
    device: &mut Device,
    out_f: &HbmTensor<bf16, Chip, m![Rq, Hqf]>,
    rows_f: &HbmTensor<i32, Chip, m![T, Hh]>,
    x: &HbmTensor<bf16, Chip, m![T, H]>,
    eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>,
    h64: &HbmTensor<bf16, Chip, m![Rc, Ob, Kb]>,
    h8: &HbmTensor<bf16, Chip, m![Rc, O8, K16]>,
    signs: &HbmTensor<bf16, Chip, m![Rc, Qf]>,
    wo: &HbmTensor<i8, Chip, m![H, Qf]>,
    ws: &HbmTensor<f32, Chip, m![H]>,
    post_attn_rms_weight: &HbmTensor<bf16, Chip, m![Rc, H]>,
) -> Row<bf16, T, HALF, m![Kh]> {
    let (p0, p1, scales) = full_o_rows_partials::<T, HALF, PS, PP, RING, XS, YS, TT, TR, TS>(device, out_f, rows_f, eye, h64, h8, signs, wo);
    let o = full_o_half_o::<T, RING, XS, YS, TT, TR, HT>(device, p0, p1, &scales, ws);
    let gamma = rotate2::gamma_nat_ag2::<T, HALF>(device, post_attn_rms_weight);
    super::post::post_attention_pass_dm::<T, HALF, PS, 1, 0>(device, &o, &gamma, x)
}

#[cfg(feature = "compile-carried")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn full_o_stage_i8_rows_dm3<T: AxisName, const HALF: usize, const PS: usize, PP: Passes + FullORowsPasses, RING: M, XS: M, YS: M, TT: M, TR: M, const TS: usize>(
    device: &mut Device,
    out_f: &HbmTensor<bf16, Chip, m![Rq, Hqf]>,
    rows_f: &HbmTensor<i32, Chip, m![T, Hh]>,
    x: &HbmTensor<bf16, Chip, m![T, H]>,
    eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>,
    h64: &HbmTensor<bf16, Chip, m![Rc, Ob, Kb]>,
    h8: &HbmTensor<bf16, Chip, m![Rc, O8, K16]>,
    signs: &HbmTensor<bf16, Chip, m![Rc, Qf]>,
    wo: &HbmTensor<i8, Chip, m![H, Qf]>,
    ws: &HbmTensor<f32, Chip, m![H]>,
    post_attn_rms_weight: &HbmTensor<bf16, Chip, m![Rc, H]>,
) -> (Row<bf16, T, HALF, m![Kh]>, Row<bf16, T, HALF, m![Kh]>, Row<bf16, T, HALF, m![Kh]>) {
    const { assert!(T::SIZE == 3 * PS && PS == 2 * HALF, "the Tok1536 rows form: three passes of two row blocks") };
    let (p0, p1, scales) = full_o_rows_partials::<T, HALF, PS, PP, RING, XS, YS, TT, TR, TS>(device, out_f, rows_f, eye, h64, h8, signs, wo);
    let o = full_o_half_o::<T, RING, XS, YS, TT, TR, 768>(device, p0, p1, &scales, ws);
    let gamma = rotate2::gamma_nat_ag2::<T, HALF>(device, post_attn_rms_weight);
    let x0 = post2::post_attention_pass_dm::<T, HALF, PS, 3, 0>(device, &o, &gamma, x);
    let x1 = post2::post_attention_pass_dm::<T, HALF, PS, 3, 1>(device, &o, &gamma, x);
    let x2 = post2::post_attention_pass_dm::<T, HALF, PS, 3, 2>(device, &o, &gamma, x);
    (x0, x1, x2)
}

#[cfg(feature = "compile-carried")]
#[allow(clippy::too_many_arguments)]
fn rotate_heads_pass_natural3<T: AxisName, const HALF: usize, const PS: usize, const PI: usize>(
    device: &mut Device,
    ad: &Row<bf16, T, HALF, m![Qf]>,
    d: &Row<bf16, T, HALF, m![Qf]>,
    _h64t: &rotate::H64Trf<T, HALF>,
    _hbt: &rotate_heads::HbTrf<T, HALF, K8, O8>,
    onehot: &Onehot<T, HALF>,
    codes: &mut HbmTensor<i8, Chip, m![T, Qf]>,
    scales: &mut HbmTensor<f32, Chip, m![T]>,
) {
    let (s, q) = {
        let ad = rotate2::scale_row_pow2::<T, HALF, Qf>(device, ad, d);
        rotate2::row_quantize_i8::<T, HALF, Qf>(device, &ad)
    };
    q.view().to_hbm_view(&mut device.tdma, codes.view_mut()
        .tile::<m![T / { PS }], 1, m![T / { PS } = 1 #{!} 3, T % { PS }, Qf]>(PI));
    rotate::store_scales::<T, HALF, m![T / { PS } = 1 #{!} 3, T % { PS }]>(
        device, s, onehot, scales.view_mut().tile::<m![T / { PS }], 1, m![T / { PS } = 1 #{!} 3, T % { PS }]>(PI),
    );
}

#[cfg(feature = "compile-carried")]
use crate::device::tokenwise_int8::passes::ThreePasses;
#[cfg(feature = "compile-carried")]
impl FullORowsPasses for ThreePasses {
    fn rotate_heads_rows<T: AxisName, const HALF: usize, const PS: usize, RING: M, XS: M, YS: M>(
        device: &mut Device,
        out_f: &HbmTensor<bf16, Chip, m![Rq, Hqf]>,
        rows_f: &HbmTensor<i32, Chip, m![T, Hh]>,
        wo: &HbmTensor<i8, Chip, m![H, Qf]>,
        d: &Row<bf16, T, HALF, m![Qf]>,
        h64: &HbmTensor<bf16, Chip, m![Rc, Ob, Kb]>,
        h8: &HbmTensor<bf16, Chip, m![Rc, O8, K16]>,
        onehot: &Onehot<T, HALF>,
        codes: &mut HbmTensor<i8, Chip, m![T, Qf]>,
        scales: &mut HbmTensor<f32, Chip, m![T]>,
    ) -> OWeights<RING, XS, YS> {
        const { assert!(T::SIZE == 3 * PS && PS == 2 * HALF, "the Tok1536 rows form: three passes of two row blocks") };
        let ad0 = gather_pass_rows_f::<T, HALF, PS, 3, 0>(device, out_f, rows_f);
        let ad1 = gather_pass_rows_f::<T, HALF, PS, 3, 1>(device, out_f, rows_f);
        let ad2 = gather_pass_rows_f::<T, HALF, PS, 3, 2>(device, out_f, rows_f);
        let d64 = h64_ag::<T, HALF>(device, h64);
        let (h64t, hbt) = rotate_heads::heads_tables_d64::<T, HALF, K8, O8, Self>(device, &d64, h8);
        let wd0 = o_weights_i8::<RING, XS, YS>(device, wo, 0);
        rotate_heads_pass_natural3::<T, HALF, PS, 0>(device, &ad0, d, &h64t, &hbt, onehot, codes, scales);
        rotate_heads_pass_natural3::<T, HALF, PS, 1>(device, &ad1, d, &h64t, &hbt, onehot, codes, scales);
        rotate_heads_pass_natural3::<T, HALF, PS, 2>(device, &ad2, d, &h64t, &hbt, onehot, codes, scales);
        let k1: DmTensor<i8, Chip, m![T / 768], m![T % 256], m![T % 768 / 256, Qf % 4096]> = codes.view()
            .tile::<m![Qf / 4096], 1, m![T, Qf / 4096 = 1 # 2, Qf % 4096]>(1)
            .to_dm(&mut device.tdma);
        let swapped: DmTensor<i8, Chip, m![T / 768], m![T % 256], m![T % 768 / 256, Qf % 4096]> =
            k1.view().cluster_swap().to_dm(&mut device.tdma);
        swapped.view().to_hbm_view(&mut device.tdma, codes.view_mut()
            .tile::<m![Qf / 4096], 1, m![T, Qf / 4096 = 1 #{!} 2, Qf % 4096]>(1));
        wd0
    }

}
