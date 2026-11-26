//! Prefix-row Sliding O with E4M3 activations and weights.
//! Quantize each activation row using `s = max(amax / 256, 1e-30)` and round to E4M3.
//! The two K halves contract in f32. Their epilogue computes `bf16(((p0 + p1) * sx) * ws)`.

use furiosa_opt_std::prelude::*;

use super::*;
use crate::device::tokenwise_int8::ffn_fast::fp8c;
use crate::device::tokenwise_int8::rotate::TSl;

use super::fp8c::Pf8;

/// The weight half `i` (`Qs / 2048 = i`) as E4M3 rows in DM (`wo_half0`'s layout).
fn wo_half8<T: AxisName, const HALF: usize, RING: M, XS: M, YS: M>(
    device: &mut Device,
    wo: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    i: usize,
) -> DmTensor<f8e4m3, Chip, TCl<T, HALF>, m![{ RING }, { XS }, { YS }], m![H % 16, Qs % 2048]> {
    wo.view().tile::<m![Qs / 2048], 1, m![H, Qs / 2048 = 1 # 2, Qs % 2048]>(i).to_dm(&mut device.tdma)
}

/// `bf16(((a + b) * sx[t]) * ws[n])` on the two K halves' f32 partial sums read in pair mode (`gemm::add_epilogue_f32`
/// with the row scale; `sx` holds the cluster's own rows, `o.rs`'s `row_scales`).
#[allow(clippy::type_complexity)]
fn add_epilogue8<T: AxisName, const HALF: usize, const PS: usize, SL: M, TT: M, TR: M, N: AxisName>(
    device: &mut Device,
    a: &DmTensor<f32, Chip, TCl<T, HALF>, SL, m![{ TT }, { TR }, N / 8 % 2, N % 8]>,
    b: &DmTensor<f32, Chip, TCl<T, HALF>, SL, m![{ TT }, { TR }, N / 8 % 2, N % 8]>,
    sx: &VrfTensor<f32, Chip, TCl<T, HALF>, SL, m![T / { PS }, T % { HALF }]>,
    ws: &VrfTensor<f32, Chip, TCl<T, HALF>, SL, m![N / 8 % 2, N % 8]>,
) -> DmTensor<bf16, Chip, TCl<T, HALF>, SL, m![{ TT }, { TR }, N / 8 % 2, N % 8]> {
    device
        .main
        .begin_interleaved::<Pf8, _, _, _, _, _>(a.view(), b.view())
        .fetch::<m![{ TT }, { TR }, N / 8 % 2, Pf8], m![N % 8]>()
        .collect::<m![{ TT }, { TR }, N / 8 % 2, Pf8], m![N % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Pf8, m![{ TT }, { TR }, N / 8 % 2]>()
        .vector_narrow_split::<m![{ TT }, { TR }, N / 8 % 2, N / 4 % 2], m![N % 4]>()
        .vector_fp_zip(FpBinaryOp::AddF)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), sx)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), ws)
        .vector_widen_concat::<m![{ TT }, { TR }, N / 8 % 2], m![N % 8]>()
        .vector_final()
        .cast::<bf16, m![N % 8 # 16]>()
        .commit_trim::<m![N % 8]>()
        .commit()
}

/// `o_gemm_store` on E4M3 codes and weights: both K halves (TRF fills from Main), `add_epilogue8`, the transpose and
/// the store of `o`.
#[allow(clippy::too_many_arguments)]
fn o_gemm_store8<T: AxisName, const HALF: usize, const PS: usize, RING: M, XS: M, YS: M, const TS: usize>(
    device: &mut Device,
    codes: HbmTensor<f8e4m3, Chip, m![T, Qs]>,
    scales: &HbmTensor<f32, Chip, m![T]>,
    wd0: DmTensor<f8e4m3, Chip, TCl<T, HALF>, m![{ RING }, { XS }, { YS }], m![H % 16, Qs % 2048]>,
    wo: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    ws: &HbmTensor<f32, Chip, m![H]>,
) -> HbmTensor<bf16, Chip, m![T, H]> {
    let codes: HbmTensor<f8e4m3, Chip, m![T, Ko2, Ko]> = unsafe { codes.reshape() };
    let x0: fp8c::XRing8<TCl<T, HALF>, RING, m![T / { PS }, T % { HALF } / 64], m![T % 64], Ko, 512> =
        codes.view().tile::<m![Ko2], 1, m![T, Ko2 = 1 # 2, Ko]>(0).to_dm(&mut device.tdma);
    let wd0: DmTensor<f8e4m3, Chip, TCl<T, HALF>, m![{ RING }, { XS }, { YS }], m![H % 16, Ko]> = unsafe { wd0.reshape() };
    let w0 = fp8c::to_trf8_main::<TCl<T, HALF>, m![{ RING }, { XS }, { YS }], H, Ko, 16, 512, 1024, m![H % 16, Ko]>(device, &wd0);
    let p0 = fp8c::contract8_f32::<TCl<T, HALF>, RING, XS, YS, m![T / { PS }, T % { HALF } / 64], m![T % 64], H, Ko, TS, 16, 512, 1024>(device, &x0, &w0);

    let x1: fp8c::XRing8<TCl<T, HALF>, RING, m![T / { PS }, T % { HALF } / 64], m![T % 64], Ko, 512> =
        codes.view().tile::<m![Ko2], 1, m![T, Ko2 = 1 # 2, Ko]>(1).to_dm(&mut device.tdma);
    let wd1 = wo_half8::<T, HALF, RING, XS, YS>(device, wo, 1);
    let wd1: DmTensor<f8e4m3, Chip, TCl<T, HALF>, m![{ RING }, { XS }, { YS }], m![H % 16, Ko]> = unsafe { wd1.reshape() };
    let w1 = fp8c::to_trf8_main::<TCl<T, HALF>, m![{ RING }, { XS }, { YS }], H, Ko, 16, 512, 1024, m![H % 16, Ko]>(device, &wd1);
    let p1 = fp8c::contract8_f32::<TCl<T, HALF>, RING, XS, YS, m![T / { PS }, T % { HALF } / 64], m![T % 64], H, Ko, TS, 16, 512, 1024>(device, &x1, &w1);

    // The row scales after the second weight half in the DMA queue: only the epilogue reads them.
    let sx = row_scales::<T, HALF, PS, m![{ RING }, { XS }, { YS }]>(device, scales);
    let wsv = gemm::col_scales::<TCl<T, HALF>, m![{ RING }, { XS }, { YS }], H, m![H]>(device, ws.view());
    let y = add_epilogue8::<T, HALF, PS, m![{ RING }, { XS }, { YS }], m![T / { PS }, T % { HALF } / 64], m![T % 64], H>(device, &p0, &p1, &sx, &wsv);
    let y: DmTensor<bf16, Chip, TCl<T, HALF>, m![H / 128 # 32, H / 16 % 8], m![T / { PS }, T % { HALF } / 64, T / 8 % 8, T % 8, H % 16]> =
        unsafe { y.reshape() };
    let y: DmTensor<bf16, Chip, TCl<T, HALF>, m![H / 128 # 32, T % 8], m![T / { PS }, T % { HALF } / 64, T / 8 % 8, H / 16 % 8, H % 16]> = device
        .main
        .begin(y.view())
        .fetch::<m![T / { PS }, T % { HALF } / 64, T / 8 % 8, T % 8], m![H % 16]>()
        .switch::<m![H / 128 # 32, T % 8], m![T / { PS }, T % { HALF } / 64, T / 8 % 8, H / 16 % 8]>(SwitchConfig::InterTranspose {
            slice1: 8,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![T / { PS }, T % { HALF } / 64, T / 8 % 8, H / 16 % 8], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();
    let mut o = HbmTensor::<bf16, Chip, m![T, H]>::new();
    y.view().to_hbm_view(&mut device.tdma, o.view_mut());
    o
}

// ---- F1 (static scales): the O stage of `precision_policy`'s `fp8-st` O carrier --
// The O input has no foldable producer in this program (its producer is the attention output), so one Main pass casts
// it: `e4m3(clamp(x[t, q] d[q], +-448))`. Here `d = 1 / sx_o` is all-gathered from the `o_signs` slot (`SlidingOF8`'s
// unread INT8-signs table, `[Rc, Qs]` bf16; F1's prep writes the layer's static factor into every column). `sx_o`
// folds into `ws` at prep, so the epilogue is `bf16((p0 + p1) * ws)`. The per-row amax pass, the scale store and the
// row-scale reload are gone.

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

fn o_rows_project8_static_1<T: AxisName, const HALF: usize, const PS: usize, RING: M, XS: M, YS: M, const TS: usize>(
    device: &mut Device,
    sigma: &HbmTensor<f32, Chip, m![1 # 8]>,
    out_s: &HbmTensor<bf16, Chip, m![Rq, Hqs]>,
    rows_s: &HbmTensor<i32, Chip, m![T, Hh]>,
    eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>,
    wo: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    ws: &HbmTensor<f32, Chip, m![H]>,
) -> HbmTensor<bf16, Chip, m![T, H]> {
    let ad0 = gather_pass_rows::<T, HALF, PS, 1, 0>(device, out_s, rows_s);
    let wd0 = wo_half8::<T, HALF, RING, XS, YS>(device, wo, 0);
    let onehot = rotate::onehot::<T, HALF>(device, eye);
    let mut codes = HbmTensor::<f8e4m3, Chip, m![T, Qs]>::new();
    let mut scales = HbmTensor::<f32, Chip, m![T]>::new();
    quantize_static_pass8::<T, HALF, PS, 1, 0>(device, sigma, &ad0, &onehot, &mut codes, &mut scales);
    o_gemm_store8::<T, HALF, PS, RING, XS, YS, TS>(device, codes, &scales, wd0, wo, ws)
}

/// The CPU-backend check of the O projection (one pass of `T = PS` rows, the Tok512 check): the gather, the per-row
/// E4M3 and `o_gemm_store8` as in `o_rows_project8_static_1`, the codes, scales and `o` copied out.
#[allow(clippy::too_many_arguments)]

/// The FP8 O stage, two passes, the post-attention rows kept in DM (`sliding_o_stage_rot_rows_v9_dm2` with
/// `o_rows_project8_static_1`; the same post-attention pass).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn sliding_o_stage_rows_static_dm1_f8<T: AxisName, const HALF: usize, const PS: usize, RING: M, XS: M, YS: M, const TS: usize>(
    device: &mut Device,
    sigma: &HbmTensor<f32, Chip, m![1 # 8]>,
    out_s: &HbmTensor<bf16, Chip, m![Rq, Hqs]>,
    rows_s: &HbmTensor<i32, Chip, m![T, Hh]>,
    x: &HbmTensor<bf16, Chip, m![T, H]>,
    eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>,
    wo: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    ws: &HbmTensor<f32, Chip, m![H]>,
    post_attn_rms_weight: &HbmTensor<bf16, Chip, m![Rc, H]>,
) -> DmTensor<bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![Kh]> {
    let o = o_rows_project8_static_1::<T, HALF, PS, RING, XS, YS, TS>(device, sigma, out_s, rows_s, eye, wo, ws);
    let gamma = rotate2::gamma_nat_ag2::<T, HALF>(device, post_attn_rms_weight);
    let x0 = crate::device::tokenwise_int8::post2::post_attention_pass_dm::<T, HALF, PS, 1, 0>(device, &o, &gamma, x);
    x0
}
