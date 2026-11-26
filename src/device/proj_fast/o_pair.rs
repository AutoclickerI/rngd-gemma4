//! Fixed FP8 scalar epilogues with fully occupied paired vector passes.
use super::*;

type OMatrix<T, const HALF: usize, const PS: usize> = DmTensor<bf16, Chip, TCl<T, HALF>, m![H / 128 # 32, T % 8], m![T / { PS }, T % { HALF } / 64, T / 8 % 8, H / 16 % 8, H % 16]>;

fn project2<T: AxisName, const HALF: usize, const PS: usize, RING: M, XS: M, YS: M, const TS: usize>(
    device: &mut Device,
    sigma: &HbmTensor<f32, Chip, m![1 # 8]>,
    out_s: &HbmTensor<bf16, Chip, m![Rq, Hqs]>,
    rows_s: &HbmTensor<i32, Chip, m![T, Hh]>,
    eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>,
    wo: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    ws: &HbmTensor<f32, Chip, m![H]>,
) -> OMatrix<T, HALF, PS> {
    let ad0 = gather_pass_rows::<T, HALF, PS, 2, 0>(device, out_s, rows_s);
    let ad1 = gather_pass_rows::<T, HALF, PS, 2, 1>(device, out_s, rows_s);
    let wd0 = wo_half8::<T, HALF, RING, XS, YS>(device, wo, 0);
    let onehot = rotate::onehot::<T, HALF>(device, eye);
    let mut codes = HbmTensor::<f8e4m3, Chip, m![T, Qs]>::new();
    let mut scales = HbmTensor::<f32, Chip, m![T]>::new();
    quantize_static_pass8::<T, HALF, PS, 2, 0>(device, sigma, &ad0, &onehot, &mut codes, &mut scales);
    quantize_static_pass8::<T, HALF, PS, 2, 1>(device, sigma, &ad1, &onehot, &mut codes, &mut scales);
    gemm_scalar::<T, HALF, PS, RING, XS, YS, TS>(device, codes, sigma, wd0, wo, ws)
}

pub(crate) fn stage2<T: AxisName, const HALF: usize, const PS: usize, RING: M, XS: M, YS: M, const TS: usize>(
    device: &mut Device,
    sigma: &HbmTensor<f32, Chip, m![1 # 8]>,
    out_s: &HbmTensor<bf16, Chip, m![Rq, Hqs]>,
    rows_s: &HbmTensor<i32, Chip, m![T, Hh]>,
    x: &HbmTensor<bf16, Chip, m![T, H]>,
    eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>,
    wo: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    ws: &HbmTensor<f32, Chip, m![H]>,
    post_attn_rms_weight: &HbmTensor<bf16, Chip, m![Rc, H]>,
) -> (DmTensor<bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![Kh]>, DmTensor<bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![Kh]>) {
    let o = project2::<T, HALF, PS, RING, XS, YS, TS>(device, sigma, out_s, rows_s, eye, wo, ws);
    let gamma = rotate2::gamma_nat_ag2::<T, HALF>(device, post_attn_rms_weight);
    let y0 = pass_rows::<T, HALF, PS, 0>(device, &o);
    let x0 = crate::device::tokenwise_int8::post2::post_attention_from_dm::<T, HALF, PS, 2, 0>(device, &y0, &gamma, x);
    let y1 = pass_rows::<T, HALF, PS, 1>(device, &o);
    let x1 = crate::device::tokenwise_int8::post2::post_attention_from_dm::<T, HALF, PS, 2, 1>(device, &y1, &gamma, x);
    (x0, x1)
}

fn add_scalar_epilogue<T: AxisName, const HALF: usize, const PS: usize, SL: M, TT: M, TR: M, N: AxisName>(
    device: &mut Device,
    a: &DmTensor<f32, Chip, TCl<T, HALF>, SL, m![{ TT }, { TR }, N / 8 % 2, N % 8]>,
    b: &DmTensor<f32, Chip, TCl<T, HALF>, SL, m![{ TT }, { TR }, N / 8 % 2, N % 8]>,
    sx: &VrfTensor<f32, Chip, TCl<T, HALF>, SL, m![1 # 8]>,
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

// Two K-half contractions, with the first weight half already staged by the caller.
macro_rules! contract_halves {
    ($device:ident, $codes:ident, $wd0:ident, $wo:ident => $p0:ident, $p1:ident) => {
        let $codes: HbmTensor<f8e4m3, Chip, m![T, Ko2, Ko]> = unsafe { $codes.reshape() };
        let x0: fp8c::XRing8<TCl<T, HALF>, RING, m![T / { PS }, T % { HALF } / 64], m![T % 64], Ko, 512> =
            $codes.view().tile::<m![Ko2], 1, m![T, Ko2 = 1 # 2, Ko]>(0).to_dm(&mut $device.tdma);
        let $wd0: DmTensor<f8e4m3, Chip, TCl<T, HALF>, m![{ RING }, { XS }, { YS }], m![H % 16, Ko]> = unsafe { $wd0.reshape() };
        let w0 = fp8c::to_trf8_main::<TCl<T, HALF>, m![{ RING }, { XS }, { YS }], H, Ko, 16, 512, 1024, m![H % 16, Ko]>($device, &$wd0);
        let $p0 = fp8c::contract8_f32::<TCl<T, HALF>, RING, XS, YS, m![T / { PS }, T % { HALF } / 64], m![T % 64], H, Ko, TS, 16, 512, 1024>($device, &x0, &w0);

        let x1: fp8c::XRing8<TCl<T, HALF>, RING, m![T / { PS }, T % { HALF } / 64], m![T % 64], Ko, 512> =
            $codes.view().tile::<m![Ko2], 1, m![T, Ko2 = 1 # 2, Ko]>(1).to_dm(&mut $device.tdma);
        let wd1 = wo_half8::<T, HALF, RING, XS, YS>($device, $wo, 1);
        let wd1: DmTensor<f8e4m3, Chip, TCl<T, HALF>, m![{ RING }, { XS }, { YS }], m![H % 16, Ko]> = unsafe { wd1.reshape() };
        let w1 = fp8c::to_trf8_main::<TCl<T, HALF>, m![{ RING }, { XS }, { YS }], H, Ko, 16, 512, 1024, m![H % 16, Ko]>($device, &wd1);
        let $p1 = fp8c::contract8_f32::<TCl<T, HALF>, RING, XS, YS, m![T / { PS }, T % { HALF } / 64], m![T % 64], H, Ko, TS, 16, 512, 1024>($device, &x1, &w1);
    };
}

fn gemm_scalar<T: AxisName, const HALF: usize, const PS: usize, RING: M, XS: M, YS: M, const TS: usize>(
    device: &mut Device,
    codes: HbmTensor<f8e4m3, Chip, m![T, Qs]>,
    sigma: &HbmTensor<f32, Chip, m![1 # 8]>,
    wd0: DmTensor<f8e4m3, Chip, TCl<T, HALF>, m![{ RING }, { XS }, { YS }], m![H % 16, Qs % 2048]>,
    wo: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    ws: &HbmTensor<f32, Chip, m![H]>,
) -> OMatrix<T, HALF, PS> {
    contract_halves!(device, codes, wd0, wo => p0, p1);
    // The row scales after the second weight half in the DMA queue: only the epilogue reads them.
    // Static sigma is the same bit pattern for every row. Keep the two
    // epilogue multiplications in their original order, without a T-row VRF.
    let sd: DmTensor<f32, Chip, TCl<T, HALF>, m![{ RING }, { XS }, { YS }], m![1 # 8]> = sigma.to_dm(&mut device.tdma);
    let sx = device.sub.begin(sd.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let wsv = gemm::col_scales::<TCl<T, HALF>, m![{ RING }, { XS }, { YS }], H, m![H]>(device, ws.view());
    let y = add_scalar_epilogue::<T, HALF, PS, m![{ RING }, { XS }, { YS }], m![T / { PS }, T % { HALF } / 64], m![T % 64], H>(device, &p0, &p1, &sx, &wsv);
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
    y
}

// Preserve the whole-bucket O GEMM and its two K-half reduction. Only redistribute
// its completed bf16 result into each post-attention row layout, within DM.
fn pass_rows<T: AxisName, const HALF: usize, const PS: usize, const PI: usize>(
    device: &mut Device, y: &OMatrix<T, HALF, PS>,
) -> DmTensor<bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![Kh]> {
    let mut rows: DmTensor<bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![H]> = DmTensor::new();
    y.view().tile::<m![T / { PS }], 1,
        m![T / { PS } = 1 # 2, T % { HALF } / 64, T / 8 % 8, H / 16 % 8, H % 16]>(PI)
        .to_dm_view(&mut device.tdma, rows.view_mut());
    unsafe { rows.reshape() }
}

// Share the sigma DMA across quantization and the epilogue. The scalar GEMM
// consumes sigma directly.
fn codes_static_pass8_s1<T: AxisName, const HALF: usize, const PS: usize, const P: usize, const PI: usize>(
    device: &mut Device,
    s: &DmTensor<f32, Chip, TCl<T, HALF>, TSl<T, HALF>, m![1 # 8]>,
    ad: &DmTensor<bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![Qs]>,
    codes: &mut HbmTensor<f8e4m3, Chip, m![T, Qs]>,
) {
    let q = rotate2::row_quantize_static_f8_dm::<T, HALF, Qs>(device, s, ad);
    q.view().to_hbm_view(
        &mut device.tdma,
        codes.view_mut().tile::<m![T / { PS }], 1, m![T / { PS } = 1 #{!} { P }, T % { PS }, Qs]>(PI),
    );
}

fn gemm_scalar_s1<T: AxisName, const HALF: usize, const PS: usize, RING: M, XS: M, YS: M, const TS: usize>(
    device: &mut Device,
    codes: HbmTensor<f8e4m3, Chip, m![T, Qs]>,
    sd: &DmTensor<f32, Chip, TCl<T, HALF>, m![{ RING }, { XS }, { YS }], m![1 # 8]>,
    wd0: DmTensor<f8e4m3, Chip, TCl<T, HALF>, m![{ RING }, { XS }, { YS }], m![H % 16, Qs % 2048]>,
    wo: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    ws: &HbmTensor<f32, Chip, m![H]>,
) -> OMatrix<T, HALF, PS> {
    contract_halves!(device, codes, wd0, wo => p0, p1);
    let sx = device.sub.begin(sd.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let wsv = gemm::col_scales::<TCl<T, HALF>, m![{ RING }, { XS }, { YS }], H, m![H]>(device, ws.view());
    let y = add_scalar_epilogue::<T, HALF, PS, m![{ RING }, { XS }, { YS }], m![T / { PS }, T % { HALF } / 64], m![T % 64], H>(device, &p0, &p1, &sx, &wsv);
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
    y
}

// The F1 schedule and whole-bucket schedule stage the same FP8 inputs.
macro_rules! stage2_codes {
    ($device:ident, $sigma:ident, $out_s:ident, $rows_s:ident, $wo:ident => $codes:ident, $wd0:ident, $sd:ident) => {
        let ad0 = gather_pass_rows::<T, HALF, PS, 2, 0>($device, $out_s, $rows_s);
        let ad1 = gather_pass_rows::<T, HALF, PS, 2, 1>($device, $out_s, $rows_s);
        let $wd0 = wo_half8::<T, HALF, RING, XS, YS>($device, $wo, 0);
        let s: DmTensor<f32, Chip, TCl<T, HALF>, TSl<T, HALF>, m![1 # 8]> = $sigma.to_dm(&mut $device.tdma);
        let mut $codes = HbmTensor::<f8e4m3, Chip, m![T, Qs]>::new();
        codes_static_pass8_s1::<T, HALF, PS, 2, 0>($device, &s, &ad0, &mut $codes);
        codes_static_pass8_s1::<T, HALF, PS, 2, 1>($device, &s, &ad1, &mut $codes);
        // SAFETY: both slice maps cover the cluster's 256 slices once (`T % HALF # 256` with HALF = 256 rows here, and
        // RING x XS x YS = 256), and every slice holds the same 32-B sigma row (the broadcast load), so each epilogue slice
        // reads the value the sigma DMA wrote.
        const { assert!(HALF == 256) };
        let $sd: DmTensor<f32, Chip, TCl<T, HALF>, m![{ RING }, { XS }, { YS }], m![1 # 8]> = unsafe { s.reshape() };
    };
}

fn project2_s1<T: AxisName, const HALF: usize, const PS: usize, RING: M, XS: M, YS: M, const TS: usize>(
    device: &mut Device,
    sigma: &HbmTensor<f32, Chip, m![1 # 8]>,
    out_s: &HbmTensor<bf16, Chip, m![Rq, Hqs]>,
    rows_s: &HbmTensor<i32, Chip, m![T, Hh]>,
    wo: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    ws: &HbmTensor<f32, Chip, m![H]>,
) -> OMatrix<T, HALF, PS> {
    stage2_codes!(device, sigma, out_s, rows_s, wo => codes, wd0, sd);
    gemm_scalar_s1::<T, HALF, PS, RING, XS, YS, TS>(device, codes, &sd, wd0, wo, ws)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn stage2_s1<T: AxisName, const HALF: usize, const PS: usize, RING: M, XS: M, YS: M, const TS: usize>(
    device: &mut Device,
    sigma: &HbmTensor<f32, Chip, m![1 # 8]>,
    out_s: &HbmTensor<bf16, Chip, m![Rq, Hqs]>,
    rows_s: &HbmTensor<i32, Chip, m![T, Hh]>,
    x: &HbmTensor<bf16, Chip, m![T, H]>,
    wo: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    ws: &HbmTensor<f32, Chip, m![H]>,
    post_attn_rms_weight: &HbmTensor<bf16, Chip, m![Rc, H]>,
) -> (DmTensor<bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![Kh]>, DmTensor<bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![Kh]>) {
    let o = project2_s1::<T, HALF, PS, RING, XS, YS, TS>(device, sigma, out_s, rows_s, wo, ws);
    let gamma = rotate2::gamma_nat_ag2::<T, HALF>(device, post_attn_rms_weight);
    let y0 = pass_rows::<T, HALF, PS, 0>(device, &o);
    let x0 = crate::device::tokenwise_int8::post2::post_attention_from_dm::<T, HALF, PS, 2, 0>(device, &y0, &gamma, x);
    let y1 = pass_rows::<T, HALF, PS, 1>(device, &o);
    let x1 = crate::device::tokenwise_int8::post2::post_attention_from_dm::<T, HALF, PS, 2, 1>(device, &y1, &gamma, x);
    (x0, x1)
}

// oprojc4 F1 (pass-pipelined O tail): `stage2_s1` with the scalar epilogue, the InterTranspose and the row DMA issued
// per token pass. In `gemm_scalar_s1` the epilogue and the switch each cover both passes before the first `pass_rows`
// DMA can start; here pass 0's DMA (Main-idle in the static schedule) runs under pass 1's epilogue and switch. Same
// contraction and K-half sum, same per-element epilogue (AddF, Mul0 sigma, Mul1 col scale, bf16), same transpose and
// row geometry per pass: exact.
#[cfg(feature = "compile-carried")]
type OPart<T, const HALF: usize, const PS: usize, SL> = DmTensor<f32, Chip, TCl<T, HALF>, SL, m![T / { PS }, T % { HALF } / 64, T % 64, H / 8 % 2, H % 8]>;

#[cfg(feature = "compile-carried")]
fn epilogue_rows_s1<T: AxisName, const HALF: usize, const PS: usize, RING: M, XS: M, YS: M, const PI: usize>(
    device: &mut Device,
    p0: &OPart<T, HALF, PS, m![{ RING }, { XS }, { YS }]>,
    p1: &OPart<T, HALF, PS, m![{ RING }, { XS }, { YS }]>,
    sx: &VrfTensor<f32, Chip, TCl<T, HALF>, m![{ RING }, { XS }, { YS }], m![1 # 8]>,
    wsv: &VrfTensor<f32, Chip, TCl<T, HALF>, m![{ RING }, { XS }, { YS }], m![H / 8 % 2, H % 8]>,
) -> DmTensor<bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![Kh]> {
    let a = p0.view().tile::<m![T / { PS }], 1, m![T / { PS } = 1 # 2, T % { HALF } / 64, T % 64, H / 8 % 2, H % 8]>(PI);
    let b = p1.view().tile::<m![T / { PS }], 1, m![T / { PS } = 1 # 2, T % { HALF } / 64, T % 64, H / 8 % 2, H % 8]>(PI);
    let y: DmTensor<bf16, Chip, TCl<T, HALF>, m![{ RING }, { XS }, { YS }], m![T % { HALF } / 64, T % 64, H / 8 % 2, H % 8]> = device
        .main
        .begin_interleaved::<Pf8, _, _, _, _, _>(a, b)
        .fetch::<m![T % { HALF } / 64, T % 64, H / 8 % 2, Pf8], m![H % 8]>()
        .collect::<m![T % { HALF } / 64, T % 64, H / 8 % 2, Pf8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Pf8, m![T % { HALF } / 64, T % 64, H / 8 % 2]>()
        .vector_narrow_split::<m![T % { HALF } / 64, T % 64, H / 8 % 2, H / 4 % 2], m![H % 4]>()
        .vector_fp_zip(FpBinaryOp::AddF)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), sx)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), wsv)
        .vector_widen_concat::<m![T % { HALF } / 64, T % 64, H / 8 % 2], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    let y: DmTensor<bf16, Chip, TCl<T, HALF>, m![H / 128 # 32, H / 16 % 8], m![T % { HALF } / 64, T / 8 % 8, T % 8, H % 16]> =
        unsafe { y.reshape() };
    let y: DmTensor<bf16, Chip, TCl<T, HALF>, m![H / 128 # 32, T % 8], m![T % { HALF } / 64, T / 8 % 8, H / 16 % 8, H % 16]> = device
        .main
        .begin(y.view())
        .fetch::<m![T % { HALF } / 64, T / 8 % 8, T % 8], m![H % 16]>()
        .switch::<m![H / 128 # 32, T % 8], m![T % { HALF } / 64, T / 8 % 8, H / 16 % 8]>(SwitchConfig::InterTranspose {
            slice1: 8,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![T % { HALF } / 64, T / 8 % 8, H / 16 % 8], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();
    let mut rows: DmTensor<bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![H]> = DmTensor::new();
    y.view().to_dm_view(&mut device.tdma, rows.view_mut());
    unsafe { rows.reshape() }
}

#[cfg(feature = "compile-carried")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn stage2_s1_pp<T: AxisName, const HALF: usize, const PS: usize, RING: M, XS: M, YS: M, const TS: usize>(
    device: &mut Device,
    sigma: &HbmTensor<f32, Chip, m![1 # 8]>,
    out_s: &HbmTensor<bf16, Chip, m![Rq, Hqs]>,
    rows_s: &HbmTensor<i32, Chip, m![T, Hh]>,
    x: &HbmTensor<bf16, Chip, m![T, H]>,
    wo: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    ws: &HbmTensor<f32, Chip, m![H]>,
    post_attn_rms_weight: &HbmTensor<bf16, Chip, m![Rc, H]>,
) -> (DmTensor<bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![Kh]>, DmTensor<bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![Kh]>) {
    stage2_codes!(device, sigma, out_s, rows_s, wo => codes, wd0, sd);
    contract_halves!(device, codes, wd0, wo => p0, p1);
    let sx = device.sub.begin(sd.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let wsv = gemm::col_scales::<TCl<T, HALF>, m![{ RING }, { XS }, { YS }], H, m![H]>(device, ws.view());
    // SAFETY: the same in-slice layout written flat (`contract8_f32` nests its time axes as m![{TT}, {TR},..]).
    let p0: OPart<T, HALF, PS, m![{ RING }, { XS }, { YS }]> = unsafe { p0.reshape() };
    let p1: OPart<T, HALF, PS, m![{ RING }, { XS }, { YS }]> = unsafe { p1.reshape() };
    let gamma = rotate2::gamma_nat_ag2::<T, HALF>(device, post_attn_rms_weight);
    let y0 = epilogue_rows_s1::<T, HALF, PS, RING, XS, YS, 0>(device, &p0, &p1, &sx, &wsv);
    let y1 = epilogue_rows_s1::<T, HALF, PS, RING, XS, YS, 1>(device, &p0, &p1, &sx, &wsv);
    let x0 = crate::device::tokenwise_int8::post2::post_attention_from_dm::<T, HALF, PS, 2, 0>(device, &y0, &gamma, x);
    let x1 = crate::device::tokenwise_int8::post2::post_attention_from_dm::<T, HALF, PS, 2, 1>(device, &y1, &gamma, x);
    (x0, x1)
}

/// Project and normalize the residual for three token passes.
#[cfg(feature = "compile-carried")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn stage3<T: AxisName, const HALF: usize, const PS: usize, RING: M, XS: M, YS: M, const TS: usize>(
    device: &mut Device,
    sigma: &HbmTensor<f32, Chip, m![1 # 8]>,
    out_s: &HbmTensor<bf16, Chip, m![Rq, Hqs]>,
    rows_s: &HbmTensor<i32, Chip, m![T, Hh]>,
    x: &HbmTensor<bf16, Chip, m![T, H]>,
    eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>,
    wo: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    ws: &HbmTensor<f32, Chip, m![H]>,
    post_attn_rms_weight: &HbmTensor<bf16, Chip, m![Rc, H]>,
) -> (
    DmTensor<bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![Kh]>,
    DmTensor<bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![Kh]>,
    DmTensor<bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![Kh]>,
) {
    const { assert!(T::SIZE == 3 * PS && PS == 2 * HALF); };
    let o = project3::<T, HALF, PS, RING, XS, YS, TS>(device, sigma, out_s, rows_s, eye, wo, ws);
    let gamma = rotate2::gamma_nat_ag2::<T, HALF>(device, post_attn_rms_weight);
    let y0 = pass_rows_p::<T, HALF, PS, 3, 0>(device, &o);
    let x0 = crate::device::tokenwise_int8::post2::post_attention_from_dm::<T, HALF, PS, 3, 0>(device, &y0, &gamma, x);
    let y1 = pass_rows_p::<T, HALF, PS, 3, 1>(device, &o);
    let x1 = crate::device::tokenwise_int8::post2::post_attention_from_dm::<T, HALF, PS, 3, 1>(device, &y1, &gamma, x);
    let y2 = pass_rows_p::<T, HALF, PS, 3, 2>(device, &o);
    let x2 = crate::device::tokenwise_int8::post2::post_attention_from_dm::<T, HALF, PS, 3, 2>(device, &y2, &gamma, x);
    (x0, x1, x2)
}

#[cfg(feature = "compile-carried")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn stage3_s1<T: AxisName, const HALF: usize, const PS: usize, RING: M, XS: M, YS: M, const TS: usize>(
    device: &mut Device,
    sigma: &HbmTensor<f32, Chip, m![1 # 8]>,
    out_s: &HbmTensor<bf16, Chip, m![Rq, Hqs]>,
    rows_s: &HbmTensor<i32, Chip, m![T, Hh]>,
    x: &HbmTensor<bf16, Chip, m![T, H]>,
    wo: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    ws: &HbmTensor<f32, Chip, m![H]>,
    post_attn_rms_weight: &HbmTensor<bf16, Chip, m![Rc, H]>,
) -> (
    DmTensor<bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![Kh]>,
    DmTensor<bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![Kh]>,
    DmTensor<bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![Kh]>,
) {
    const { assert!(T::SIZE == 3 * PS && PS == 2 * HALF); };
    let o = project3_s1::<T, HALF, PS, RING, XS, YS, TS>(device, sigma, out_s, rows_s, wo, ws);
    let gamma = rotate2::gamma_nat_ag2::<T, HALF>(device, post_attn_rms_weight);
    let y0 = pass_rows_p::<T, HALF, PS, 3, 0>(device, &o);
    let x0 = crate::device::tokenwise_int8::post2::post_attention_from_dm::<T, HALF, PS, 3, 0>(device, &y0, &gamma, x);
    let y1 = pass_rows_p::<T, HALF, PS, 3, 1>(device, &o);
    let x1 = crate::device::tokenwise_int8::post2::post_attention_from_dm::<T, HALF, PS, 3, 1>(device, &y1, &gamma, x);
    let y2 = pass_rows_p::<T, HALF, PS, 3, 2>(device, &o);
    let x2 = crate::device::tokenwise_int8::post2::post_attention_from_dm::<T, HALF, PS, 3, 2>(device, &y2, &gamma, x);
    (x0, x1, x2)
}

#[cfg(feature = "compile-carried")]
#[allow(clippy::too_many_arguments)]
fn project3<T: AxisName, const HALF: usize, const PS: usize, RING: M, XS: M, YS: M, const TS: usize>(
    device: &mut Device,
    sigma: &HbmTensor<f32, Chip, m![1 # 8]>,
    out_s: &HbmTensor<bf16, Chip, m![Rq, Hqs]>,
    rows_s: &HbmTensor<i32, Chip, m![T, Hh]>,
    eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>,
    wo: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    ws: &HbmTensor<f32, Chip, m![H]>,
) -> OMatrix<T, HALF, PS> {
    let ad0 = gather_pass_rows::<T, HALF, PS, 3, 0>(device, out_s, rows_s);
    let ad1 = gather_pass_rows::<T, HALF, PS, 3, 1>(device, out_s, rows_s);
    let ad2 = gather_pass_rows::<T, HALF, PS, 3, 2>(device, out_s, rows_s);
    let wd0 = wo_half8::<T, HALF, RING, XS, YS>(device, wo, 0);
    let onehot = rotate::onehot::<T, HALF>(device, eye);
    let mut codes = HbmTensor::<f8e4m3, Chip, m![T, Qs]>::new();
    let mut scales = HbmTensor::<f32, Chip, m![T]>::new();
    quantize_static_pass8::<T, HALF, PS, 3, 0>(device, sigma, &ad0, &onehot, &mut codes, &mut scales);
    quantize_static_pass8::<T, HALF, PS, 3, 1>(device, sigma, &ad1, &onehot, &mut codes, &mut scales);
    quantize_static_pass8::<T, HALF, PS, 3, 2>(device, sigma, &ad2, &onehot, &mut codes, &mut scales);
    gemm_scalar::<T, HALF, PS, RING, XS, YS, TS>(device, codes, sigma, wd0, wo, ws)
}

#[cfg(feature = "compile-carried")]
#[allow(clippy::too_many_arguments)]
fn project3_s1<T: AxisName, const HALF: usize, const PS: usize, RING: M, XS: M, YS: M, const TS: usize>(
    device: &mut Device,
    sigma: &HbmTensor<f32, Chip, m![1 # 8]>,
    out_s: &HbmTensor<bf16, Chip, m![Rq, Hqs]>,
    rows_s: &HbmTensor<i32, Chip, m![T, Hh]>,
    wo: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    ws: &HbmTensor<f32, Chip, m![H]>,
) -> OMatrix<T, HALF, PS> {
    let ad0 = gather_pass_rows::<T, HALF, PS, 3, 0>(device, out_s, rows_s);
    let ad1 = gather_pass_rows::<T, HALF, PS, 3, 1>(device, out_s, rows_s);
    let ad2 = gather_pass_rows::<T, HALF, PS, 3, 2>(device, out_s, rows_s);
    let wd0 = wo_half8::<T, HALF, RING, XS, YS>(device, wo, 0);
    let sd0: DmTensor<f32, Chip, TCl<T, HALF>, TSl<T, HALF>, m![1 # 8]> = sigma.to_dm(&mut device.tdma);
    let mut codes = HbmTensor::<f8e4m3, Chip, m![T, Qs]>::new();
    codes_static_pass8_s1::<T, HALF, PS, 3, 0>(device, &sd0, &ad0, &mut codes);
    codes_static_pass8_s1::<T, HALF, PS, 3, 1>(device, &sd0, &ad1, &mut codes);
    codes_static_pass8_s1::<T, HALF, PS, 3, 2>(device, &sd0, &ad2, &mut codes);
    const { assert!(HALF == 256) };
    let sd: DmTensor<f32, Chip, TCl<T, HALF>, m![{ RING }, { XS }, { YS }], m![1 # 8]> = unsafe { sd0.reshape() };
    gemm_scalar_s1::<T, HALF, PS, RING, XS, YS, TS>(device, codes, &sd, wd0, wo, ws)
}

#[cfg(feature = "compile-carried")]
#[allow(clippy::too_many_arguments)]
fn pass_rows_p<T: AxisName, const HALF: usize, const PS: usize, const P: usize, const PI: usize>(
    device: &mut Device, y: &OMatrix<T, HALF, PS>,
) -> DmTensor<bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![Kh]> {
    let mut rows: DmTensor<bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![H]> = DmTensor::new();
    y.view().tile::<m![T / { PS }], 1,
        m![T / { PS } = 1 # { P }, T % { HALF } / 64, T / 8 % 8, H / 16 % 8, H % 16]>(PI)
        .to_dm_view(&mut device.tdma, rows.view_mut());
    unsafe { rows.reshape() }
}
