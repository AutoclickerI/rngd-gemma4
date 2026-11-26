//! The BF16 carrier on the stationary-weight core (`gemm16`): the current `ops_tokenwise`
//! arithmetic (bf16 dot product, then the bf16 scale), with the supplied FP8 / NVFP4 codes
//! decoded exactly to bf16 on the way into the TRF, and the per-token phases of the INT8 stages
//! (one token per slice).
use furiosa_opt_std::prelude::*;

use super::gemm::Pi;
use crate::axes::Dummy256;
use super::gemm16::{self, Acc};
use super::passes::Passes;
use super::post;
use super::rotate::{self, Rc};
use crate::Chip;
use crate::axes::{H, L};

/// FP8 weight rows (DM, `[N % 16, K % KP]`) decoded exactly to bf16 (the fetch unit's table).
pub(crate) fn fp8_rows<CL: M, SL: M, N: AxisName, K: AxisName, const KP: usize>(
    device: &mut Device,
    w: &DmTensor<f8e4m3, Chip, CL, SL, m![N % 16, K % { KP }]>,
) -> DmTensor<bf16, Chip, CL, SL, m![N % 16, K % { KP }]> {
    device
        .main
        .begin(w.view())
        .fetch::<m![N % 16, K % { KP } / 32], m![K % 32]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![N % 16, K % { KP } / 16], m![K % 16]>()
        .commit_trim::<m![K % 16]>()
        .commit()
}

// ------------------------------------------------------------------------------------ Sliding O

// ---------------------------------------------------------------------------------- Sliding QKV

/// K half `p` (`H / 1920 = p`) of whole FP8 weight rows in DM (`[N % 16, H]`, loaded as one
/// DMA of 3,840-byte rows), decoded exactly to bf16 (`fp8_rows` on the half's DM tile).
pub(crate) fn fp8_rows_half<CL: M, SL: M, N: AxisName>(
    device: &mut Device,
    w: &DmTensor<f8e4m3, Chip, CL, SL, m![N % 16, H]>,
    p: usize,
) -> DmTensor<bf16, Chip, CL, SL, m![N % 16, H % 1920]> {
    device
        .main
        .begin(w.view().tile::<m![H / 1920], 1, m![N % 16, H / 1920 = 1 # 2, H % 1920]>(p))
        .fetch::<m![N % 16, H % 1920 / 32], m![H % 32]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![N % 16, H % 1920 / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit()
}

// ------------------------------------------------------------------------------------------ FFN

/// Up/Gate rows' cluster (`L / 256 % 2`) and Down's K half per cluster (`L / 7680`).
pub(crate) type UCl16 = m![L / 256 % 2];
pub(crate) type DCl16 = m![L / 7680];

/// A GEMM output tile as 16 consecutive output rows per token.
type Tile16<CL, SL, TT, TR> = DmTensor<bf16, Chip, CL, SL, m![{ TT }, { TR }, L % 16]>;

/// The GeGLU of row group `j`: `act = bf16(u * (erf(g / sqrt 2) + 1) * g / 2)` from the bf16 Up
/// and Gate outputs (computed in f32), committed into group `j` of `act`.
#[allow(clippy::type_complexity)]
fn geglu16<CL: M, SL: M, TT: M, TR: M>(
    device: &mut Device,
    up: &Tile16<CL, SL, TT, TR>,
    gate: &Tile16<CL, SL, TT, TR>,
    j: usize,
    act: &mut DmTensor<bf16, Chip, CL, SL, m![{ TT }, { TR }, L / 128 % 2, L % 16]>,
) {
    const INVSQRT2: f32 = 0.70710678118f32;
    let gl: DmTensor<f32, Chip, CL, SL, m![{ TT }, { TR }, L % 16]> = device
        .main
        .begin(gate.view())
        .fetch::<m![{ TT }, { TR }], m![L % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![{ TT }, { TR }, L / 8 % 2], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{ TT }, { TR }, L / 4 % 4], m![L % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), INVSQRT2)
        .vector_fp_unary(FpUnaryOp::Erf)
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_widen_concat::<m![{ TT }, { TR }, L / 8 % 2], m![L % 8]>()
        .vector_final()
        .commit_trim::<m![L % 8]>()
        .commit();
    let u: DmTensor<f32, Chip, CL, SL, m![{ TT }, { TR }, L % 16]> = device
        .main
        .begin(up.view())
        .fetch::<m![{ TT }, { TR }], m![L % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![{ TT }, { TR }, L / 8 % 2], m![L % 8]>()
        .commit_trim::<m![L % 8]>()
        .commit();
    device
        .main
        .begin_interleaved::<Pi, _, _, _, _, _>(u.view(), gl.view())
        .fetch::<m![{ TT }, { TR }, L / 8 % 2, Pi], m![L % 8]>()
        .collect::<m![{ TT }, { TR }, L / 8 % 2, Pi], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Pi, m![{ TT }, { TR }, L / 8 % 2]>()
        .vector_narrow_split::<m![{ TT }, { TR }, L / 4 % 4], m![L % 4]>()
        .vector_fp_zip(FpBinaryOp::MulF(FpMulAlu::Mul0))
        .vector_fp_div(2f32)
        .vector_widen_concat::<m![{ TT }, { TR }, L / 8 % 2], m![L % 8]>()
        .vector_final()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit_view(act.view_mut().tile::<m![L / 128 % 2], 1, m![{ TT }, { TR }, L / 128 % 2 = 1 #{!} 2, L % 16]>(j));
}

/// Up/Gate rows `L / 128 % 2 = j` of the slice with their full K: packed codes (one 30 KB run).
fn ug_codes<SL: M>(device: &mut Device, w: &HbmTensor<f4e2m1, Chip, m![L, H]>, j: usize) -> DmTensor<f4e2m1, Chip, UCl16, SL, m![L % 16, H]> {
    w.view().tile::<m![L / 128 % 2], 1, m![L / 256, L / 128 % 2 = 1 # 2, L % 128, H]>(j).to_dm(&mut device.tdma)
}

/// The same rows' E4M3 block scales (one 3.8 KB run).
fn ug_scales<SL: M>(device: &mut Device, s: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>, j: usize) -> DmTensor<f8e4m3, Chip, UCl16, SL, m![L % 16, H / 16]> {
    s.view().tile::<m![L / 128 % 2], 1, m![L / 256, L / 128 % 2 = 1 # 2, L % 128, H / 16]>(j).to_dm(&mut device.tdma)
}

/// K half `k` of Up/Gate rows decoded exactly to bf16 (`core16::fp4_rows` on views of the full
/// rows): the fetch table unpacks the codes to E4M3, the vector engine multiplies by the block scale.
#[allow(clippy::type_complexity)]
fn ug_decode<SL: M>(
    device: &mut Device,
    w: &DmTensor<f4e2m1, Chip, UCl16, SL, m![L % 16, H]>,
    s: &DmTensor<f8e4m3, Chip, UCl16, SL, m![L % 16, H / 16]>,
    k: usize,
) -> DmTensor<bf16, Chip, UCl16, SL, m![L % 16, H % 1920]> {
    let sv: VrfTensor<f32, Chip, UCl16, SL, m![L % 16, H / 16 % 120]> = device
        .sub
        .begin(s.view().tile::<m![H / 1920], 1, m![L % 16, H / 1920 = 1 # 2, H / 16 % 120]>(k))
        .fetch::<m![L % 16, H / 128 % 15], m![H / 16 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 16, H / 128 % 15], m![H / 16 % 8]>()
        .to_vrf();
    // The fetch unit unpacks and widens the codes to f32 in one Main pass.
    device
        .main
        .begin(w.view().tile::<m![H / 1920], 1, m![L % 16, H / 1920 = 1 # 2, H % 1920]>(k))
        .fetch::<m![L % 16], m![H % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![L % 16, H / 8 % 240], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 16, H / 4 % 480], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &sv)
        .vector_widen_concat::<m![L % 16, H / 8 % 240], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

/// One Up or Gate pass from the loaded rows: both K halves against the resident activation
/// halves, `bf16(bf16(h0 + h1) * global)` (`core16::up_gate_pass`).
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn ug_pass<RING: M, XS: M, YS: M, TT: M, TR: M, const TS: usize>(
    device: &mut Device,
    x0: &gemm16::XRing16<UCl16, RING, TT, TR, H, 480>,
    x1: &gemm16::XRing16<UCl16, RING, TT, TR, H, 480>,
    w: &DmTensor<f4e2m1, Chip, UCl16, m![{ RING }, { XS }, { YS }], m![L % 16, H]>,
    s: &DmTensor<f8e4m3, Chip, UCl16, m![{ RING }, { XS }, { YS }], m![L % 16, H / 16]>,
    g: &VrfTensor<f32, Chip, UCl16, m![{ RING }, { XS }, { YS }], m![1 # 8]>,
) -> Tile16<UCl16, m![{ RING }, { XS }, { YS }], TT, TR> {
    let b0 = ug_decode::<m![{ RING }, { XS }, { YS }]>(device, w, s, 0);
    let t0 = gemm16::to_trf::<UCl16, m![{ RING }, { XS }, { YS }], L, H, 30, 480, 960, m![L % 16, H % 1920]>(device, &b0);
    let h0 = gemm16::contract::<UCl16, RING, XS, YS, TT, TR, L, H, TS, 30, 480, 960>(device, x0, &t0);
    let b1 = ug_decode::<m![{ RING }, { XS }, { YS }]>(device, w, s, 1);
    let t1 = gemm16::to_trf::<UCl16, m![{ RING }, { XS }, { YS }], L, H, 30, 480, 960, m![L % 16, H % 1920]>(device, &b1);
    let h1 = gemm16::contract::<UCl16, RING, XS, YS, TT, TR, L, H, TS, 30, 480, 960>(device, x1, &t1);
    let dot = gemm16::add_round::<UCl16, m![{ RING }, { XS }, { YS }], TT, TR, L>(device, &h0, &h1);
    let y = gemm16::scale_all::<UCl16, m![{ RING }, { XS }, { YS }], TT, TR, L>(device, &dot, g);
    unsafe { y.reshape() }
}

/// The Down rows of the slice with the cluster's whole K half (16 runs of 3,840 B).
fn down_codes<SL: M>(device: &mut Device, w: &HbmTensor<f4e2m1, Chip, m![H, L]>) -> DmTensor<f4e2m1, Chip, DCl16, SL, m![H % 16, L % 7680]> {
    w.to_dm(&mut device.tdma)
}

/// Their block scales (16 runs of 480 B).
fn down_scales<SL: M>(device: &mut Device, s: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>) -> DmTensor<f8e4m3, Chip, DCl16, SL, m![H % 16, L / 16 % 480]> {
    s.to_dm(&mut device.tdma)
}

/// K portion `p` (`L / 1920 % 4`) of the Down rows decoded exactly to bf16.
#[allow(clippy::type_complexity)]
fn down_decode<SL: M>(
    device: &mut Device,
    w: &DmTensor<f4e2m1, Chip, DCl16, SL, m![H % 16, L % 7680]>,
    s: &DmTensor<f8e4m3, Chip, DCl16, SL, m![H % 16, L / 16 % 480]>,
    p: usize,
) -> DmTensor<bf16, Chip, DCl16, SL, m![H % 16, L % 1920]> {
    let sv: VrfTensor<f32, Chip, DCl16, SL, m![H % 16, L / 16 % 120]> = device
        .sub
        .begin(s.view().tile::<m![L / 1920 % 4], 1, m![H % 16, L / 1920 % 4 = 1 # 4, L / 16 % 120]>(p))
        .fetch::<m![H % 16, L / 128 % 15], m![L / 16 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 16, L / 128 % 15], m![L / 16 % 8]>()
        .to_vrf();
    device
        .main
        .begin(w.view().tile::<m![L / 1920 % 4], 1, m![H % 16, L / 1920 % 4 = 1 # 4, L % 1920]>(p))
        .fetch::<m![H % 16], m![L % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![H % 16, L / 8 % 240], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 16, L / 4 % 480], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &sv)
        .vector_widen_concat::<m![H % 16, L / 8 % 240], m![L % 8]>()
        .vector_final()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit()
}

/// One Down pass from the loaded rows: K portion `p` of the cluster's half, the f32 sums.
#[allow(clippy::type_complexity)]
fn down_pass<T: AxisName, RING: M, XS: M, YS: M, TT: M, TR: M, const TS: usize>(
    device: &mut Device,
    act: &HbmTensor<bf16, Chip, m![T, L]>,
    w: &DmTensor<f4e2m1, Chip, DCl16, m![{ RING }, { XS }, { YS }], m![H % 16, L % 7680]>,
    s: &DmTensor<f8e4m3, Chip, DCl16, m![{ RING }, { XS }, { YS }], m![H % 16, L / 16 % 480]>,
    p: usize,
) -> Acc<DCl16, m![{ RING }, { XS }, { YS }], TT, TR, H> {
    let at = act.view().tile::<m![L / 1920 % 4], 1, m![T, L / 7680, L / 1920 % 4 = 1 # 4, L % 1920]>(p);
    let xr: gemm16::XRing16<DCl16, RING, TT, TR, L, 480> = at.to_dm(&mut device.tdma);
    let wb = down_decode::<m![{ RING }, { XS }, { YS }]>(device, w, s, p);
    let wtrf = gemm16::to_trf::<DCl16, m![{ RING }, { XS }, { YS }], H, L, 30, 480, 960, m![H % 16, L % 1920]>(device, &wb);
    gemm16::contract::<DCl16, RING, XS, YS, TT, TR, H, L, TS, 30, 480, 960>(device, &xr, &wtrf)
}

/// Read a global scale once per cluster and broadcast it to all 256 slice VRFs.
fn scalar_bcast<CL: M, SL: M>(device: &mut Device, g: &HbmTensor<f32, Chip, m![1]>) -> VrfTensor<f32, Chip, CL, SL, m![1 # 8]> {
    let one: DmTensor<f32, Chip, CL, m![1 # 256], m![1 # 8]> = g.to_dm(&mut device.tdma);
    let rep: DmTensor<f32, Chip, CL, m![Dummy256], m![1 # 8]> = device
        .main
        .begin(one.view())
        .fetch::<m![1], m![1 # 8]>()
        .switch::<m![Dummy256], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![1], m![1 # 8]>()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rep: DmTensor<f32, Chip, CL, SL, m![1 # 8]> = unsafe { rep.reshape() };
    device.sub.begin(rep.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf()
}

/// The BF16 FFN stage for `T` tokens (`ffn_stage` of `ops_tokenwise` on the stationary core):
///
/// 1. pre-FFN norm, one token per slice; the normalized rows' two K halves into the ring once;
/// 2. Up and Gate: per slice two row groups (`L / 128 % 2`) of 16 rows, each group's codes and
///    block scales loaded with their full K as one run each and decoded per K half from views
///    (fetch table to E4M3, widened to f32 in the same pass, times the block scale), two K halves
///    summed in order, `bf16(bf16(dot) * global)`, then the GeGLU per group; the next group's
///    weights are issued one group ahead;
/// 3. the activation transposed through eight-slice rings into token-major `[T, L]`;
/// 4. Down with K split across clusters (7,680 each, loaded once) and four passes of 1,920, f32
///    sums added in pass order, the cluster swap, `bf16(bf16(dot) * global)`;
/// 5. the post-FFN norm, residual add and layer scalar, one token per slice.
///
/// The data movement is (`ops_core::decode`): full-K runs, activations
/// loaded once, global scales broadcast through the switch from one slice.
///
/// Ring parameters: `URING`, `UXS`, `UYS` (Up/Gate rows' slice axes), `DRING`, `DXS`, `DYS`
/// (Down rows'), `TT`, `TR`, `TS` (tokens).
#[allow(clippy::too_many_arguments)]
pub(crate) fn ffn_stage<
    T: AxisName,
    const HALF: usize,
    const PS: usize,
    PP: Passes,
    URING: M,
    UXS: M,
    UYS: M,
    DRING: M,
    DXS: M,
    DYS: M,
    TT: M,
    TR: M,
    const TS: usize,
>(
    device: &mut Device,
    x: &mut HbmTensor<bf16, Chip, m![T, H]>,
    pre_ff_rms_weight: &HbmTensor<bf16, Chip, m![Rc, H]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
    post_ff_rms_weight: &HbmTensor<bf16, Chip, m![Rc, H]>,
    layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
) {
    // The scheduler orders the DMA by dependencies (the norm's gamma, then Up group 0, whatever the
    // program order). The ~24k cycles before the first contraction are the gamma (7.2k) and Up
    // group 0 (13.5k + 2.3k); loading that group as two K-half tiles to start sooner cost more
    // (+5.6k static: 960 B runs) than it saved.
    let g = rotate::gamma::<T, HALF, PP>(device, pre_ff_rms_weight);
    let u0w = ug_codes::<m![{ URING }, { UXS }, { UYS }]>(device, up_weight_packed, 0);
    let u0s = ug_scales::<m![{ URING }, { UXS }, { UYS }]>(device, up_weight_scale, 0);

    // 1. Pre-FFN norm, then the normalized rows' two K halves into the ring, once.
    let mut h = HbmTensor::<bf16, Chip, m![T, H]>::new();
    PP::normalize_rows::<T, HALF, PS>(device, x, &g, &mut h);
    let x0: gemm16::XRing16<UCl16, URING, TT, TR, H, 480> =
        h.view().tile::<m![H / 1920], 1, m![T, H / 1920 = 1 # 2, H % 1920]>(0).to_dm(&mut device.tdma);
    let x1: gemm16::XRing16<UCl16, URING, TT, TR, H, 480> =
        h.view().tile::<m![H / 1920], 1, m![T, H / 1920 = 1 # 2, H % 1920]>(1).to_dm(&mut device.tdma);
    let g0w = ug_codes::<m![{ URING }, { UXS }, { UYS }]>(device, gate_weight_packed, 0);
    let g0s = ug_scales::<m![{ URING }, { UXS }, { UYS }]>(device, gate_weight_scale, 0);
    let ug = scalar_bcast::<UCl16, m![{ URING }, { UXS }, { UYS }]>(device, up_global_scale);
    let gg = scalar_bcast::<UCl16, m![{ URING }, { UXS }, { UYS }]>(device, gate_global_scale);

    // 2. Up/Gate and the GeGLU per row group, the next group's weights issued ahead.
    let mut act: DmTensor<bf16, Chip, UCl16, m![{ URING }, { UXS }, { UYS }], m![{ TT }, { TR }, L / 128 % 2, L % 16]> = DmTensor::new();
    let u1w = ug_codes::<m![{ URING }, { UXS }, { UYS }]>(device, up_weight_packed, 1);
    let u1s = ug_scales::<m![{ URING }, { UXS }, { UYS }]>(device, up_weight_scale, 1);
    let up = ug_pass::<URING, UXS, UYS, TT, TR, TS>(device, &x0, &x1, &u0w, &u0s, &ug);
    let g1w = ug_codes::<m![{ URING }, { UXS }, { UYS }]>(device, gate_weight_packed, 1);
    let g1s = ug_scales::<m![{ URING }, { UXS }, { UYS }]>(device, gate_weight_scale, 1);
    let gate = ug_pass::<URING, UXS, UYS, TT, TR, TS>(device, &x0, &x1, &g0w, &g0s, &gg);
    geglu16::<UCl16, m![{ URING }, { UXS }, { UYS }], TT, TR>(device, &up, &gate, 0, &mut act);
    let dw = down_codes::<m![{ DRING }, { DXS }, { DYS }]>(device, down_weight_packed);
    let ds = down_scales::<m![{ DRING }, { DXS }, { DYS }]>(device, down_weight_scale);
    let up = ug_pass::<URING, UXS, UYS, TT, TR, TS>(device, &x0, &x1, &u1w, &u1s, &ug);
    let gate = ug_pass::<URING, UXS, UYS, TT, TR, TS>(device, &x0, &x1, &g1w, &g1s, &gg);
    geglu16::<UCl16, m![{ URING }, { UXS }, { UYS }], TT, TR>(device, &up, &gate, 1, &mut act);

    // 3. Token-major activation (`core16::ffn_stage` step 3).
    let act: DmTensor<bf16, Chip, UCl16, m![L / 512 # 32, L / 16 % 8], m![T, L / 128 % 2, L % 16]> = unsafe { act.reshape() };
    let act: DmTensor<bf16, Chip, UCl16, m![L / 512 # 32, T % 8], m![T / 8, L / 128 % 2, L / 16 % 8, L % 16]> = device
        .main
        .begin(act.view())
        .fetch::<m![T / 8, T % 8, L / 128 % 2], m![L % 16]>()
        .switch::<m![L / 512 # 32, T % 8], m![T / 8, L / 128 % 2, L / 16 % 8]>(SwitchConfig::InterTranspose {
            slice1: 8,
            slice0: 1,
            time0: 2,
        })
        .collect::<m![T / 8, L / 128 % 2, L / 16 % 8], m![L % 16]>()
        .commit_trim::<m![L % 16]>()
        .commit();
    let act: DmTensor<bf16, Chip, UCl16, m![L / 512 # 32, T % 8], m![T / 8, L % 256]> = unsafe { act.reshape() };
    let mut act_h = HbmTensor::<bf16, Chip, m![T, L]>::new();
    act.view().to_hbm_view(&mut device.tdma, act_h.view_mut());

    // 4. Down from the loaded rows: four passes per cluster in a fixed order, the cluster swap.
    let p0 = down_pass::<T, DRING, DXS, DYS, TT, TR, TS>(device, &act_h, &dw, &ds, 0);
    let p1 = down_pass::<T, DRING, DXS, DYS, TT, TR, TS>(device, &act_h, &dw, &ds, 1);
    let acc = gemm16::add::<DCl16, m![{ DRING }, { DXS }, { DYS }], TT, TR, H>(device, &p0, &p1);
    let p2 = down_pass::<T, DRING, DXS, DYS, TT, TR, TS>(device, &act_h, &dw, &ds, 2);
    let acc = gemm16::add::<DCl16, m![{ DRING }, { DXS }, { DYS }], TT, TR, H>(device, &acc, &p2);
    let p3 = down_pass::<T, DRING, DXS, DYS, TT, TR, TS>(device, &act_h, &dw, &ds, 3);
    let own = gemm16::add::<DCl16, m![{ DRING }, { DXS }, { DYS }], TT, TR, H>(device, &acc, &p3);
    let other: Acc<DCl16, m![{ DRING }, { DXS }, { DYS }], TT, TR, H> = own.view().cluster_swap().to_dm(&mut device.tdma);
    let dot = gemm16::add_round::<DCl16, m![{ DRING }, { DXS }, { DYS }], TT, TR, H>(device, &own, &other);
    let dg = scalar_bcast::<DCl16, m![{ DRING }, { DXS }, { DYS }]>(device, down_global_scale);
    let y = gemm16::scale_all::<DCl16, m![{ DRING }, { DXS }, { DYS }], TT, TR, H>(device, &dot, &dg);
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
    let mut d = HbmTensor::<bf16, Chip, m![T, H]>::new();
    y.view().to_hbm_view(&mut device.tdma, d.view_mut());

    // 5. Post-FFN norm, residual add, layer scalar.
    let gp = post::vector::<T, HALF, PP>(device, post_ff_rms_weight);
    PP::post_ffn::<T, HALF, PS>(device, &d, &gp, layer_scalar, x);
}
