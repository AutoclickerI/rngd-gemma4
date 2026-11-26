//! Static FP8 Q exchange: identical norm/cast and sx-then-ws epilogue, on-chip token halves.
use crate::device::tokenwise_int8::rotate2;
use super::*;
use crate::device::tokenwise_int8::ffn_fast::fp8c;
type Row<D,T,const HALF:usize,E> = DmTensor<D,Chip,TCl<T,HALF>,TSl<T,HALF>,E>;
use crate::axes::Dummy2 as Xh;

/// Both passes store the local code rows at `Xh = 0` and the rows from `cluster_swap`
/// at `Xh = 1`. On cluster `c`, slot `k` holds token half `c ^ k`.
type XPair<T, const HALF: usize, const PS: usize> = Row<f8e4m3, T, HALF, m![Xh, T / { PS }, H]>;
/// Token half `h` of both passes in every cluster (`XPair`'s slot `h ^ c` on cluster `c`, by a Sub-context
/// `asymmetric_cluster_slice`), typed as what it holds: the same tokens in both clusters, under the consuming GEMM's
/// cluster axis `CL`; element `[T / PS, H]`, slice `T % HALF`.
type XHalf<CL, T, const HALF: usize, const PS: usize> = DmTensor<f8e4m3, Chip, CL, TSl<T, HALF>, m![T / { PS }, H]>;

/// One pass's own code rows `q` into slot 0 of pass `p` and the other cluster's into slot 1: one DM->DM
/// `cluster_swap` into a fresh tensor (the only exchange bytes on the DMA queue; a swap straight into a slot of
/// `x` fails the 0.8.1 redistribute verifier), and two short Main copies.
fn pair_rows_pass<T: AxisName, const HALF: usize, const PS: usize, const P: usize>(
    device: &mut Device,
    q: &Row<f8e4m3, T, HALF, m![H]>,
    p: usize,
    x: &mut XPair<T, HALF, PS>,
) {
    let o: Row<f8e4m3, T, HALF, m![H]> = q.view().cluster_swap().to_dm(&mut device.tdma);
    copy_rows_into::<T, HALF, PS, P>(device, q, 0, p, x);
    copy_rows_into::<T, HALF, PS, P>(device, &o, 1, p, x);
}

/// Code rows `r` into slot `k`, pass `p` of `x` (a short Main copy).
fn copy_rows_into<T: AxisName, const HALF: usize, const PS: usize, const P: usize>(
    device: &mut Device,
    r: &Row<f8e4m3, T, HALF, m![H]>,
    k: usize,
    p: usize,
    x: &mut XPair<T, HALF, PS>,
) {
    device
        .main
        .begin(r.view())
        .fetch::<m![H / 32], m![H % 32]>()
        .collect::<m![H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit_view(
            x.view_mut()
                .tile::<m![Xh], 1, m![Xh = 1 #{!} 2, T / { PS }, H]>(k)
                .tile::<m![T / { PS }], 1, m![Xh = 1 #{!} 2, T / { PS } = 1 #{!} { P }, H]>(p),
        );
}

/// Token halves 0 and 1 of every pass in every cluster: cluster `c` takes slot `h ^ c` of `x` for half `h`.
fn split_halves<CL: M, T: AxisName, const HALF: usize, const PS: usize>(
    device: &mut Device,
    x: &XPair<T, HALF, PS>,
) -> (XHalf<CL, T, HALF, PS>, XHalf<CL, T, HALF, PS>) {
    let x0: Row<f8e4m3, T, HALF, m![T / { PS }, H]> = x.asymmetric_cluster_slice::<m![Xh], m![T / { PS }, H]>(&mut device.sub, &[0, 1]);
    let x1: Row<f8e4m3, T, HALF, m![T / { PS }, H]> = x.asymmetric_cluster_slice::<m![Xh], m![T / { PS }, H]>(&mut device.sub, &[1, 0]);
    (unsafe { x0.reshape() }, unsafe { x1.reshape() })
}

/// `contract_ws_rows_half` for pass `p` only: token half `h` of pass `p`, committed into half `h` of the pass's `y`.
#[allow(clippy::type_complexity)]
fn contract_ws_rows_pass_half<T: AxisName, const HALF: usize, const PS: usize, const P: usize, RING: M, XS: M, YS: M>(
    device: &mut Device,
    x: &XHalf<QkvCl, T, HALF, PS>,
    w: &fp8c::WTrf8<QkvCl, m![{ RING }, { XS }, { YS }], Nq, H, 30, 960, 1920>,
    sx: &VrfTensor<f32, Chip, QkvCl, m![{ RING }, { XS }, { YS }], m![1 # 8]>,
    ws: &VrfTensor<f32, Chip, QkvCl, m![{ RING }, { XS }, { YS }], m![Nq / 8 % 2, Nq % 8]>,
    p: usize,
    h: usize,
    y: &mut DmTensor<bf16, Chip, QkvCl, m![{ RING }, { XS }, { YS }], m![T / { HALF } % 2, T % { HALF }, Nq / 8 % 2, Nq % 8]>,
) {
    const { assert!(HALF == 64); }
    let mut ring: DmTensor<f8e4m3, Chip, QkvCl, m![T % { HALF }, H / 960 % 4], m![H % 960]> = DmTensor::new();
    x.view().tile::<m![T / { PS }], 1, m![T / { PS } = 1 # { P }, H]>(p)
        .to_dm_view(&mut device.tdma, ring.view_mut());
    device
        .main
        .begin(ring.view())
        .fetch::<m![H / 32 % 30], m![H % 32]>()
        .switch::<m![{ RING }, { XS }, { YS }], m![T % { HALF }, H / 32 % 30, H / 1920 % 2, H / 960 % 2]>(SwitchConfig::Broadcast01 {
            slice1: 64,
            slice0: 4,
            time0: 30,
        })
        .collect::<m![T % { HALF }, H / 32 % 30, H / 1920 % 2, H / 960 % 2], m![H % 32]>()
        .contract_outer::<m![T % { HALF }, H / 32 % 30, H / 1920 % 2, Nq / 8 % 2], m![H / 960 % 2, H % 32], _, _, _>(w)
        .contract_packet::<m![1]>()
        .contract_time::<m![T % { HALF }, Nq / 8 % 2]>()
        .contract_lane::<m![T % { HALF }, Nq / 8 % 2], m![Nq % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![T % { HALF }, Nq / 8 % 2, Nq / 4 % 2], m![Nq % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), sx)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), ws)
        .vector_widen_concat::<m![T % { HALF }, Nq / 8 % 2], m![Nq % 8]>()
        .vector_final()
        .cast::<bf16, m![Nq % 8 # 16]>()
        .commit_trim::<m![Nq % 8]>()
        .commit_view(y.view_mut().tile::<m![T / { HALF } % 2], 1, m![T / { HALF } % 2 = 1 #{!} 2, T % { HALF }, Nq / 8 % 2, Nq % 8]>(h));
}

/// One pass's projection output `y` (both token halves), in the contraction's layout.
type YPass<T, const HALF: usize, RING, XS, YS> = DmTensor<bf16, Chip, QkvCl, m![{ RING }, { XS }, { YS }], m![T / { HALF } % 2, T % { HALF }, Nq / 8 % 2, Nq % 8]>;

/// A pass's `y`: the 8-slice transpose and the store into the pass's own raw tensor.
#[allow(clippy::type_complexity)]
fn qkv_pass_store<T: AxisName, const HALF: usize, const PS: usize, RING: M, XS: M, YS: M>(device: &mut Device, y: YPass<T, HALF, RING, XS, YS>) -> HbmTensor<bf16, Chip, m![T % { PS }, Nq]> {
    let y: DmTensor<bf16, Chip, QkvCl, m![Nq / 128 % 32, Nq / 16 % 8], m![T % { PS }, Nq % 16]> = unsafe { y.reshape() };
    let y: DmTensor<bf16, Chip, QkvCl, m![Nq / 128 % 32, T % 8], m![T % { PS } / 8, Nq / 16 % 8, Nq % 16]> = device
        .main
        .begin(y.view())
        .fetch::<m![T % { PS } / 8, T % 8], m![Nq % 16]>()
        .switch::<m![Nq / 128 % 32, T % 8], m![T % { PS } / 8, Nq / 16 % 8]>(SwitchConfig::InterTranspose {
            slice1: 8,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![T % { PS } / 8, Nq / 16 % 8], m![Nq % 16]>()
        .commit_trim::<m![Nq % 16]>()
        .commit();
    let mut raw = HbmTensor::<bf16, Chip, m![T % { PS }, Nq]>::new();
    y.view().to_hbm_view(&mut device.tdma, raw.view_mut());
    raw
}

/// Pass `p`'s raw projection rows: both token halves contracted, the 8-slice transpose, the store.
#[allow(clippy::type_complexity)]
fn qkv_pass_raw<T: AxisName, const HALF: usize, const PS: usize, const P: usize, RING: M, XS: M, YS: M>(
    device: &mut Device,
    xh: (&XHalf<QkvCl, T, HALF, PS>, &XHalf<QkvCl, T, HALF, PS>),
    w: &fp8c::WTrf8<QkvCl, m![{ RING }, { XS }, { YS }], Nq, H, 30, 960, 1920>,
    sx: &VrfTensor<f32, Chip, QkvCl, m![{ RING }, { XS }, { YS }], m![1 # 8]>,
    ws: &VrfTensor<f32, Chip, QkvCl, m![{ RING }, { XS }, { YS }], m![Nq / 8 % 2, Nq % 8]>,
    p: usize,
) -> HbmTensor<bf16, Chip, m![T % { PS }, Nq]> {
    let mut y: YPass<T, HALF, RING, XS, YS> = DmTensor::new();
    contract_ws_rows_pass_half::<T, HALF, PS, P, RING, XS, YS>(device, xh.0, w, sx, ws, p, 0, &mut y);
    contract_ws_rows_pass_half::<T, HALF, PS, P, RING, XS, YS>(device, xh.1, w, sx, ws, p, 1, &mut y);
    qkv_pass_store::<T, HALF, PS, RING, XS, YS>(device, y)
}

/// One-pass Sliding Q input exchange (Tok512); no second pass or cross-pass overlap.
pub(crate) fn stage1<
    T: AxisName,
    const HALF: usize,
    const PS2: usize,
    PP: Passes + HeadsRowsEarlyPasses8,
    RING: M,
    XS: M,
    YS: M,
    TT: M,
    TR: M,
    const TS: usize,
>(
    device: &mut Device,
    sigma: &HbmTensor<f32, Chip, m![1 # 8]>,
    xd: &DmTensor<bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![Kh]>,
    input_rms_weight_signed: &HbmTensor<bf16, Chip, m![Rc, H]>,
    eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>,
    wq: &HbmTensor<f8e4m3, Chip, m![Nq, H]>,
    ws: &HbmTensor<f32, Chip, m![Nq]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    rope_offsets: &HbmTensor<i32, Chip, m![T]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
    rows_s: &HbmTensor<i32, Chip, m![T, Hh]>,
    stage_s: &HbmTensor<i32, Chip, m![T]>,
    qin_s: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
    k_n: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
    v_n: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
) {
    let _ = eye;
    let wd: DmTensor<f8e4m3, Chip, QkvCl, m![{ RING }, { XS }, { YS }], m![Nq % 16, H]> = wq.to_dm(&mut device.tdma);
    let g = rotate2::gamma_nat_ag2::<T, HALF>(device, input_rms_weight_signed);
    let q = norm_codes::<T, HALF>(device, sigma, xd, &g);
    let mut xp: XPair<T, HALF, PS2> = DmTensor::new();
    pair_rows_pass::<T, HALF, PS2, 1>(device, &q, 0, &mut xp);
    let (x0, x1) = split_halves::<QkvCl, T, HALF, PS2>(device, &xp);
    let w = fp8c::to_trf8_main::<QkvCl, m![{ RING }, { XS }, { YS }], Nq, H, 30, 960, 1920, m![Nq % 16, H]>(device, &wd);
    let wsv = gemm::col_scales::<QkvCl, m![{ RING }, { XS }, { YS }], Nq, m![Nq]>(device, ws.view());
    let sx = scalar_scale::<RING, XS, YS>(device, sigma);
    let r0 = qkv_pass_raw::<T, HALF, PS2, 1, RING, XS, YS>(device, (&x0, &x1), &w, &sx, &wsv, 0);
    let k_gamma = super::super::heads::table_vrf::<T, HALF>(device, k_rms_weight);
    let q_gamma = super::super::heads::table_vrf::<T, HALF>(device, q_rms_weight);
    super::super::heads::heads_pass_rows_early8_raw::<T, HALF, PS2, 1, 0>(device, &r0, &q_gamma, &k_gamma, rope_offsets, cos, sin, rows_s, stage_s, qin_s, k_n, v_n);
}

fn norm_codes<T: AxisName,const HALF:usize>(device:&mut Device,sigma:&HbmTensor<f32,Chip,m![1 # 8]>,xd:&Row<bf16,T,HALF,m![Kh]>,g:&Row<bf16,T,HALF,m![Kh]>)->Row<f8e4m3,T,HALF,m![H]> {
    let h=rotate2::normalize_pair::<T,HALF>(device,xd,g);
    let (_,q)=rotate2::row_quantize_static_f8::<T,HALF,Kh>(device,sigma,&h);
    unsafe {q.reshape()}
}
fn scalar_scale<RING:M,XS:M,YS:M>(device:&mut Device,sigma:&HbmTensor<f32,Chip,m![1 # 8]>)->VrfTensor<f32,Chip,QkvCl,m![{RING},{XS},{YS}],m![1 # 8]> {
    let s:DmTensor<f32,Chip,QkvCl,m![{RING},{XS},{YS}],m![1 # 8]>=sigma.to_dm(&mut device.tdma);
    device.sub.begin(s.view()).fetch::<m![1],m![1 # 8]>().collect::<m![1],m![1 # 8]>().to_vrf()
}
