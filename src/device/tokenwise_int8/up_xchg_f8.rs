//! Static-FP8 Up input exchange, composed with row-owned Down.
//! Original normalized BF16/E4M3 values and sx-then-ws epilogue are retained.
//! Input rows stay in DM and cross clusters once; HBM code/scales roundtrips are removed.
use super::*;
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

fn contract_rows_half<T: AxisName, const HALF: usize, const PS: usize, const P: usize, RING: M, XS: M, YS: M>(
    device: &mut Device,
    x: &XHalf<UCl, T, HALF, PS>,
    w: &WTrf8<UCl, m![{ RING }, { XS }, { YS }], Lu, H, 30, 960, 1920>,
    sx: &VrfTensor<f32, Chip, UCl, m![{ RING }, { XS }, { YS }], m![1 # 8]>,
    ws: &VrfTensor<f32, Chip, UCl, m![{ RING }, { XS }, { YS }], m![Lu / 8 % 2, Lu % 8]>,
    h: usize,
    y: &mut DmTensor<bf16, Chip, UCl, m![{ RING }, { XS }, { YS }], m![T / { PS }, T / { HALF } % 2, T % { HALF }, Lu / 8 % 2, Lu % 8]>,
) {
    device
        .main
        .begin(x.view())
        .fetch::<m![T / { PS }, H / 32 % 30, H / 1920 % 2, H / 960 % 2], m![H % 32]>()
        .switch::<m![{ RING }, { XS }, { YS }], m![T / { PS }, T % { HALF }, H / 32 % 30, H / 1920 % 2, H / 960 % 2]>(SwitchConfig::Broadcast01 {
            slice1: 256,
            slice0: 1,
            time0: 120,
        })
        .collect::<m![T / { PS }, T % { HALF }, H / 32 % 30, H / 1920 % 2, H / 960 % 2], m![H % 32]>()
        .contract_outer::<m![T / { PS }, T % { HALF }, H / 32 % 30, H / 1920 % 2, Lu / 8 % 2], m![H / 960 % 2, H % 32], _, _, _>(w)
        .contract_packet::<m![1]>()
        .contract_time::<m![T / { PS }, T % { HALF }, Lu / 8 % 2]>()
        .contract_lane::<m![T / { PS }, T % { HALF }, Lu / 8 % 2], m![Lu % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![T / { PS }, T % { HALF }, Lu / 8 % 2, Lu / 4 % 2], m![Lu % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), sx)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), ws)
        .vector_widen_concat::<m![T / { PS }, T % { HALF }, Lu / 8 % 2], m![Lu % 8]>()
        .vector_final()
        .cast::<bf16, m![Lu % 8 # 16]>()
        .commit_trim::<m![Lu % 8]>()
        .commit_view(y.view_mut().tile::<m![T / { HALF } % 2], 1, m![T / { PS }, T / { HALF } % 2 = 1 #{!} 2, T % { HALF }, Lu / 8 % 2, Lu % 8]>(h));
}

type Y<T,const HALF:usize,const PS:usize,R,X,Z> = DmTensor<bf16,Chip,UCl,USl<R,X,Z>,m![T / { PS },T / { HALF } % 2,T % { HALF },Lu / 8 % 2,Lu % 8]>;
fn weights<URING:M,UXS:M,UYS:M>(device:&mut Device,wug:&HbmTensor<f8e4m3,Chip,m![Lp,H]>,p:usize)->DmTensor<f8e4m3,Chip,UCl,USl<URING,UXS,UYS>,m![Lu %16,H]> {
    let wt = wug.view().tile::<m![Lp / 16 % 4], 1, m![Lp / 64, Lp / 16 % 4 = 1 # 4, Lp % 16, H]>(p);
    let wdm: DmTensor<f8e4m3, Chip, m![Lp / 512 % 2], m![Lp / 1024, Lp / 256 % 2, Lp / 64 % 4], m![Lp % 16, H]> = wt.to_dm(&mut device.tdma);
    // `Lp / 1024 < 30` is `Lu / 1024`; rings 30 and 31 are `Lu`'s padding (`gl_up_scales`' GSl -> USl relabel).
    let wdm: DmTensor<f8e4m3, Chip, UCl, USl<URING, UXS, UYS>, m![Lu % 16, H]> = unsafe { wdm.reshape() };
    wdm
}
fn pass<T:AxisName,const HALF:usize,const PS:usize,const P:usize,URING:M,UXS:M,UYS:M,TT:M,TR:M>(
 device:&mut Device,
 x0:&XHalf<UCl,T,HALF,PS>, x1:&XHalf<UCl,T,HALF,PS>,
 sx:&VrfTensor<f32,Chip,UCl,USl<URING,UXS,UYS>,m![1 # 8]>,
 wdm:&DmTensor<f8e4m3,Chip,UCl,USl<URING,UXS,UYS>,m![Lu %16,H]>,
 sd:&DmTensor<f32,Chip,UCl,USl<URING,UXS,UYS>,m![Lu /16 %4,Lu /8 %2,Lu %8]>,p:usize,
 act:&mut DmTensor<bf16,Chip,UCl,USl<URING,UXS,UYS>,m![Lu /32 %2,{TT},{TR},Lu /16 %2,Lu %8]>,
) {
    let w = to_trf8::<UCl, USl<URING, UXS, UYS>, Lu, H, 30, 960, 1920, m![Lu % 16, H]>(device, wdm);
    let ws: VrfTensor<f32, Chip, UCl, USl<URING, UXS, UYS>, m![Lu / 8 % 2, Lu % 8]> = device
        .sub
        .begin(sd.view().tile::<m![Lu / 16 % 4], 1, m![Lu / 16 % 4 = 1 # 4, Lu / 8 % 2, Lu % 8]>(p))
        .fetch::<m![Lu / 8 % 2], m![Lu % 8]>()
        .collect::<m![Lu / 8 % 2], m![Lu % 8]>()
        .to_vrf();

 let mut y=Y::<T,HALF,PS,URING,UXS,UYS>::new();
 contract_rows_half::<T,HALF,PS,P,URING,UXS,UYS>(device,x0,&w,sx,&ws,0,&mut y);
 contract_rows_half::<T,HALF,PS,P,URING,UXS,UYS>(device,x1,&w,sx,&ws,1,&mut y);
 let y:DmTensor<bf16,Chip,UCl,USl<URING,UXS,UYS>,m![{TT},{TR},Lu /8 %2,Lu %8]>=unsafe{y.reshape()};
 gl_geglu::<TT,TR,URING,UXS,UYS>(device,&y,p,act);
}

// Separate constructors prevent dormant P=1 branches from reaching MIR/VISA.
pub(super) struct TwoInputs;
pub(super) trait InputPasses {
 fn fill<T:AxisName,const HALF:usize,const PS:usize,const P:usize>(
  device:&mut Device,sv:&VrfTensor<f32,Chip,TCl<T,HALF>,TSl<T,HALF>,m![1 #8]>,
  xd:[&Row<bf16,T,HALF,m![Kh]>;2],g:&Row<bf16,T,HALF,m![Kh]>,xp:&mut XPair<T,HALF,PS>);
}
fn input<T:AxisName,const HALF:usize,const PS:usize,const P:usize>(
 device:&mut Device,sv:&VrfTensor<f32,Chip,TCl<T,HALF>,TSl<T,HALF>,m![1 #8]>,
 xd:&Row<bf16,T,HALF,m![Kh]>,g:&Row<bf16,T,HALF,m![Kh]>,p:usize,xp:&mut XPair<T,HALF,PS>) {
 let n=rotate2::normalize_pair::<T,HALF>(device,xd,g);
 let q=quantize::<T,HALF,Kh>(device,&n,sv);
 let q:Row<f8e4m3,T,HALF,m![H]>=unsafe{q.reshape()};
 pair_rows_pass::<T,HALF,PS,P>(device,&q,p,xp);
}
impl InputPasses for TwoInputs {
 fn fill<T:AxisName,const HALF:usize,const PS:usize,const P:usize>(device:&mut Device,sv:&VrfTensor<f32,Chip,TCl<T,HALF>,TSl<T,HALF>,m![1 #8]>,xd:[&Row<bf16,T,HALF,m![Kh]>;2],g:&Row<bf16,T,HALF,m![Kh]>,xp:&mut XPair<T,HALF,PS>) {
  const{assert!(P==2);}
  input::<T,HALF,PS,P>(device,sv,xd[0],g,0,xp);
  input::<T,HALF,PS,P>(device,sv,xd[1],g,1,xp);
 }
}

/// Same producer boundary and operation order as rotate2::row_quantize_static_f8;
/// callers provide the already loaded identical scale, shared by both input passes.
fn quantize<T:AxisName,const HALF:usize,Q:AxisName>(device:&mut Device,
 y:&Row<bf16,T,HALF,m![Q]>,sv:&VrfTensor<f32,Chip,TCl<T,HALF>,TSl<T,HALF>,m![1 #8]>) -> Row<f8e4m3,T,HALF,m![Q]> {
    let q: Row<f8e4m3, T, HALF, m![Q]> = device
        .main
        .begin(y.view())
        .fetch::<m![Q / 16], m![Q % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Q / 8], m![Q % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Q / 4], m![Q % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, sv)
        .vector_widen_concat::<m![Q / 8], m![Q % 8]>()
        .vector_clip(ClipBinaryOpF32::Min, 448f32)
        .vector_clip(ClipBinaryOpF32::Max, -448f32)
        .vector_final()
        .cast::<f8e4m3, m![Q % 8 # 32]>()
        .commit_trim::<m![Q % 8]>()
        .commit();
    q
}

// ---- upgate: pass-pair Down-input epilogue for the served pair route ----

/// The Up/Gate body ending in Down's input: both pass pairs' E4M3 codes in HBM (`codes_pair_split8`'s layout and
/// values), pair 0's codes made right after pass 1 (its GeGLU is complete) and pair 1's after pass 3.
/// Exact: every output element is the same arithmetic as `act_codes_static` + `codes_pair_split8` on the
/// four-pass act; only the time and the tensor unit of the two epilogue halves change.
pub(super) fn up_pp<I:InputPasses,T:AxisName,const HALF:usize,const PS:usize,const P:usize,URING:M,UXS:M,UYS:M,TT:M,TR:M>(
 device:&mut Device,sigma:&HbmTensor<f32,Chip,m![1 #8]>,sigma_down:&HbmTensor<f32,Chip,m![1 #8]>,
 xd:[&Row<bf16,T,HALF,m![Kh]>;2],g:&Row<bf16,T,HALF,m![Kh]>,
 wug:&HbmTensor<f8e4m3,Chip,m![Lp,H]>,wug_scale:&HbmTensor<f32,Chip,m![Lu]>,
) -> (HbmTensor<f8e4m3,Chip,m![Kq4 / 2,T,Kd]>,HbmTensor<f8e4m3,Chip,m![Kq4 / 2,T,Kd]>) {
 const{assert!(P==1 || P==2);assert!(PS==2*HALF);assert!(T::SIZE==P*PS);}
 let w0=weights::<URING,UXS,UYS>(device,wug,0);
 let scale:Row<f32,T,HALF,m![1 #8]>=sigma.to_dm(&mut device.tdma);
 let sv:VrfTensor<f32,Chip,TCl<T,HALF>,TSl<T,HALF>,m![1 #8]>=device.sub.begin(scale.view()).fetch::<m![1],m![1 #8]>().collect::<m![1],m![1 #8]>().to_vrf();
 let mut xp=XPair::<T,HALF,PS>::new();
 I::fill::<T,HALF,PS,P>(device,&sv,xd,g,&mut xp);
 let (x0,x1)=split_halves::<UCl,T,HALF,PS>(device,&xp);
 let sd=gl_up_scales::<URING,UXS,UYS>(device,wug_scale);
 let s:DmTensor<f32,Chip,UCl,USl<URING,UXS,UYS>,m![1 #8]>=unsafe{scale.reshape()};
 let sx:VrfTensor<f32,Chip,UCl,USl<URING,UXS,UYS>,m![1 #8]>=device.sub.begin(s.view()).fetch::<m![1],m![1 #8]>().collect::<m![1],m![1 #8]>().to_vrf();
 let mut act:DmTensor<bf16,Chip,UCl,USl<URING,UXS,UYS>,m![Lu /32 %2,{TT},{TR},Lu /16 %2,Lu %8]>=DmTensor::new();
 pass::<T,HALF,PS,P,URING,UXS,UYS,TT,TR>(device,&x0,&x1,&sx,&w0,&sd,0,&mut act);
 let w1=weights::<URING,UXS,UYS>(device,wug,1);
 pass::<T,HALF,PS,P,URING,UXS,UYS,TT,TR>(device,&x0,&x1,&sx,&w1,&sd,1,&mut act);
 // Pass pair 0 is complete: its codes and their ring transpose + HBM store run beside passes 2-3 (Pair0's units).
 let dsd:DmTensor<f32,Chip,UCl,GSl,m![1 #8]>=sigma_down.to_dm(&mut device.tdma);
 let dsv:VrfTensor<f32,Chip,UCl,GSl,m![1 #8]>=device.sub.begin(dsd.view()).fetch::<m![1],m![1 #8]>().collect::<m![1],m![1 #8]>().to_vrf();
 let q0={
  let a:DmTensorView<'_,bf16,Chip,UCl,GSl,m![Gpp,T,Kb %16]>=unsafe{act.view().reshape()};
  Pair0::quantize::<T>(device,a.tile::<m![Gpp],1,m![Gpp = 1 # 2,T,Kb %16]>(0),&dsv)
 };
 let dcodes0=Pair0::split::<T>(device,&q0);
 let w2=weights::<URING,UXS,UYS>(device,wug,2);
 pass::<T,HALF,PS,P,URING,UXS,UYS,TT,TR>(device,&x0,&x1,&sx,&w2,&sd,2,&mut act);
 let w3=weights::<URING,UXS,UYS>(device,wug,3);
 pass::<T,HALF,PS,P,URING,UXS,UYS,TT,TR>(device,&x0,&x1,&sx,&w3,&sd,3,&mut act);
 // Pass pair 1: Down's first contraction (pair 0's codes) needs none of it, so its transpose runs beside Down (Pair1's units).
 let q1={
  let a:DmTensorView<'_,bf16,Chip,UCl,GSl,m![Gpp,T,Kb %16]>=unsafe{act.view().reshape()};
  Pair1::quantize::<T>(device,a.tile::<m![Gpp],1,m![Gpp = 1 # 2,T,Kb %16]>(1),&dsv)
 };
 let dcodes1=Pair1::split::<T>(device,&q1);
 (dcodes0,dcodes1)
}

/// Where a pass pair's code epilogue runs: the quantization (vector engine) and the ring transpose + store (no vector
/// engine) each on Main or Sub. Trait dispatch, so only the chosen unit's commands reach MIR/VISA.
trait CodesCtx {
 fn quantize<T:AxisName>(device:&mut Device,a:DmTensorView<'_,bf16,Chip,UCl,GSl,m![Gpp = 1 # 2,T,Kb %16]>,
  sv:&VrfTensor<f32,Chip,UCl,GSl,m![1 #8]>) -> DmTensor<f8e4m3,Chip,UCl,GSl,m![T,Kb %16]>;
 fn split<T:AxisName>(device:&mut Device,q:&DmTensor<f8e4m3,Chip,UCl,GSl,m![T,Kb %16]>) -> HbmTensor<f8e4m3,Chip,m![Kq4 / 2,T,Kd]>;
}
/// `act_codes_static`'s pipeline, operation for operation, on one pass pair and tensor unit `$tu`.
macro_rules! pair_quantize {($device:ident,$tu:ident,$a:ident,$sv:ident) => {
 $device.$tu.begin($a)
  .fetch::<m![T],m![Kb %16]>()
  .fetch_cast::<f32>()
  .collect::<m![T,Kb /8 %2],m![Kb %8]>()
  .vector_init()
  .vector_intra_slice_tag(TagMode::Zero)
  .vector_narrow_split::<m![T,Kb /8 %2,Kb /4 %2],m![Kb %4]>()
  .vector_fp_binary(FpBinaryOp::DivF,$sv)
  .vector_widen_concat::<m![T,Kb /8 %2],m![Kb %8]>()
  .vector_clip(ClipBinaryOpF32::Min,448f32)
  .vector_clip(ClipBinaryOpF32::Max,-448f32)
  .vector_final()
  .cast::<f8e4m3,m![Kb %8 # 32]>()
  .commit_trim::<m![Kb %8]>()
  .commit()
}}
/// `codes_pair_split8`'s sixteen-slice ring transpose into GL's `GItSl` and its HBM store, on tensor unit `$tu`.
macro_rules! pair_split {($device:ident,$tu:ident,$q:ident) => {{
 let t:DmTensor<f8e4m3,Chip,UCl,GItSl<T>,m![T / 16,Lu / 1024 % 2,Lu / 256 % 2,Kb]>=$device.$tu.begin($q.view())
  .fetch::<m![T / 16,T % 16],m![Kb % 16]>()
  .switch::<GItSl<T>,m![T / 16,Lu / 1024 % 2,Lu / 256 % 2,Kb / 16]>(SwitchConfig::InterTranspose{slice1:16,slice0:1,time0:1})
  .collect::<m![T / 16,Lu / 1024 % 2,Lu / 256 % 2,Kb / 16],m![Kb % 16 # 32]>()
  .commit_trim::<m![Kb % 16]>()
  .commit();
 let t:DmTensor<f8e4m3,Chip,m![Kq4 / 2],m![Kd / 256 # 16,T % 16],m![T / 16,Kd % 256]>=unsafe{t.reshape()};
 t.to_hbm(&mut $device.tdma)
}}}
/// Pass pair 0: codes on Main (the vector engine is Main's through passes 2-3), transpose + store on Sub.
struct Pair0;
/// Pass pair 1: codes on Main right after pass 3's GeGLU (Sub-context vector work does not lower: Opt 0.9.0 LIR
/// `no entry found for key` at npu-isa sfr_impl.rs:868), transpose + store on Sub beside Down's first (pair 0)
/// contraction, which needs none of pair 1's codes.
struct Pair1;
impl CodesCtx for Pair0 {
 fn quantize<T:AxisName>(device:&mut Device,a:DmTensorView<'_,bf16,Chip,UCl,GSl,m![Gpp = 1 # 2,T,Kb %16]>,
  sv:&VrfTensor<f32,Chip,UCl,GSl,m![1 #8]>) -> DmTensor<f8e4m3,Chip,UCl,GSl,m![T,Kb %16]> { pair_quantize!(device,main,a,sv) }
 fn split<T:AxisName>(device:&mut Device,q:&DmTensor<f8e4m3,Chip,UCl,GSl,m![T,Kb %16]>) -> HbmTensor<f8e4m3,Chip,m![Kq4 / 2,T,Kd]> { pair_split!(device,sub,q) }
}
impl CodesCtx for Pair1 {
 fn quantize<T:AxisName>(device:&mut Device,a:DmTensorView<'_,bf16,Chip,UCl,GSl,m![Gpp = 1 # 2,T,Kb %16]>,
  sv:&VrfTensor<f32,Chip,UCl,GSl,m![1 #8]>) -> DmTensor<f8e4m3,Chip,UCl,GSl,m![T,Kb %16]> { pair_quantize!(device,main,a,sv) }
 fn split<T:AxisName>(device:&mut Device,q:&DmTensor<f8e4m3,Chip,UCl,GSl,m![T,Kb %16]>) -> HbmTensor<f8e4m3,Chip,m![Kq4 / 2,T,Kd]> { pair_split!(device,sub,q) }
}

#[cfg(feature = "compile-carried")]
/// Three 512-row passes with the current pass-pair epilogue.
pub(super) fn up_pp3<T:AxisName,const HALF:usize,const PS:usize,const P:usize,URING:M,UXS:M,UYS:M,TT:M,TR:M>(
 device:&mut Device,sigma:&HbmTensor<f32,Chip,m![1 #8]>,sigma_down:&HbmTensor<f32,Chip,m![1 #8]>,
 xd:[&Row<bf16,T,HALF,m![Kh]>;3],g:&Row<bf16,T,HALF,m![Kh]>,
 wug:&HbmTensor<f8e4m3,Chip,m![Lp,H]>,wug_scale:&HbmTensor<f32,Chip,m![Lu]>,
) -> (HbmTensor<f8e4m3,Chip,m![Kq4 / 2,T,Kd]>,HbmTensor<f8e4m3,Chip,m![Kq4 / 2,T,Kd]>) {
 const{assert!(P==3);assert!(PS==2*HALF);assert!(T::SIZE==P*PS);}
 let w0=weights::<URING,UXS,UYS>(device,wug,0);
 let scale:Row<f32,T,HALF,m![1 #8]>=sigma.to_dm(&mut device.tdma);
 let sv:VrfTensor<f32,Chip,TCl<T,HALF>,TSl<T,HALF>,m![1 #8]>=device.sub.begin(scale.view()).fetch::<m![1],m![1 #8]>().collect::<m![1],m![1 #8]>().to_vrf();
 let mut xp=XPair::<T,HALF,PS>::new();
 input::<T,HALF,PS,P>(device,&sv,xd[0],g,0,&mut xp);
 input::<T,HALF,PS,P>(device,&sv,xd[1],g,1,&mut xp);
 input::<T,HALF,PS,P>(device,&sv,xd[2],g,2,&mut xp);
 let (x0,x1)=split_halves::<UCl,T,HALF,PS>(device,&xp);
 let sd=gl_up_scales::<URING,UXS,UYS>(device,wug_scale);
 let s:DmTensor<f32,Chip,UCl,USl<URING,UXS,UYS>,m![1 #8]>=unsafe{scale.reshape()};
 let sx:VrfTensor<f32,Chip,UCl,USl<URING,UXS,UYS>,m![1 #8]>=device.sub.begin(s.view()).fetch::<m![1],m![1 #8]>().collect::<m![1],m![1 #8]>().to_vrf();
 let mut act:DmTensor<bf16,Chip,UCl,USl<URING,UXS,UYS>,m![Lu /32 %2,{TT},{TR},Lu /16 %2,Lu %8]>=DmTensor::new();
 pass::<T,HALF,PS,P,URING,UXS,UYS,TT,TR>(device,&x0,&x1,&sx,&w0,&sd,0,&mut act);
 let w1=weights::<URING,UXS,UYS>(device,wug,1);
 pass::<T,HALF,PS,P,URING,UXS,UYS,TT,TR>(device,&x0,&x1,&sx,&w1,&sd,1,&mut act);
 // Pass pair 0 is complete: its codes and their ring transpose + HBM store run beside passes 2-3 (Pair0's units).
 let dsd:DmTensor<f32,Chip,UCl,GSl,m![1 #8]>=sigma_down.to_dm(&mut device.tdma);
 let dsv:VrfTensor<f32,Chip,UCl,GSl,m![1 #8]>=device.sub.begin(dsd.view()).fetch::<m![1],m![1 #8]>().collect::<m![1],m![1 #8]>().to_vrf();
 let q0={
  let a:DmTensorView<'_,bf16,Chip,UCl,GSl,m![Gpp,T,Kb %16]>=unsafe{act.view().reshape()};
  Pair0::quantize::<T>(device,a.tile::<m![Gpp],1,m![Gpp = 1 # 2,T,Kb %16]>(0),&dsv)
 };
 let dcodes0=Pair0::split::<T>(device,&q0);
 let w2=weights::<URING,UXS,UYS>(device,wug,2);
 pass::<T,HALF,PS,P,URING,UXS,UYS,TT,TR>(device,&x0,&x1,&sx,&w2,&sd,2,&mut act);
 let w3=weights::<URING,UXS,UYS>(device,wug,3);
 pass::<T,HALF,PS,P,URING,UXS,UYS,TT,TR>(device,&x0,&x1,&sx,&w3,&sd,3,&mut act);
 // Pass pair 1: Down's first contraction (pair 0's codes) needs none of it, so its transpose runs beside Down (Pair1's units).
 let q1={
  let a:DmTensorView<'_,bf16,Chip,UCl,GSl,m![Gpp,T,Kb %16]>=unsafe{act.view().reshape()};
  Pair1::quantize::<T>(device,a.tile::<m![Gpp],1,m![Gpp = 1 # 2,T,Kb %16]>(1),&dsv)
 };
 let dcodes1=Pair1::split::<T>(device,&q1);
 (dcodes0,dcodes1)
}

// The FP8 pass-pair epilogue converts activations to the packed Down layout.
pub(crate) fn pair0_codes<T:AxisName>(device:&mut Device,a:DmTensorView<'_,bf16,Chip,UCl,GSl,m![Gpp=1 #2,T,Kb%16]>,sv:&VrfTensor<f32,Chip,UCl,GSl,m![1 #8]>)->HbmTensor<f8e4m3,Chip,m![Kq4/2,T,Kd]> {
 let q=Pair0::quantize::<T>(device,a,sv);Pair0::split::<T>(device,&q)
}
pub(crate) fn pair1_codes<T:AxisName>(device:&mut Device,a:DmTensorView<'_,bf16,Chip,UCl,GSl,m![Gpp=1 #2,T,Kb%16]>,sv:&VrfTensor<f32,Chip,UCl,GSl,m![1 #8]>)->HbmTensor<f8e4m3,Chip,m![Kq4/2,T,Kd]> {
 let q=Pair1::quantize::<T>(device,a,sv);Pair1::split::<T>(device,&q)
}
