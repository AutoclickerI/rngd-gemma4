//! Row-partitioned Gate/Up using packed INT4 or static E4M3.
//! Both native packing variants reuse the gathered activations and avoid reloading
//! weights per row group. Padding in FP8 weights is explicit zero, as in main.
#![cfg(feature = "compile-carried")]

use super::*;

/// One straight-line pure pass over literal row groups: no `for` region, so the scheduler overlaps the next pass's
/// weight fill with this pass's contraction.
macro_rules! pure_pass_sl {
 ($device:ident,$xr:ident,$sxd:ident,$r2d:ident,$w4:ident,$sd:ident,$cd:ident,$p:literal,$act:ident; $($q:literal)*) => {{
  let wt=$w4.view().tile::<m![Lu/16%4],1,m![Lu/64,Lu/16%4=1 #4,Lu%16,K]>($p);
  let wdm:DmTensor<i8,Chip,UCl,USl<URING,UXS,UYS>,m![Lu%16,K]>=wt.to_dm(&mut $device.tdma);
  let w=gemm::to_trf::<UCl,USl<URING,UXS,UYS>,Lu,K,F,CH,CH2,m![Lu%16,K]>($device,&wdm);
  let ws=col_rows::<URING,UXS,UYS>($device,&$sd,$p);
  let cw=col_rows::<URING,UXS,UYS>($device,&$cd,$p);
  let mut y:YQ<T,URING,UXS,UYS,QS>=DmTensor::new();
  $({
   let sx=vrf_q_f32::<T,URING,UXS,UYS,QS,NQ>($device,&$sxd,$q);
   let r2=vrf_q_i32::<T,URING,UXS,UYS,QS,NQ>($device,&$r2d,$q);
   contract4q::<T,URING,UXS,UYS,K,F,CH,CH2,QS,NQ>($device,&$xr,&w,&r2,&cw,&sx,&ws,$q,&mut y);
  })*
  let y:DmTensor<bf16,Chip,UCl,USl<URING,UXS,UYS>,m![{TT},{TR},Lu/8%2,Lu%8]>=unsafe{y.reshape()};
  gl_geglu::<TT,TR,URING,UXS,UYS>($device,&y,$p,&mut $act);
 }};
}
macro_rules! up_gate_pure_sl {
 ($name:ident; $($q:literal)*) => {
 /// Straight-line pure 4-bit Gate/Up for a bucket of exactly as many 64-row groups as listed (NQ must match).
 pub(crate) fn $name<T:AxisName,URING:M,UXS:M,UYS:M,TT:M,TR:M,K:AxisName,const F:usize,const CH:usize,const CH2:usize,const QS:usize,const NQ:usize,XE:M>(
  device:&mut Device,codes4:HbmTensorView<'_,i8,Chip,XE>,scales4:&HbmTensor<f32,Chip,m![T]>,r2s:&HbmTensor<f32,Chip,m![T]>,
  w4:&HbmTensor<i8,Chip,m![Lu,K]>,ws4:&HbmTensor<f32,Chip,m![Xc2,Lu]>,
 ) -> DmTensor<bf16,Chip,UCl,GSl,m![Gpp,T,Kb%16]> {
  let mut act:DmTensor<bf16,Chip,UCl,USl<URING,UXS,UYS>,m![Lu/32%2,{TT},{TR},Lu/16%2,Lu%8]>=DmTensor::new();
  let sx4=rows_ag_f32::<T,URING,UXS,UYS,QS>(device,scales4);
  let r2=rows_ag_i32::<T,URING,UXS,UYS,QS>(device,r2s);
  let xr4:XR4<T,URING,K,CH,QS>=codes4.to_dm(&mut device.tdma);
  let sd4=up_rows::<URING,UXS,UYS>(device,ws4,0);
  let cd4=up_rows::<URING,UXS,UYS>(device,ws4,1);
  pure_pass_sl!(device,xr4,sx4,r2,w4,sd4,cd4,0,act; $($q)*);
  pure_pass_sl!(device,xr4,sx4,r2,w4,sd4,cd4,1,act; $($q)*);
  pure_pass_sl!(device,xr4,sx4,r2,w4,sd4,cd4,2,act; $($q)*);
  pure_pass_sl!(device,xr4,sx4,r2,w4,sd4,cd4,3,act; $($q)*);
  unsafe { act.reshape() }
 }
 };
}
// the Tok128 class512 Q4 bodies (`ops_core::smallbatch`) run two 64-row groups.
up_gate_pure_sl!(up_gate_pure_sl2; 0 1);
