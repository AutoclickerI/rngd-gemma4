//! Tok1024 packed Q4: two 32-weight-row passes, with 512-token paired streams.
//! Arithmetic, packed-byte order, coefficients and K reduction match contract4q.
use super::*;
use crate::axes::Dummy2 as Xh;
type W32<R,X,Y> = TrfTensor<i8,Chip,UCl,USl<R,X,Y>,m![Lu%8],m![Lu/8%4,Hb/32%15,Hb/960%2,Hb/480%2,Hb%32]>;
type Y32<T,R,X,Y,const QS:usize> = DmTensor<bf16,Chip,UCl,USl<R,X,Y>,m![T/{QS},T%{QS}/64,T%64,Lu/8%4,Lu%8]>;

type PackedPair<T,const HALF:usize,const PS:usize> = Row<i8,T,HALF,m![Xh,T/{PS},Hb]>;
type PackedHalf<T,const HALF:usize,const PS:usize> = DmTensor<i8,Chip,UCl,TSl<T,HALF>,m![T/{PS},Hb]>;

fn copy_pair<T:AxisName,const HALF:usize,const PS:usize>(
 device:&mut Device, r:&Row<i8,T,HALF,m![T/{PS},Hb]>, k:usize,
 pair:&mut PackedPair<T,HALF,PS>,
) {
 device.main.begin(r.view())
 .fetch::<m![T/{PS},Hb/32],m![Hb%32]>()
 .collect::<m![T/{PS},Hb/32],m![Hb%32]>()
 .commit_trim::<m![Hb%32]>()
 .commit_view(pair.view_mut().tile::<m![Xh],1,m![Xh=1 #{!}2,T/{PS},Hb]>(k));
}

fn exchange<T:AxisName,const HALF:usize,const PS:usize>(
 device:&mut Device,pk:&Row<i8,T,HALF,m![T/{PS},Hb]>,
)->(PackedHalf<T,HALF,PS>,PackedHalf<T,HALF,PS>) {
 let other:Row<i8,T,HALF,m![T/{PS},Hb]>=pk.view().cluster_swap().to_dm(&mut device.tdma);
 let mut pair=PackedPair::<T,HALF,PS>::new();
 copy_pair::<T,HALF,PS>(device,pk,0,&mut pair);
 copy_pair::<T,HALF,PS>(device,&other,1,&mut pair);
 let a:Row<i8,T,HALF,m![T/{PS},Hb]>=pair.asymmetric_cluster_slice::<m![Xh],m![T/{PS},Hb]>(&mut device.sub,&[0,1]);
 let b:Row<i8,T,HALF,m![T/{PS},Hb]>=pair.asymmetric_cluster_slice::<m![Xh],m![T/{PS},Hb]>(&mut device.sub,&[1,0]);
 (unsafe{a.reshape()},unsafe{b.reshape()})
}

#[allow(clippy::too_many_arguments)]
fn contract_pair<T:AxisName,const HALF:usize,const PS:usize,URING:M,UXS:M,UYS:M>(
 device:&mut Device,x:&PackedHalf<T,HALF,PS>,
 w:&W32<URING,UXS,UYS>,
 r2:&VrfTensor<i32,Chip,UCl,USl<URING,UXS,UYS>,m![T/512,T%256]>,
 cw:&VrfTensor<f32,Chip,UCl,USl<URING,UXS,UYS>,m![Lu/8%4,Lu%8]>,
 sx:&VrfTensor<f32,Chip,UCl,USl<URING,UXS,UYS>,m![T/512,T%256]>,
 ws:&VrfTensor<f32,Chip,UCl,USl<URING,UXS,UYS>,m![Lu/8%4,Lu%8]>,
 q:usize,y:&mut Y32<T,URING,UXS,UYS,256>,
) {
 device.main.begin(x.view())
 .fetch::<m![T/512,Hb/32%15,Hb/960%2,Hb/480%2],m![Hb%32]>()
 .switch::<USl<URING,UXS,UYS>,m![T/512,T%{HALF},Hb/32%15,Hb/960%2,Hb/480%2]>(SwitchConfig::Broadcast01{slice1:HALF,slice0:1,time0:60})
 .collect::<m![T/512,T%{HALF},Hb/32%15,Hb/960%2,Hb/480%2],m![Hb%32]>()
 .contract_outer::<m![T/512,T%{HALF},Hb/32%15,Hb/960%2,Lu/8%4],m![Hb/480%2,Hb%32],_,_,_>(w)
 .contract_packet::<m![1]>()
 .contract_time::<m![T/512,T%{HALF},Lu/8%4]>()
 .contract_lane::<m![T/512,T%{HALF},Lu/8%4],m![Lu%8]>(LaneMode::Interleaved)
 .vector_init().vector_intra_slice_tag(TagMode::Zero)
 .vector_fxp(FxpBinaryOp::LeftShift,2)
 .vector_fxp(FxpBinaryOp::AddFxp,r2)
 .vector_fxp_to_fp(31)
 .vector_narrow_split::<m![T/512,T%{HALF},Lu/8%4,Lu/4%2],m![Lu%4]>()
 .vector_fp_binary(FpBinaryOp::AddF,cw)
 .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),sx)
 .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1),ws)
 .vector_widen_concat::<m![T/512,T%{HALF},Lu/8%4],m![Lu%8]>()
 .vector_final().cast::<bf16,m![Lu%8 #16]>().commit_trim::<m![Lu%8]>()
 .commit_view(y.view_mut().tile::<m![T/256%2],1,m![T/512,T/256%2=1 #{!}2,T%256/64,T%64,Lu/8%4,Lu%8]>(q));
}

fn coeff_f32<T:AxisName,URING:M,UXS:M,UYS:M>(
 device:&mut Device,d:&TokAll<f32,T,URING,UXS,UYS,256>,half:usize,
)->VrfTensor<f32,Chip,UCl,USl<URING,UXS,UYS>,m![T/512,T%256]> {
 device.sub.begin(d.view().tile::<m![T/256%2],1,m![T/512,T/256%2=1 #2,T%256]>(half))
 .fetch::<m![T/512,T%256/8],m![T%8]>()
 .collect::<m![T/512,T%256/8],m![T%8]>().to_vrf()
}

fn coeff_i32<T:AxisName,URING:M,UXS:M,UYS:M>(
 device:&mut Device,d:&TokAll<i32,T,URING,UXS,UYS,256>,half:usize,
)->VrfTensor<i32,Chip,UCl,USl<URING,UXS,UYS>,m![T/512,T%256]> {
 device.sub.begin(d.view().tile::<m![T/256%2],1,m![T/512,T/256%2=1 #2,T%256]>(half))
 .fetch::<m![T/512,T%256/8],m![T%8]>()
 .collect::<m![T/512,T%256/8],m![T%8]>().to_vrf()
}

#[allow(clippy::too_many_arguments)]
fn pass<T:AxisName,const HALF:usize,const PS:usize,URING:M,UXS:M,UYS:M,TT:M,TR:M>(
 device:&mut Device,x0:&PackedHalf<T,HALF,PS>,x1:&PackedHalf<T,HALF,PS>,
 sxd:&TokAll<f32,T,URING,UXS,UYS,256>,r2d:&TokAll<i32,T,URING,UXS,UYS,256>,
 w4:&HbmTensor<i8,Chip,m![Lu,Hb]>,sd:&UpRows<URING,UXS,UYS>,cd:&UpRows<URING,UXS,UYS>,
 p:usize,act:&mut DmTensor<bf16,Chip,UCl,USl<URING,UXS,UYS>,m![Lu/32%2,{TT},{TR},Lu/16%2,Lu%8]>,
) {
 let wt=w4.view().tile::<m![Lu/32%2],1,m![Lu/64,Lu/32%2=1 #2,Lu%32,Hb]>(p);
 let wdm:DmTensor<i8,Chip,UCl,USl<URING,UXS,UYS>,m![Lu%32,Hb]>=wt.to_dm(&mut device.tdma);
 let w:W32<URING,UXS,UYS>=device.sub.begin(wdm.view())
 .fetch::<m![Lu%8,Lu/8%4,Hb/32%15,Hb/960%2,Hb/480%2],m![Hb%32]>()
 .collect::<m![Lu%8,Lu/8%4,Hb/32%15,Hb/960%2,Hb/480%2],m![Hb%32]>().to_trf();
 let ws=col_rows32::<URING,UXS,UYS>(device,sd,p);
 let cw=col_rows32::<URING,UXS,UYS>(device,cd,p);
 let mut y:Y32<T,URING,UXS,UYS,256>=DmTensor::new();
 let sx0=coeff_f32::<T,URING,UXS,UYS>(device,sxd,0);
 let r20=coeff_i32::<T,URING,UXS,UYS>(device,r2d,0);
 contract_pair::<T,HALF,PS,URING,UXS,UYS>(device,x0,&w,&r20,&cw,&sx0,&ws,0,&mut y);
 let sx1=coeff_f32::<T,URING,UXS,UYS>(device,sxd,1);
 let r21=coeff_i32::<T,URING,UXS,UYS>(device,r2d,1);
 contract_pair::<T,HALF,PS,URING,UXS,UYS>(device,x1,&w,&r21,&cw,&sx1,&ws,1,&mut y);
 let y:DmTensor<bf16,Chip,UCl,USl<URING,UXS,UYS>,m![{TT},{TR},Lu/16%2,Lu/8%2,Lu%8]>=unsafe{y.reshape()};
 gl_geglu32::<TT,TR,URING,UXS,UYS>(device,&y,p,act);
}

pub(super) fn up_pp<T:AxisName,const HALF:usize,const PS:usize,URING:M,UXS:M,UYS:M,TT:M,TR:M>(
 device:&mut Device,pk:&Row<i8,T,HALF,m![T/{PS},Hb]>,
 scales4:&HbmTensor<f32,Chip,m![T]>,r2s:&HbmTensor<f32,Chip,m![T]>,
 sigma_down:&HbmTensor<f32,Chip,m![1 #8]>,w4:&HbmTensor<i8,Chip,m![Lu,Hb]>,ws4:&HbmTensor<f32,Chip,m![Xc2,Lu]>,
)->(HbmTensor<f8e4m3,Chip,m![Kq4/2,T,Kd]>,HbmTensor<f8e4m3,Chip,m![Kq4/2,T,Kd]>) {
 let (x0,x1)=exchange::<T,HALF,PS>(device,pk);
 let sx4=rows_ag_f32::<T,URING,UXS,UYS,256>(device,scales4);
 let r2=rows_ag_i32::<T,URING,UXS,UYS,256>(device,r2s);
 let sd=up_rows::<URING,UXS,UYS>(device,ws4,0);
 let cd=up_rows::<URING,UXS,UYS>(device,ws4,1);
 let mut act:DmTensor<bf16,Chip,UCl,USl<URING,UXS,UYS>,m![Lu/32%2,{TT},{TR},Lu/16%2,Lu%8]>=DmTensor::new();
 pass::<T,HALF,PS,URING,UXS,UYS,TT,TR>(device,&x0,&x1,&sx4,&r2,w4,&sd,&cd,0,&mut act);
 let dsd:DmTensor<f32,Chip,UCl,GSl,m![1 #8]>=sigma_down.to_dm(&mut device.tdma);
 let dsv:VrfTensor<f32,Chip,UCl,GSl,m![1 #8]>=device.sub.begin(dsd.view()).fetch::<m![1],m![1 #8]>().collect::<m![1],m![1 #8]>().to_vrf();
 let dcodes0={let a:DmTensorView<'_,bf16,Chip,UCl,GSl,m![Gpp,T,Kb%16]>=unsafe{act.view().reshape()};
 crate::device::tokenwise_int8::ffn_fast::fp8c::up_xchg::pair0_codes::<T>(device,a.tile::<m![Gpp],1,m![Gpp=1 #2,T,Kb%16]>(0),&dsv)};

 pass::<T,HALF,PS,URING,UXS,UYS,TT,TR>(device,&x0,&x1,&sx4,&r2,w4,&sd,&cd,1,&mut act);
 let dcodes1={let a:DmTensorView<'_,bf16,Chip,UCl,GSl,m![Gpp,T,Kb%16]>=unsafe{act.view().reshape()};
 crate::device::tokenwise_int8::ffn_fast::fp8c::up_xchg::pair1_codes::<T>(device,a.tile::<m![Gpp],1,m![Gpp=1 #2,T,Kb%16]>(1),&dsv)};
 (dcodes0,dcodes1)
}

fn col_rows32<R:M,X:M,Y:M>(device:&mut Device,d:&UpRows<R,X,Y>,p:usize)->VrfTensor<f32,Chip,UCl,USl<R,X,Y>,m![Lu/8%4,Lu%8]> {
 device.sub.begin(d.view().tile::<m![Lu/32%2],1,m![Lu/32%2=1 #2,Lu/8%4,Lu%8]>(p))
 .fetch::<m![Lu/8%4],m![Lu%8]>().collect::<m![Lu/8%4],m![Lu%8]>().to_vrf()
}

fn gl_geglu32<TT:M,TR:M,R:M,X:M,Y:M>(
 device:&mut Device,
 y:&DmTensor<bf16,Chip,UCl,USl<R,X,Y>,m![{TT},{TR},Lu/16%2,Lu/8%2,Lu%8]>,
 p:usize,act:&mut DmTensor<bf16,Chip,UCl,USl<R,X,Y>,m![Lu/32%2,{TT},{TR},Lu/16%2,Lu%8]>,
) {
 let gate=y.view().tile::<m![Lu/8%2],1,m![{TT},{TR},Lu/16%2,Lu/8%2=1 #2,Lu%8]>(1);
 let up=y.view().tile::<m![Lu/8%2],1,m![{TT},{TR},Lu/16%2,Lu/8%2=1 #2,Lu%8]>(0);
 let gl:DmTensor<f32,Chip,UCl,USl<R,X,Y>,m![{TT},{TR},Lu/16%2,Lu%8]>=device.main.begin(gate)
 .fetch::<m![{TT},{TR},Lu/16%2],m![Lu%8]>().fetch_cast::<f32>()
 .collect::<m![{TT},{TR},Lu/16%2],m![Lu%8]>()
 .vector_init().vector_intra_slice_tag(TagMode::Zero)
 .vector_narrow_split::<m![{TT},{TR},Lu/16%2,Lu/4%2],m![Lu%4]>()
 .vector_stash().vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),INVSQRT2)
 .vector_fp_unary(FpUnaryOp::Erf).vector_fp_binary(FpBinaryOp::AddF,1f32)
 .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1),Stash)
 .vector_widen_concat::<m![{TT},{TR},Lu/16%2],m![Lu%8]>()
 .vector_final().commit_trim::<m![Lu%8]>().commit();
 let u:DmTensor<f32,Chip,UCl,USl<R,X,Y>,m![{TT},{TR},Lu/16%2,Lu%8]>=device.main.begin(up)
 .fetch::<m![{TT},{TR},Lu/16%2],m![Lu%8]>().fetch_cast::<f32>()
 .collect::<m![{TT},{TR},Lu/16%2],m![Lu%8]>().commit_trim::<m![Lu%8]>().commit();
 device.main.begin_interleaved::<Pi,_,_,_,_,_>(u.view(),gl.view())
 .fetch::<m![{TT},{TR}/16,Pi,{TR}%16,Lu/16%2],m![Lu%8]>()
 .collect::<m![{TT},{TR}/16,Pi,{TR}%16,Lu/16%2],m![Lu%8]>()
 .vector_init().vector_intra_slice_unzip::<Pi,m![{TT},{TR},Lu/16%2]>()
 .vector_narrow_split::<m![{TT},{TR},Lu/16%2,Lu/4%2],m![Lu%4]>()
 .vector_fp_zip(FpBinaryOp::MulF(FpMulAlu::Mul0)).vector_fp_div(2f32)
 .vector_widen_concat::<m![{TT},{TR},Lu/16%2],m![Lu%8]>()
 .vector_final().cast::<bf16,m![Lu%8 #16]>().commit_trim::<m![Lu%8]>()
 .commit_view(act.view_mut().tile::<m![Lu/32%2],1,m![Lu/32%2=1 #{!}2,{TT},{TR},Lu/16%2,Lu%8]>(p));
}
