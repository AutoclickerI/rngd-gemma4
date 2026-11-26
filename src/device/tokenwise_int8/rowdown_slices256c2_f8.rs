//! Two-pass activation-stationary FP8 Down on all 256 slices of each cluster.
//! Clusters are the Row's token halves (TCl), slices are H/480 (8) x T%HALF/8 (32). Each slice keeps 16 token rows (both
//! PS passes x T%8) in TRF and streams its 480 weight rows once per quarter;
//! every 64-byte weight packet feeds both passes, so the stream rate equals the
//! stock weight-stationary form while 256 instead of 240 slices contract.
//! The four K quarters retain their contraction order and pairwise f32 sum.
//! Load code half 0 during the later Up/Gate passes to spread DMA across both stages.
use super::*;
use super::rowdown::{down_pair_sum, weight_quarter};
type Sl<T,const HALF:usize>=m![H/480,T%{HALF}/8];
type Part<T,const HALF:usize,const PS:usize>=DmTensor<f32,Chip,TCl<T,HALF>,Sl<T,HALF>,m![H%480,T/{PS},T%8]>;
type X<T,const HALF:usize,const PS:usize>=DmTensor<f8e4m3,Chip,TCl<T,HALF>,Sl<T,HALF>,m![T/{PS},T%8,Kd]>;
/// Load one group of a code half. Quarter activations are replicated 8x over the H/480 slice groups by one DMA from the whole code half.
pub(super) fn load_x_half<T:AxisName,const HALF:usize,const PS:usize>(device:&mut Device,dcodes:&HbmTensor<f8e4m3,Chip,m![Kq4/2,T,Kd]>,g:usize)->X<T,HALF,PS>{
    dcodes.view().tile::<m![Kq4/2],1,m![Kq4/2=1 #2,T,Kd]>(g).to_dm(&mut device.tdma)
}
fn quarter<T:AxisName,const HALF:usize,const PS:usize>(device:&mut Device,x:&X<T,HALF,PS>,wd:&HbmTensor<f8e4m3,Chip,m![Kq4,H,Kd]>,p:usize,g:usize)->Part<T,HALF,PS>{
    let xt:TrfTensor<f8e4m3,Chip,TCl<T,HALF>,Sl<T,HALF>,m![T%8],m![T/{PS},Kd/32%30,Kd/1920%2,Kd/960%2,Kd%32]>=device.main.begin(x.view())
        .fetch::<m![T%8,T/{PS},Kd/32%30,Kd/1920%2,Kd/960%2],m![Kd%32]>()
        .collect::<m![T%8,T/{PS},Kd/32%30,Kd/1920%2,Kd/960%2],m![Kd%32]>().to_trf();
    let w:DmTensor<f8e4m3,Chip,TCl<T,HALF>,m![H/480,H%32],m![H/32%15,Kd]>=weight_quarter(wd, p, g).to_dm(&mut device.tdma);
    device.main.begin(w.view())
        .fetch::<m![H/32%15,Kd/32%30,Kd/960%4],m![Kd%32]>()
        .switch::<Sl<T,HALF>,m![H/32%15,H%32,Kd/32%30,Kd/960%4]>(SwitchConfig::Broadcast01{slice1:32,slice0:1,time0:120})
        .collect::<m![H/32%15,H%32,Kd/32%30,Kd/960%4],m![Kd%32]>()
        .contract_outer::<m![H/32%15,H%32,Kd/32%30,Kd/1920%2,T/{PS}],m![Kd/960%2,Kd%32],_,_,_>(&xt)
        .contract_packet::<m![1]>().contract_time::<m![H/32%15,H%32,T/{PS}]>()
        .contract_lane::<m![H/32%15,H%32,T/{PS}],m![T%8]>(LaneMode::Interleaved)
        .commit_trim::<m![T%8]>().commit()
}
fn add<T:AxisName,const HALF:usize,const PS:usize>(device:&mut Device,a:&Part<T,HALF,PS>,b:&Part<T,HALF,PS>)->Part<T,HALF,PS>{
    down_pair_sum!(device.sub, a.view(), b.view(), slices PS)
        .vector_widen_concat::<m![T/{PS},H%480],m![T%8]>()
        .vector_final().commit_trim::<m![T%8]>().commit()
}
fn add_main<T:AxisName,const HALF:usize,const PS:usize>(device:&mut Device,a:&Part<T,HALF,PS>,b:&Part<T,HALF,PS>)->Part<T,HALF,PS>{
    down_pair_sum!(device.main, a.view(), b.view(), slices PS)
        .vector_widen_concat::<m![T/{PS},H%480],m![T%8]>()
        .vector_final().commit_trim::<m![T%8]>().commit()
}
/// Both groups' half-0 activations (`x00`, `x01`) were loaded by the caller during Up/Gate passes 2-3 (`load_x_half`);
/// half 1 is loaded here per group from `dcodes1`.
pub(super) fn down_split<T:AxisName,const HALF:usize,const PS:usize>(device:&mut Device,x00:X<T,HALF,PS>,x01:X<T,HALF,PS>,dcodes1:&HbmTensor<f8e4m3,Chip,m![Kq4/2,T,Kd]>,wd:&HbmTensor<f8e4m3,Chip,m![Kq4,H,Kd]>,wd_scale:&HbmTensor<f32,Chip,m![H]>,sigma:&HbmTensor<f32,Chip,m![1 #8]>)->(Row<bf16,T,HALF,m![Kh]>,Row<bf16,T,HALF,m![Kh]>){
    const{assert!(T::SIZE==1024 && HALF==256 && PS==512);}
    let a={let a=quarter::<T,HALF,PS>(device,&x00,wd,0,0);
        let x=load_x_half::<T,HALF,PS>(device,dcodes1,0);let b=quarter::<T,HALF,PS>(device,&x,wd,1,0);add(device,&a,&b)};
    let b={let a=quarter::<T,HALF,PS>(device,&x01,wd,0,1);
        let x=load_x_half::<T,HALF,PS>(device,dcodes1,1);let b=quarter::<T,HALF,PS>(device,&x,wd,1,1);add_main(device,&a,&b)};
    tail::<T,HALF,PS>(device,&a,&b,wd_scale,sigma)
}
fn tail<T:AxisName,const HALF:usize,const PS:usize>(device:&mut Device,a:&Part<T,HALF,PS>,b:&Part<T,HALF,PS>,wd_scale:&HbmTensor<f32,Chip,m![H]>,sigma:&HbmTensor<f32,Chip,m![1 #8]>)->(Row<bf16,T,HALF,m![Kh]>,Row<bf16,T,HALF,m![Kh]>){
    let sd:DmTensor<f32,Chip,TCl<T,HALF>,Sl<T,HALF>,m![1 #8]>=sigma.to_dm(&mut device.tdma);
    let sx:VrfTensor<f32,Chip,TCl<T,HALF>,Sl<T,HALF>,m![1 #8]>=device.sub.begin(sd.view()).fetch::<m![1],m![1 #8]>().collect::<m![1],m![1 #8]>().to_vrf();
    let sd:DmTensor<f32,Chip,TCl<T,HALF>,Sl<T,HALF>,m![H%480]>=wd_scale.to_dm(&mut device.tdma);
    let ws:VrfTensor<f32,Chip,TCl<T,HALF>,Sl<T,HALF>,m![H%480]>=device.sub.begin(sd.view()).fetch::<m![H%480/8],m![H%8]>().collect::<m![H%480/8],m![H%8]>().to_vrf();
    let y:DmTensor<bf16,Chip,TCl<T,HALF>,Sl<T,HALF>,m![T/{PS},T%8,H%480]>=down_pair_sum!(device.main, a.view(), b.view(), slices PS)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&sx)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1),&ws)
        .vector_widen_concat::<m![T/{PS},H%480],m![T%8]>().vector_final()
        .cast::<bf16,m![T%8 #16]>()
        .transpose::<m![T/{PS},H%480/4,T%8],m![H%4 #16]>()
        .commit_trim::<m![H%4]>().commit();
    let mut r0:DmTensor<bf16,Chip,TCl<T,HALF>,TSl<T,HALF>,m![H]>=DmTensor::new();
    let mut r1:DmTensor<bf16,Chip,TCl<T,HALF>,TSl<T,HALF>,m![H]>=DmTensor::new();
    y.view().tile::<m![T/{PS}],1,m![T/{PS}=1 #2,T%8,H%480]>(0).to_dm_view(&mut device.tdma,r0.view_mut());
    y.view().tile::<m![T/{PS}],1,m![T/{PS}=1 #2,T%8,H%480]>(1).to_dm_view(&mut device.tdma,r1.view_mut());
    (unsafe{r0.reshape()},unsafe{r1.reshape()})
}
