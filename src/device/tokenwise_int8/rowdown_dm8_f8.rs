//! Row-owned FP8 Down: interleaved token passes and a DM-only post-FFN handoff.
//! Sum the four K-quarter partials in f32 as `(q0 + q1) + (q2 + q3)`.
//! Each cluster owns HALF rows of each PS-row pass and reads all four weight quarters.
//! Stream one weight quarter at a time and keep the partial sums in DM.
use super::*;
use super::rowdown::{down_pair_sum, weight_quarter};
axes![RowGroup = 2];
type RCl = m![RowGroup];
type RSl = DSl<m![1], m![H / 64 # 64], m![H / 16 % 4]>;
type Partial<T, const HALF: usize, const PS: usize> = DmTensor<f32, Chip, RCl, RSl, m![T / { PS }, T % { HALF } / 64, T % 64, H / 8 % 2, H % 8]>;
type Weight = fp8c::WTrf8<RCl, RSl, H, Kd, 30, 960, 1920>;

fn add_sub<T:AxisName,const HALF: usize, const PS: usize>(device:&mut Device,a:&Partial<T,HALF,PS>,b:&Partial<T,HALF,PS>) -> Partial<T,HALF,PS> {
    down_pair_sum!(device.sub, a.view(), b.view(), ring [T / { PS }, T % { HALF } / 64,])
        .vector_widen_concat::<m![T / { PS }, T % { HALF } / 64,T % 64,H / 8 % 2],m![H % 8]>()
        .vector_final().commit_trim::<m![H % 8]>()
        .commit()
}

fn add_main<T:AxisName,const HALF: usize, const PS: usize>(device:&mut Device,a:&Partial<T,HALF,PS>,b:&Partial<T,HALF,PS>) -> Partial<T,HALF,PS> {
    down_pair_sum!(device.main, a.view(), b.view(), ring [T / { PS }, T % { HALF } / 64,])
        .vector_widen_concat::<m![T / { PS }, T % { HALF } / 64,T % 64,H / 8 % 2],m![H % 8]>()
        .vector_final().commit_trim::<m![H % 8]>()
        .commit()
}

// Independently versioned HBM code halves.
fn quarter_split<T: AxisName, const HALF: usize, const PS: usize>(device:&mut Device,dcodes:&HbmTensor<f8e4m3,Chip,m![Kq4 /2,T,Kd]>,w:&Weight,group:usize) -> Partial<T,HALF,PS> {
    let codes:HbmTensorView<'_,f8e4m3,Chip,m![Kq4 /2,T / { PS },RowGroup,T % { HALF },Kd]>=unsafe {dcodes.view().reshape()};
    let x:DmTensor<f8e4m3,Chip,RCl,m![1,T % 64,Kd / 960 % 4],m![T / { PS }, T % { HALF } / 64,Kd % 960]>=
        codes.tile::<m![Kq4 / 2],1,m![Kq4 / 2 = 1 # 2,T / { PS },RowGroup,T % { HALF },Kd]>(group).to_dm(&mut device.tdma);
    device.main.begin(x.view())
        .fetch::<m![T / { PS }, T % { HALF } / 64,Kd / 32 % 30],m![Kd % 32]>()
        .switch::<RSl,m![T / { PS }, T % { HALF } / 64,T % 64,Kd / 32 % 30,Kd / 960 % 4]>(SwitchConfig::Broadcast01{slice1:64,slice0:4,time0:30})
        .collect::<m![T / { PS }, T % { HALF } / 64,T % 64,Kd / 32 % 30,Kd / 960 % 4],m![Kd % 32]>()
        .contract_outer::<m![T / { PS }, T % { HALF } / 64,T % 64,Kd / 32 % 30,Kd / 1920 % 2,H / 8 % 2],m![Kd / 960 % 2,Kd % 32],_,_,_>(w)
        .contract_packet::<m![1]>()
        .contract_time::<m![T / { PS }, T % { HALF } / 64,T % 64,H / 8 % 2]>()
        .contract_lane::<m![T / { PS }, T % { HALF } / 64,T % 64,H / 8 % 2],m![H % 8]>(LaneMode::Interleaved)
        .commit_trim::<m![H % 8]>().commit()
}

// Stream one weight quarter at a time and pass all three results directly through DM.
#[cfg(feature = "compile-carried")]
fn streamed_weight_quarter(device:&mut Device,wd:&HbmTensor<f8e4m3,Chip,m![Kq4,H,Kd]>,p:usize,group:usize)->Weight {
    let wt=weight_quarter(wd, p, group);
    let wdm:DmTensor<f8e4m3,Chip,RCl,RSl,m![H % 16,Kd]>=wt.to_dm(&mut device.tdma);
    fp8c::to_trf8_main::<RCl,RSl,H,Kd,30,960,1920,m![H % 16,Kd]>(device,&wdm)
}

#[cfg(feature = "compile-carried")]
pub(super) fn down_split3<T:AxisName,const HALF:usize,const PS:usize>(
    device:&mut Device,dcodes0:&HbmTensor<f8e4m3,Chip,m![Kq4 / 2,T,Kd]>,
    dcodes1:&HbmTensor<f8e4m3,Chip,m![Kq4 / 2,T,Kd]>,
    wd:&HbmTensor<f8e4m3,Chip,m![Kq4,H,Kd]>,wd_scale:&HbmTensor<f32,Chip,m![H]>,
    sigma:&HbmTensor<f32,Chip,m![1 #8]>
)->(Row<bf16,T,HALF,m![Kh]>,Row<bf16,T,HALF,m![Kh]>,Row<bf16,T,HALF,m![Kh]>) {
    const { assert!(T::SIZE==1536 && T::SIZE==3*PS && PS==2*HALF && HALF==256); }
    let a={
        let w0=streamed_weight_quarter(device,wd,0,0);
        let a=quarter_split::<T,HALF,PS>(device,dcodes0,&w0,0);
        let w1=streamed_weight_quarter(device,wd,1,0);
        let b=quarter_split::<T,HALF,PS>(device,dcodes1,&w1,0);
        add_sub::<T,HALF,PS>(device,&a,&b)
    };
    let b={
        let w0=streamed_weight_quarter(device,wd,0,1);
        let a=quarter_split::<T,HALF,PS>(device,dcodes0,&w0,1);
        let w1=streamed_weight_quarter(device,wd,1,1);
        let b=quarter_split::<T,HALF,PS>(device,dcodes1,&w1,1);
        add_main::<T,HALF,PS>(device,&a,&b)
    };
    let a = a.view();
    let b = b.view();
    let ws=gemm::col_scales::<RCl,RSl,H,m![H]>(device,wd_scale.view());
    let sd:DmTensor<f32,Chip,RCl,RSl,m![1 #8]>=sigma.to_dm(&mut device.tdma);
    let sx:VrfTensor<f32,Chip,RCl,RSl,m![1 #8]>=device.sub.begin(sd.view()).fetch::<m![1],m![1 #8]>().collect::<m![1],m![1 #8]>().to_vrf();
    let y:DmTensor<bf16,Chip,RCl,RSl,m![T / { PS }, T % { HALF } / 64,T %64,H /8 %2,H %8]>=down_pair_sum!(device.main, a, b, ring [T / { PS }, T % { HALF } / 64,])
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&sx)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1),&ws)
        .vector_widen_concat::<m![T / { PS }, T % { HALF } / 64,T %64,H /8 %2],m![H %8]>().vector_final()
        .cast::<bf16,m![H %8 #16]>().commit_trim::<m![H %8]>().commit();
    let y:DmTensor<bf16,Chip,RCl,m![H /128 #32,H /16 %8],m![T / { PS }, T % { HALF },H %16]>=unsafe {y.reshape()};
    let y:DmTensor<bf16,Chip,RCl,m![H /128 #32,T %8],m![T / { PS }, T % { HALF } /8,H /16 %8,H %16]>=device.main.begin(y.view())
        .fetch::<m![T / { PS }, T % { HALF } /8,T %8],m![H %16]>()
        .switch::<m![H /128 #32,T %8],m![T / { PS }, T % { HALF } /8,H /16 %8]>(SwitchConfig::InterTranspose{slice1:8,slice0:1,time0:1})
        .collect::<m![T / { PS }, T % { HALF } /8,H /16 %8],m![H %16]>().commit_trim::<m![H %16]>().commit();
    // The initial eight-slice transpose gives the DMA engine aligned 32-byte
    // feature fragments. Redistribute those directly to token-owned slices.
    // Write each pass straight into its consumer layout. Separate transfers let
    // post-FFN pass zero start while the second pass is still being redistributed.
    let mut r0: DmTensor<bf16,Chip,RCl,TSl<T,HALF>,m![H]>=DmTensor::new();
    let mut r1: DmTensor<bf16,Chip,RCl,TSl<T,HALF>,m![H]>=DmTensor::new();
    let mut r2: DmTensor<bf16,Chip,RCl,TSl<T,HALF>,m![H]>=DmTensor::new();
    y.view().tile::<m![T / { PS }],1,m![T / { PS }=1 #3,T % { HALF } /8,H /16 %8,H %16]>(0)
        .to_dm_view(&mut device.tdma,r0.view_mut());
    y.view().tile::<m![T / { PS }],1,m![T / { PS }=1 #3,T % { HALF } /8,H /16 %8,H %16]>(1)
        .to_dm_view(&mut device.tdma,r1.view_mut());
    y.view().tile::<m![T / { PS }],1,m![T / { PS }=1 #3,T % { HALF } /8,H /16 %8,H %16]>(2)
        .to_dm_view(&mut device.tdma,r2.view_mut());
    (unsafe {r0.reshape()},unsafe {r1.reshape()},unsafe {r2.reshape()})
}
