//! FP8 Down contraction with token rows split across clusters.
//! Sum the four K-quarter partials in f32 as `(q0 + q1) + (q2 + q3)`.
//! Each cluster owns half the token rows and reads all four weight quarters.
//! First-quarter weights are prefetched outside the group scopes; partial sums stay in DM.
use super::*;
use super::rowdown::{down_pair_sum, weight_quarter};
axes![RowGroup = 2];
type RCl = m![RowGroup];
type RSl = DSl<m![1], m![H / 64 # 64], m![H / 16 % 4]>;
type Partial<T, const OWN: usize> = DmTensor<f32, Chip, RCl, RSl, m![T % { OWN } / 64, T % 64, H / 8 % 2, H % 8]>;
type Weight = fp8c::WTrf8<RCl, RSl, H, Kd, 30, 960, 1920>;
type Weights = DmTensor<f8e4m3, Chip, RCl, RSl, m![Kq4 / 2, H % 16, Kd]>;
type Codes<'a, T, const OWN: usize> = HbmTensorView<'a, f8e4m3, Chip, m![Kq4 % 2, Kq4 / 2, RowGroup, T % { OWN }, Kd]>;

fn first_weight(device: &mut Device, w: &Weights, group: usize) -> Weight {
    device.main.begin(w.view().tile::<m![Kq4 / 2], 1, m![Kq4 / 2 = 1 # 2, H % 16, Kd]>(group))
        .fetch::<m![H % 8, H / 8 % 2, Kd / 32 % 30, Kd / 1920 % 2, Kd / 960 % 2], m![Kd % 32]>()
        .collect::<m![H % 8, H / 8 % 2, Kd / 32 % 30, Kd / 1920 % 2, Kd / 960 % 2], m![Kd % 32]>().to_trf()
}
fn next_weight(device: &mut Device, wd: &HbmTensor<f8e4m3, Chip, m![Kq4, H, Kd]>, group: usize) -> Weight {
    let wt=weight_quarter(wd, 1, group);
    let wdm: DmTensor<f8e4m3,Chip,RCl,RSl,m![H % 16,Kd]>=wt.to_dm(&mut device.tdma);
    fp8c::to_trf8_main::<RCl,RSl,H,Kd,30,960,1920,m![H % 16,Kd]>(device,&wdm)
}

fn quarter<T: AxisName, const OWN: usize>(device:&mut Device,dcodes:&HbmTensor<f8e4m3,Chip,m![Kq4 %2,Kq4 /2,T,Kd]>,w:&Weight,p:usize,group:usize) -> Partial<T,OWN> {
    let codes:Codes<'_,T,OWN>=unsafe {dcodes.view().reshape()};
    let x:DmTensor<f8e4m3,Chip,RCl,m![1,T % 64,Kd / 960 % 4],m![T % { OWN } / 64,Kd % 960]>=
        codes.tile::<m![Kq4 % 2],1,m![Kq4 % 2 = 1 # 2,Kq4 / 2,RowGroup,T % { OWN },Kd]>(p)
        .tile::<m![Kq4 / 2],1,m![Kq4 % 2 = 1 # 2,Kq4 / 2 = 1 # 2,RowGroup,T % { OWN },Kd]>(group).to_dm(&mut device.tdma);
    device.main.begin(x.view())
        .fetch::<m![T % { OWN } / 64,Kd / 32 % 30],m![Kd % 32]>()
        .switch::<RSl,m![T % { OWN } / 64,T % 64,Kd / 32 % 30,Kd / 960 % 4]>(SwitchConfig::Broadcast01{slice1:64,slice0:4,time0:30})
        .collect::<m![T % { OWN } / 64,T % 64,Kd / 32 % 30,Kd / 960 % 4],m![Kd % 32]>()
        .contract_outer::<m![T % { OWN } / 64,T % 64,Kd / 32 % 30,Kd / 1920 % 2,H / 8 % 2],m![Kd / 960 % 2,Kd % 32],_,_,_>(w)
        .contract_packet::<m![1]>()
        .contract_time::<m![T % { OWN } / 64,T % 64,H / 8 % 2]>()
        .contract_lane::<m![T % { OWN } / 64,T % 64,H / 8 % 2],m![H % 8]>(LaneMode::Interleaved)
        .commit_trim::<m![H % 8]>().commit()
}

fn add_sub<T:AxisName,const OWN:usize>(device:&mut Device,a:&Partial<T,OWN>,b:&Partial<T,OWN>) -> Partial<T,OWN> {
    down_pair_sum!(device.sub, a.view(), b.view(), ring [T % { OWN } / 64,])
        .vector_widen_concat::<m![T % { OWN } / 64,T % 64,H / 8 % 2],m![H % 8]>()
        .vector_final().commit_trim::<m![H % 8]>()
        .commit()
}

fn add_main<T:AxisName,const OWN:usize>(device:&mut Device,a:&Partial<T,OWN>,b:&Partial<T,OWN>) -> Partial<T,OWN> {
    down_pair_sum!(device.main, a.view(), b.view(), ring [T % { OWN } / 64,])
        .vector_widen_concat::<m![T % { OWN } / 64,T % 64,H / 8 % 2],m![H % 8]>()
        .vector_final().commit_trim::<m![H % 8]>()
        .commit()
}

pub(super) fn down<T:AxisName,const OWN:usize>(device:&mut Device,dcodes:&HbmTensor<f8e4m3,Chip,m![Kq4 % 2,Kq4 / 2,T,Kd]>,wd:&HbmTensor<f8e4m3,Chip,m![Kq4,H,Kd]>,wd_scale:&HbmTensor<f32,Chip,m![H]>, sigma: &HbmTensor<f32,Chip,m![1 # 8]>) -> HbmTensor<bf16,Chip,m![T,H]> {
    const { assert!(T::SIZE == 2*OWN); }
    let wf:Weights=wd.view().tile::<m![Kq4 % 2],1,m![Kq4 / 2,Kq4 % 2=1 #2,H,Kd]>(0).to_dm(&mut device.tdma);
    let a = {
        let group = 0;
        let w0=first_weight(device,&wf,group);
        let a=quarter::<T,OWN>(device,dcodes,&w0,0,group);
        let w1=next_weight(device,wd,group);
        let b=quarter::<T,OWN>(device,dcodes,&w1,1,group);
        add_sub::<T,OWN>(device,&a,&b)
    };
    let b = {
        let group = 1;
        let w0=first_weight(device,&wf,group);
        let a=quarter::<T,OWN>(device,dcodes,&w0,0,group);
        let w1=next_weight(device,wd,group);
        let b=quarter::<T,OWN>(device,dcodes,&w1,1,group);
        add_main::<T,OWN>(device,&a,&b)
    };

    let a = a.view();
    let b = b.view();
    let ws=gemm::col_scales::<RCl,RSl,H,m![H]>(device,wd_scale.view());
    let sd:DmTensor<f32,Chip,RCl,RSl,m![1 #8]>=sigma.to_dm(&mut device.tdma);
    let sx:VrfTensor<f32,Chip,RCl,RSl,m![1 #8]>=device.sub.begin(sd.view()).fetch::<m![1],m![1 #8]>().collect::<m![1],m![1 #8]>().to_vrf();
    let y:DmTensor<bf16,Chip,RCl,RSl,m![T % { OWN } /64,T %64,H /8 %2,H %8]>=down_pair_sum!(device.main, a, b, ring [T % { OWN } / 64,])
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0),&sx)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1),&ws)
        .vector_widen_concat::<m![T % { OWN } /64,T %64,H /8 %2],m![H %8]>().vector_final()
        .cast::<bf16,m![H %8 #16]>().commit_trim::<m![H %8]>().commit();
    let y:DmTensor<bf16,Chip,RCl,m![H /128 #32,H /16 %8],m![T % { OWN },H %16]>=unsafe {y.reshape()};
    let y:DmTensor<bf16,Chip,RCl,m![H /128 #32,T %8],m![T % { OWN } /8,H /16 %8,H %16]>=device.main.begin(y.view())
        .fetch::<m![T % { OWN } /8,T %8],m![H %16]>()
        .switch::<m![H /128 #32,T %8],m![T % { OWN } /8,H /16 %8]>(SwitchConfig::InterTranspose{slice1:8,slice0:1,time0:1})
        .collect::<m![T % { OWN } /8,H /16 %8],m![H %16]>().commit_trim::<m![H %16]>().commit();
    let out:HbmTensor<bf16,Chip,m![RowGroup,T % { OWN },H]>=y.to_hbm(&mut device.tdma);
    unsafe {out.reshape()}
}
