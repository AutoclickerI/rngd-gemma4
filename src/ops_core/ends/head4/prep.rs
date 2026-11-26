//! Prepare the W4 head's input activations. Rotate the normalized rows, quantize each
//! BF16 row to INT8 with range 119, then pack two signed INT4 planes and their row sums.
use furiosa_opt_std::prelude::*;
use crate::{Chip, axes::*};
use crate::device::tokenwise_int8::{gemm::Pi, passes::OnePass, rotate};
use crate::device::tokenwise_int8::rotate::{Ka, Kb, Ob, Ra, Kh, Kq, RotPad, RotK, H60Hbm, Sgi, Sl8, Sgo};
use crate::device::tokenwise_int8::ffn_fast::ug4::{Hb, G240, J8, P2};

const MAGIC: f32 = 12_582_912.0;
const MAGIC_NEG: i32 = -12_582_912;
type CL<T, const HALF:usize> = rotate::TCl<T, HALF>;
type SL<T, const HALF:usize> = rotate::TSl<T, HALF>;
type Row<D,T,const HALF:usize,E> = DmTensor<D,Chip,CL<T,HALF>,SL<T,HALF>,E>;
type Scalar<T,const HALF:usize> = Row<f32,T,HALF,m![1 # 8]>;
type Onehot<T,const HALF:usize> = VrfTensor<f32,Chip,CL<T,HALF>,m![1 # 256],m![Sgi,Sl8]>;

fn scale<T:AxisName,const HALF:usize>(device: &mut Device, y: &Row<bf16,T,HALF, RotPad>) -> Scalar<T,HALF> {
    let y: DmTensorView<'_, bf16, Chip, CL<T,HALF>, SL<T,HALF>, m![Kq]> = unsafe { y.view().reshape() };
    device.main.begin(y)
        .fetch::<m![Kq / 16], m![Kq % 16]>().fetch_cast::<f32>()
        .collect::<m![Kq / 8], m![Kq % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd, i32::MAX).vector_reinterpret::<f32>()
        .vector_narrow_split::<m![Kq / 4], m![Kq % 4]>()
        .vector_intra_slice_reduce::<Kq, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_fp_div(119f32).vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1e-30f32)
        .vector_final().commit_trim::<m![1 # 8]>().commit()
}

fn quantize<T:AxisName,const HALF:usize>(device: &mut Device, y: &Row<bf16,T,HALF, RotPad>, s: &Scalar<T,HALF>) -> Row<i8,T,HALF, RotK> {
    let sv: VrfTensor<f32, Chip, CL<T,HALF>, SL<T,HALF>, m![1 # 8]> = device.sub.begin(s.view())
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    device.main.begin(y.view())
        .fetch::<m![Ob / 32, Ra, Ob % 32 / 16], m![Ob % 16]>().fetch_cast::<f32>()
        .collect::<m![Ob / 32, Ra, Ob % 32 / 8], m![Ob % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ob / 32, Ra, Ob % 32 / 4], m![Ob % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &sv).vector_fp_binary(FpBinaryOp::AddF, MAGIC)
        .vector_widen_concat::<m![Ob / 32, Ra, Ob % 32 / 8], m![Ob % 8]>()
        .vector_fp_to_fxp(31).vector_clip(ClipBinaryOpI32::AddFxp, MAGIC_NEG)
        .vector_clip(ClipBinaryOpI32::Min, 119).vector_clip(ClipBinaryOpI32::Max, -119)
        .vector_final().cast::<i8, m![Ob % 8 # 32]>().commit_trim::<m![Ob % 8]>().commit()
}

fn sum_q<T:AxisName,const HALF:usize>(device: &mut Device, q: &Row<i8,T,HALF, RotK>) -> Scalar<T,HALF> {
    let q: DmTensorView<'_, i8, Chip, CL<T,HALF>, SL<T,HALF>, m![Kh]> = unsafe { q.view().reshape() };
    device.main.begin(q).fetch::<m![Kh / 16], m![Kh % 16]>().fetch_cast::<i32>()
        .collect::<m![Kh / 8], m![Kh % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_fxp_to_fp(31)
        .vector_narrow_split::<m![Kh / 4], m![Kh % 4]>()
        .vector_intra_slice_reduce::<Kh, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>().vector_final().commit_trim::<m![1 # 8]>().commit()
}

/// hi=floor((q+8)/16)=rne(q/16+1/32) for integral q. The 1/32
/// bias assigns exact half-integers upward, so the low digit never becomes +8.
fn hi<T:AxisName,const HALF:usize>(device: &mut Device, q: &Row<i8,T,HALF, RotK>) -> Row<i8,T,HALF, RotK> {
    let shifted: Row<f32,T,HALF, RotK> = device.main.begin(q.view())
        .fetch::<m![Ob / 32, Ra, Ob % 32 / 16], m![Ob % 16]>().fetch_cast::<i32>()
        .collect::<m![Ob / 32, Ra, Ob % 32 / 8], m![Ob % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_fxp_to_fp(31)
        .vector_narrow_split::<m![Ob / 32, Ra, Ob % 32 / 4], m![Ob % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0625f32)
        .vector_fp_binary(FpBinaryOp::AddF, 0.03125f32)
        .vector_widen_concat::<m![Ob / 32, Ra, Ob % 32 / 8], m![Ob % 8]>()
        .vector_final().commit_trim::<m![Ob % 8]>().commit();
    device.main.begin(shifted.view())
        .fetch::<m![Ob / 32, Ra, Ob % 32 / 8], m![Ob % 8]>()
        .collect::<m![Ob / 32, Ra, Ob % 32 / 8], m![Ob % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ob / 32, Ra, Ob % 32 / 4], m![Ob % 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, MAGIC)
        .vector_widen_concat::<m![Ob / 32, Ra, Ob % 32 / 8], m![Ob % 8]>()
        .vector_fp_to_fxp(31).vector_clip(ClipBinaryOpI32::AddFxp, MAGIC_NEG)
        .vector_final().cast::<i8, m![Ob % 8 # 32]>().commit_trim::<m![Ob % 8]>().commit()
}

fn lo<T:AxisName,const HALF:usize>(device: &mut Device, q: &Row<i8,T,HALF, RotK>) -> Row<i8,T,HALF, RotK> {
    // Signed low nibble: ((q xor8) and15)-8 == q-16floor((q+8)/16).
    // All intermediates fit i32 and the result lies in [-8,7].
    device.main.begin(q.view())
        .fetch::<m![Ob / 32, Ra, Ob % 32 / 16], m![Ob % 16]>().fetch_cast::<i32>()
        .collect::<m![Ob / 32, Ra, Ob % 32 / 8], m![Ob % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpI32::BitXor,8)
        .vector_logic(LogicBinaryOpI32::BitAnd,15)
        .vector_clip(ClipBinaryOpI32::AddFxp,-8)
        .vector_final().cast::<i8,m![Ob % 8 # 32]>().commit_trim::<m![Ob % 8]>().commit()
}

/// Pack signed digits with integer masking/shifting. Every input is [-8,7],
/// so low is [0,15], high is [-128,112] and their sum fits signed i8.
fn pack<T:AxisName,const HALF:usize>(device:&mut Device,c:Row<i8,T,HALF,RotK>)->Row<i8,T,HALF,m![Hb]> {
    let c:Row<i8,T,HALF,m![G240,P2,J8]>=unsafe{c.reshape()};
    let low:Row<i8,T,HALF,m![G240,J8]>=device.main.begin(c.view().tile::<m![P2],1,m![G240,P2=1 # 2,J8]>(0))
        .fetch::<m![G240],m![J8]>().fetch_cast::<i32>().collect::<m![G240],m![J8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpI32::BitAnd,15)
        .vector_final().cast::<i8,m![J8 # 32]>().commit_trim::<m![J8]>().commit();
    let high:Row<i8,T,HALF,m![G240,J8]>=device.main.begin(c.view().tile::<m![P2],1,m![G240,P2=1 # 2,J8]>(1))
        .fetch::<m![G240],m![J8]>().fetch_cast::<i32>().collect::<m![G240],m![J8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::LeftShift,4)
        .vector_final().cast::<i8,m![J8 # 32]>().commit_trim::<m![J8]>().commit();
    let packed:Row<i8,T,HALF,m![G240,J8]>=device.main.begin_interleaved::<Pi,_,_,_,_,_>(low.view(),high.view())
        .fetch::<m![G240/16,Pi,G240%16],m![J8]>().fetch_cast::<i32>()
        .collect::<m![G240/16,Pi,G240%16],m![J8]>()
        .vector_init().vector_intra_slice_unzip::<Pi,m![G240]>()
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        .vector_final().cast::<i8,m![J8 # 32]>().commit_trim::<m![J8]>().commit();
    unsafe{packed.reshape()}
}

/// rotate::store_scales' compaction, preserving the sign. Every input here is
/// an integer sum of clipped q codes, including padding slices, so it is finite.
fn store_sum<T:AxisName,const HALF:usize>(device: &mut Device, s: Scalar<T,HALF>, onehot: &Onehot<T,HALF>, dst: &mut HbmTensor<f32, Chip, m![T]>) {
    let sv: VrfTensor<f32, Chip, CL<T,HALF>, SL<T,HALF>, m![1 # 8]> = device.sub.begin(s.view())
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let s: Row<f32,T,HALF, m![Sl8]> = unsafe { s.reshape() };
    let s8: Row<f32,T,HALF, m![Sl8]> = device.main.begin(s.view())
        .fetch::<m![1], m![Sl8]>().collect::<m![1], m![Sl8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_clip_with_mode(ClipBinaryOpF32::Max, BinaryArgMode::Mode11, &sv)
        .vector_final().commit_trim::<m![Sl8]>().commit();
    let s8: DmTensor<f32, Chip, CL<T,HALF>, m![Sgo, Sgi], m![Sl8]> = unsafe { s8.reshape() };
    let g: DmTensor<f32, Chip, CL<T,HALF>, m![1 # 256], m![Sgo, Sl8]> = device.main.begin(s8.view())
        .fetch::<m![1], m![Sl8]>()
        .switch::<m![1 # 256], m![Sgo, Sgi]>(SwitchConfig::Broadcast1 { slice1:256, slice0:1 })
        .collect::<m![Sgo, Sgi], m![Sl8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Sgo, Sgi, Sl8 / 4], m![Sl8 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), onehot)
        .vector_intra_slice_reduce::<Sgi, m![Sgo, Sl8 / 4], m![Sl8 % 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_concat::<m![Sgo], m![Sl8]>()
        .vector_final().commit_trim::<m![Sl8]>().commit();
    let g: DmTensor<f32, Chip, CL<T,HALF>, m![1 # 256], m![T % { HALF } # 256]> = unsafe { g.reshape() };
    g.view().to_hbm_view(&mut device.tdma, dst.view_mut());
}

pub(in crate::ops_core::ends) fn prepare<T:AxisName,const HALF:usize,const PS:usize>(
    device: &mut Device, signed_h: &HbmTensor<bf16, Chip, m![T, H]>,
    h64: &HbmTensor<bf16, Chip, m![Rc, Ob, Kb]>, h60: &HbmTensor<bf16, Chip, m![Rc, { H60Hbm }]>,
    eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>,
    xhi: &mut HbmTensor<i8, Chip, m![T, Hb]>, xlo: &mut HbmTensor<i8, Chip, m![T, Hb]>,
    sx: &mut HbmTensor<f32, Chip, m![T]>, sums: &mut HbmTensor<f32, Chip, m![T]>,
) {
    let c=rotate::consts::<T,HALF,OnePass>(device,h64,h60);
    let x: Row<bf16,T,HALF,m![H]>=signed_h.to_dm(&mut device.tdma);
    let x: Row<bf16,T,HALF,m![Ka,Kb]>=unsafe{x.reshape()};
    let y=rotate::rotate::<T,HALF>(device,&x,&c,rotate::INV_SQRT_3840);
    let s=scale::<T,HALF>(device,&y);let q=quantize::<T,HALF>(device,&y,&s);let sums_dm=sum_q::<T,HALF>(device,&q);
    let high=hi::<T,HALF>(device,&q);let low=lo::<T,HALF>(device,&q);
    let packed_hi=pack::<T,HALF>(device,high);
    let packed_lo=pack::<T,HALF>(device,low);
    packed_hi.view().to_hbm_view(&mut device.tdma,xhi.view_mut());
    packed_lo.view().to_hbm_view(&mut device.tdma,xlo.view_mut());
    let onehot=rotate::onehot::<T,HALF>(device,eye);
    rotate::store_scales::<T,HALF,m![T]>(device,s,&onehot,sx.view_mut());
    store_sum::<T,HALF>(device,sums_dm,&onehot,sums);
}
