//! Full head normalization and RoPE over `PS = 2 HALF` tokens per pass. Each slice holds
//! one token row and all 16 query heads. Each pass gathers its rotary tables directly into
//! that layout and keeps only the current table half and gamma in VRF.
//!
//! Q and K use `bf16(x * gamma / rms)` and V uses `bf16(x / rms)`. RoPE reads both head
//! halves in pair mode and rounds `x cos + rotate_half(x) sin` to BF16.

use furiosa_opt_std::prelude::*;

use super::gemm::Pi;
use super::passes::{OnePass, TwoPasses};
use super::rotate::{TCl, TSl};
use crate::axes::{Df, E, Gf, Hh, Hqf, Pf, Qf, Rf, Rq, Tok512, Tok1024};
use crate::{Chip, EPS};

const DF_F32: f32 = Df::SIZE as f32;

type Row<D, T, const HALF: usize, EL> = DmTensor<D, Chip, TCl<T, HALF>, TSl<T, HALF>, EL>;
type Vrf<T, const HALF: usize, EL> = VrfTensor<f32, Chip, TCl<T, HALF>, TSl<T, HALF>, EL>;
/// The query heads of a row as the outer axis of `[.., Df]`.
type QO = m![Gf];
/// The single K head.
type KO = m![1];

/// A `[Df]` bf16 row as an f32 VRF operand.
fn vrf_df<T: AxisName, const HALF: usize>(device: &mut Device, x: &Row<bf16, T, HALF, m![Df]>) -> Vrf<T, HALF, m![Df]> {
    device
        .sub
        .begin(x.view())
        .fetch::<m![Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Df / 8], m![Df % 8]>()
        .to_vrf()
}

/// A `[Df]` gamma in every slice's VRF.
fn table_vrf<T: AxisName, const HALF: usize>(device: &mut Device, w: &HbmTensor<bf16, Chip, m![Df]>) -> Vrf<T, HALF, m![Df]> {
    let w: Row<bf16, T, HALF, m![Df]> = w.to_dm(&mut device.tdma);
    vrf_df::<T, HALF>(device, &w)
}

/// `rms = sqrt(fold(x^2) / 512 + eps)` of every head (outer axes `O`), as `heads_full::rms`.
fn rms<T: AxisName, const HALF: usize, O: M>(device: &mut Device, x: &Row<bf16, T, HALF, m![{ O }, Df]>) -> Vrf<T, HALF, m![{ O }, 1 # 8]> {
    let mean_square: Row<f32, T, HALF, m![{ O }, 1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![{ O }, Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![{ O }, Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{ O }, Df / 4], m![Df % 4]>()
        .vector_stash()
        // Delay the stash read by two vector stages to keep the square pipelined.
        .vector_fp_binary(FpBinaryOp::AddF, -0f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<Df, m![{ O }], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DF_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let r: Row<f32, T, HALF, m![{ O }, 1 # 8]> = device
        .main
        .begin(mean_square.view())
        .fetch::<m![{ O }], m![1 # 8]>()
        .collect::<m![{ O }], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    device.sub.begin(r.view()).fetch::<m![{ O }], m![1 # 8]>().collect::<m![{ O }], m![1 # 8]>().to_vrf()
}

/// `bf16(x * gamma / rms)` per head (`heads_full::normalize_weighted`).
fn normalize_weighted<T: AxisName, const HALF: usize, O: M>(
    device: &mut Device,
    x: &Row<bf16, T, HALF, m![{ O }, Df]>,
    w: &Vrf<T, HALF, m![Df]>,
) -> Row<bf16, T, HALF, m![{ O }, Df]> {
    let r = rms::<T, HALF, O>(device, x);
    device
        .main
        .begin(x.view())
        .fetch::<m![{ O }, Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![{ O }, Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{ O }, Df / 4], m![Df % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), w)
        .vector_fp_div(&r)
        .vector_widen_concat::<m![{ O }, Df / 8], m![Df % 8]>()
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit()
}

/// `bf16(x / rms)` per head, no gamma (`heads_full::normalize_value`).
fn normalize_value<T: AxisName, const HALF: usize, O: M>(device: &mut Device, x: &Row<bf16, T, HALF, m![{ O }, Df]>) -> Row<bf16, T, HALF, m![{ O }, Df]> {
    let r = rms::<T, HALF, O>(device, x);
    device
        .main
        .begin(x.view())
        .fetch::<m![{ O }, Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![{ O }, Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{ O }, Df / 4], m![Df % 4]>()
        .vector_fp_div(&r)
        .vector_widen_concat::<m![{ O }, Df / 8], m![Df % 8]>()
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit()
}

/// Half `j` (`Df / 256 = j`) of a `[Df]` bf16 row as an f32 VRF operand.
fn vrf_half<T: AxisName, const HALF: usize>(device: &mut Device, x: &Row<bf16, T, HALF, m![Df]>, j: usize) -> Vrf<T, HALF, m![Df % 256]> {
    device
        .sub
        .begin(x.view().tile::<m![Df / 256], 1, m![Df / 256 = 1 # 2, Df % 256]>(j))
        .fetch::<m![Df % 256 / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Df % 256 / 8], m![Df % 8]>()
        .to_vrf()
}

/// Output half `j` of `rope`: `bf16(f32(x_j) cos_j + f32(x_{1-j}) sin_j)`, the two halves of every
/// head read as one interleaved pair (`proj_fast/heads.rs`'s `rope_pair_half` at `Df`).
fn rope_half<T: AxisName, const HALF: usize, O: M>(
    device: &mut Device,
    x: &Row<bf16, T, HALF, m![{ O }, Df]>,
    cos: &Vrf<T, HALF, m![Df % 256]>,
    sin: &Vrf<T, HALF, m![Df % 256]>,
    j: usize,
    out: &mut Row<bf16, T, HALF, m![{ O }, Df]>,
) {
    device
        .main
        .begin_interleaved::<Pi, _, _, _, _, _>(
            x.view().tile::<m![Df / 256], 1, m![{ O }, Df / 256 = 1 # 2, Df % 256]>(j),
            x.view().tile::<m![Df / 256], 1, m![{ O }, Df / 256 = 1 # 2, Df % 256]>(1 - j),
        )
        .fetch::<m![{ O }, Df % 256 / 8, Pi], m![Df % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![{ O }, Df % 256 / 8, Pi], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Pi, m![{ O }, Df % 256 / 8]>()
        .vector_narrow_split::<m![{ O }, Df % 256 / 4], m![Df % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), cos, ())
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), (), sin)
        .vector_widen_concat::<m![{ O }, Df % 256 / 8], m![Df % 8]>()
        .vector_clip_zip(ClipBinaryOpF32::Add)
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit_view(out.view_mut().tile::<m![Df / 256], 1, m![{ O }, Df / 256 = 1 #{!} 2, Df % 256]>(j));
}

/// Each pass interleaves both activation halves, forms two f32 products and rounds
/// their sum to BF16. The sine table carries the sign of the paired half.
/// Only the current sine and cosine halves occupy VRF.
fn rope_original<T: AxisName, const HALF: usize, O: M>(
    device: &mut Device,
    x: &Row<bf16, T, HALF, m![{ O }, Df]>,
    cos: &Row<bf16, T, HALF, m![Df]>,
    sin: &Row<bf16, T, HALF, m![Df]>,
) -> Row<bf16, T, HALF, m![{ O }, Df]> {
    let mut out: Row<bf16, T, HALF, m![{ O }, Df]> = DmTensor::new();
    {
        let (c, s) = (vrf_half::<T, HALF>(device, cos, 0), vrf_half::<T, HALF>(device, sin, 0));
        rope_half::<T, HALF, O>(device, x, &c, &s, 0, &mut out);
    }
    {
        let (c, s) = (vrf_half::<T, HALF>(device, cos, 1), vrf_half::<T, HALF>(device, sin, 1));
        rope_half::<T, HALF, O>(device, x, &c, &s, 1, &mut out);
    }
    out
}

/// A contiguous part of a Full head. Only the first 64 dimensions of each half rotate.
fn rope_tail<T: AxisName, const HALF: usize, O: M, const WIDTH: usize, const PARTS: usize, const J: usize, const K: usize>(
    device: &mut Device,
    x: &Row<bf16, T, HALF, m![{ O }, Df]>,
    _cos: &Row<bf16, T, HALF, m![Df]>,
    _sin: &Row<bf16, T, HALF, m![Df]>,
    out: &mut Row<bf16, T, HALF, m![{ O }, Df]>,
) {

        // Keep the paired product and add: copying x changes -0 + +0 to -0.
        // Both operands still participate, including NaN/Inf propagation.
        device.main.begin_interleaved::<Pi, _, _, _, _, _>(
            x.view().tile::<m![Df / {WIDTH}], 1, m![{ O }, Df / {WIDTH} = 1 # {PARTS}, Df % {WIDTH}]>(J),
            x.view().tile::<m![Df / {WIDTH}], 1, m![{ O }, Df / {WIDTH} = 1 # {PARTS}, Df % {WIDTH}]>(K),
        )
        .fetch::<m![{ O }, Df % {WIDTH} / 8, Pi], m![Df % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![{ O }, Df % {WIDTH} / 8, Pi], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Pi, m![{ O }, Df % {WIDTH} / 8]>()
        .vector_narrow_split::<m![{ O }, Df % {WIDTH} / 4], m![Df % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1f32, ())
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), (), 0f32)
        .vector_widen_concat::<m![{ O }, Df % {WIDTH} / 8], m![Df % 8]>()
        .vector_clip_zip(ClipBinaryOpF32::Add)
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit_view(out.view_mut().tile::<m![Df / {WIDTH}], 1, m![{ O }, Df / {WIDTH} = 1 #{!} {PARTS}, Df % {WIDTH}]>(J));
}

fn rope_active<T: AxisName, const HALF: usize, O: M, const WIDTH: usize, const PARTS: usize, const J: usize, const K: usize>(
    device: &mut Device,
    x: &Row<bf16, T, HALF, m![{ O }, Df]>,
    cos: &Row<bf16, T, HALF, m![Df]>,
    sin: &Row<bf16, T, HALF, m![Df]>,
    out: &mut Row<bf16, T, HALF, m![{ O }, Df]>,
) {

        let c: Vrf<T, HALF, m![Df % {WIDTH}]> = device.sub
            .begin(cos.view().tile::<m![Df / {WIDTH}], 1, m![Df / {WIDTH} = 1 # {PARTS}, Df % {WIDTH}]>(J))
            .fetch::<m![Df % {WIDTH} / 16], m![Df % 16]>().fetch_cast::<f32>()
            .collect::<m![Df % {WIDTH} / 8], m![Df % 8]>().to_vrf();
        let s: Vrf<T, HALF, m![Df % {WIDTH}]> = device.sub
            .begin(sin.view().tile::<m![Df / {WIDTH}], 1, m![Df / {WIDTH} = 1 # {PARTS}, Df % {WIDTH}]>(J))
            .fetch::<m![Df % {WIDTH} / 16], m![Df % 16]>().fetch_cast::<f32>()
            .collect::<m![Df % {WIDTH} / 8], m![Df % 8]>().to_vrf();
        device.main.begin_interleaved::<Pi, _, _, _, _, _>(
            x.view().tile::<m![Df / {WIDTH}], 1, m![{ O }, Df / {WIDTH} = 1 # {PARTS}, Df % {WIDTH}]>(J),
            x.view().tile::<m![Df / {WIDTH}], 1, m![{ O }, Df / {WIDTH} = 1 # {PARTS}, Df % {WIDTH}]>(K),
        )
        .fetch::<m![{ O }, Df % {WIDTH} / 8, Pi], m![Df % 8]>().fetch_cast::<f32>()
        .collect::<m![{ O }, Df % {WIDTH} / 8, Pi], m![Df % 8]>()
        .vector_init().vector_intra_slice_unzip::<Pi, m![{ O }, Df % {WIDTH} / 8]>()
        .vector_narrow_split::<m![{ O }, Df % {WIDTH} / 4], m![Df % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &c, ())
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), (), &s)
        .vector_widen_concat::<m![{ O }, Df % {WIDTH} / 8], m![Df % 8]>()
        .vector_clip_zip(ClipBinaryOpF32::Add).vector_final()
        .cast::<bf16, m![Df % 8 # 16]>().commit_trim::<m![Df % 8]>()
        .commit_view(out.view_mut().tile::<m![Df / {WIDTH}], 1, m![{ O }, Df / {WIDTH} = 1 #{!} {PARTS}, Df % {WIDTH}]>(J));
}

axes![RopeHalf = 2, RopeBlock = 4, RopeElem = 64];

/// Relabel the whole head before taking views, preserving its physical 2x4x64 order.
/// Keeping the 3-block tail as a separate named axis avoids the compiler folding it
/// into an opaque 192-of-512 crop.
fn rope_tail_compact<T: AxisName, const HALF: usize, O: M, const J: usize, const K: usize>(
    device: &mut Device,
    x: &DmTensorView<'_, bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![{ O }, RopeHalf, RopeBlock, RopeElem]>,
    out: &mut Row<bf16, T, HALF, m![{ O }, RopeHalf, RopeBlock, RopeElem]>,
) {
    device.main.begin_interleaved::<Pi, _, _, _, _, _>(
        x.tile::<m![RopeHalf], 1, m![{ O }, RopeHalf = 1 # 2, RopeBlock, RopeElem]>(J)
            .tile::<m![RopeBlock], 3, m![{ O }, RopeHalf = 1 # 2, RopeBlock = 3 # 4, RopeElem]>(1),
        x.tile::<m![RopeHalf], 1, m![{ O }, RopeHalf = 1 # 2, RopeBlock, RopeElem]>(K)
            .tile::<m![RopeBlock], 3, m![{ O }, RopeHalf = 1 # 2, RopeBlock = 3 # 4, RopeElem]>(1),
    )
    .fetch::<m![{ O }, RopeBlock = 3, RopeElem / 8, Pi], m![RopeElem % 8]>()
    .fetch_cast::<f32>()
    .collect::<m![{ O }, RopeBlock = 3, RopeElem / 8, Pi], m![RopeElem % 8]>()
    .vector_init()
    .vector_intra_slice_unzip::<Pi, m![{ O }, RopeBlock = 3, RopeElem / 8]>()
    .vector_narrow_split::<m![{ O }, RopeBlock = 3, RopeElem / 4], m![RopeElem % 4]>()
    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), (), 0f32)
    .vector_widen_concat::<m![{ O }, RopeBlock = 3, RopeElem / 8], m![RopeElem % 8]>()
    .vector_clip_zip(ClipBinaryOpF32::Add).vector_final()
    .cast::<bf16, m![RopeElem % 8 # 16]>().commit_trim::<m![RopeElem % 8]>()
    .commit_view(out.view_mut().tile::<m![RopeHalf], 1, m![{ O }, RopeHalf = 1 #{!} 2, RopeBlock, RopeElem]>(J)
        .tile::<m![RopeBlock], 3, m![{ O }, RopeHalf = 1 #{!} 2, RopeBlock = 3 #{!} 4, RopeElem]>(1));
}
fn rope_active_compact<T: AxisName, const HALF: usize, O: M, const J: usize, const K: usize>(
    device: &mut Device,
    x: &DmTensorView<'_, bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![{ O }, RopeHalf, RopeBlock, RopeElem]>,
    cos: &Row<bf16, T, HALF, m![Df]>,
    sin: &Row<bf16, T, HALF, m![Df]>,
    out: &mut Row<bf16, T, HALF, m![{ O }, RopeHalf, RopeBlock, RopeElem]>,
) {
    let cos: DmTensorView<'_, bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![RopeHalf, RopeBlock, RopeElem]> = unsafe { cos.view().reshape() };
    let sin: DmTensorView<'_, bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![RopeHalf, RopeBlock, RopeElem]> = unsafe { sin.view().reshape() };
    let c: Vrf<T, HALF, m![RopeElem]> = device.sub
        .begin(cos.tile::<m![RopeHalf], 1, m![RopeHalf = 1 # 2, RopeBlock, RopeElem]>(J).tile::<m![RopeBlock], 1, m![RopeHalf = 1 # 2, RopeBlock = 1 # 4, RopeElem]>(0))
        .fetch::<m![RopeElem / 16], m![RopeElem % 16]>().fetch_cast::<f32>()
        .collect::<m![RopeElem / 8], m![RopeElem % 8]>().to_vrf();
    let s: Vrf<T, HALF, m![RopeElem]> = device.sub
        .begin(sin.tile::<m![RopeHalf], 1, m![RopeHalf = 1 # 2, RopeBlock, RopeElem]>(J).tile::<m![RopeBlock], 1, m![RopeHalf = 1 # 2, RopeBlock = 1 # 4, RopeElem]>(0))
        .fetch::<m![RopeElem / 16], m![RopeElem % 16]>().fetch_cast::<f32>()
        .collect::<m![RopeElem / 8], m![RopeElem % 8]>().to_vrf();
    device.main.begin_interleaved::<Pi, _, _, _, _, _>(
        x.tile::<m![RopeHalf], 1, m![{ O }, RopeHalf = 1 # 2, RopeBlock, RopeElem]>(J)
            .tile::<m![RopeBlock], 1, m![{ O }, RopeHalf = 1 # 2, RopeBlock = 1 # 4, RopeElem]>(0),
        x.tile::<m![RopeHalf], 1, m![{ O }, RopeHalf = 1 # 2, RopeBlock, RopeElem]>(K)
            .tile::<m![RopeBlock], 1, m![{ O }, RopeHalf = 1 # 2, RopeBlock = 1 # 4, RopeElem]>(0),
    )
    .fetch::<m![{ O }, RopeElem / 8, Pi], m![RopeElem % 8]>().fetch_cast::<f32>()
    .collect::<m![{ O }, RopeElem / 8, Pi], m![RopeElem % 8]>()
    .vector_init().vector_intra_slice_unzip::<Pi, m![{ O }, RopeElem / 8]>()
    .vector_narrow_split::<m![{ O }, RopeElem / 4], m![RopeElem % 4]>()
    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &c, ())
    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), (), &s)
    .vector_widen_concat::<m![{ O }, RopeElem / 8], m![RopeElem % 8]>()
    .vector_clip_zip(ClipBinaryOpF32::Add).vector_final()
    .cast::<bf16, m![RopeElem % 8 # 16]>().commit_trim::<m![RopeElem % 8]>()
    .commit_view(out.view_mut().tile::<m![RopeHalf], 1, m![{ O }, RopeHalf = 1 #{!} 2, RopeBlock, RopeElem]>(J)
        .tile::<m![RopeBlock], 1, m![{ O }, RopeHalf = 1 #{!} 2, RopeBlock = 1 #{!} 4, RopeElem]>(0));
}

fn rope_compact<T: AxisName, const HALF: usize, O: M>(
    device: &mut Device,
    x: &Row<bf16, T, HALF, m![{ O }, Df]>,
    cos: &Row<bf16, T, HALF, m![Df]>,
    sin: &Row<bf16, T, HALF, m![Df]>,
) -> Row<bf16, T, HALF, m![{ O }, Df]> {
    let x: DmTensorView<'_, bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![{ O }, RopeHalf, RopeBlock, RopeElem]> = unsafe { x.view().reshape() };
    let mut out: Row<bf16, T, HALF, m![{ O }, RopeHalf, RopeBlock, RopeElem]> = DmTensor::new();
    rope_active_compact::<T, HALF, O, 0, 1>(device, &x, cos, sin, &mut out);
    rope_tail_compact::<T, HALF, O, 0, 1>(device, &x, &mut out);
    rope_active_compact::<T, HALF, O, 1, 0>(device, &x, cos, sin, &mut out);
    rope_tail_compact::<T, HALF, O, 1, 0>(device, &x, &mut out);
    unsafe { out.reshape() }
}

fn rope_split<T: AxisName,const HALF:usize,O:M>(device:&mut Device,x:&Row<bf16,T,HALF,m![{O},Df]>,cos:&Row<bf16,T,HALF,m![Df]>,sin:&Row<bf16,T,HALF,m![Df]>)->Row<bf16,T,HALF,m![{O},Df]>{
 let mut out=DmTensor::new();
 rope_active::<T,HALF,O,64,8,0,4>(device,x,cos,sin,&mut out);
 rope_tail::<T,HALF,O,64,8,1,5>(device,x,cos,sin,&mut out);
 rope_tail::<T,HALF,O,128,4,1,3>(device,x,cos,sin,&mut out);
 rope_active::<T,HALF,O,64,8,4,0>(device,x,cos,sin,&mut out);
 rope_tail::<T,HALF,O,64,8,5,1>(device,x,cos,sin,&mut out);
 rope_tail::<T,HALF,O,128,4,3,1>(device,x,cos,sin,&mut out);
 out
}

/// Compile-time selection of the measured faster scatter layout for each bucket.
/// T512 uses four compact passes; T1024 uses six aligned passes. Small buckets keep
/// the original arithmetic/layout. A scalar `if` here would create device control flow.
trait RopePolicy: AxisName + Sized {
    fn apply<const HALF: usize, O: M>(device: &mut Device,
        x: &Row<bf16, Self, HALF, m![{O}, Df]>, cos: &Row<bf16, Self, HALF, m![Df]>,
        sin: &Row<bf16, Self, HALF, m![Df]>) -> Row<bf16, Self, HALF, m![{O}, Df]>;
}
impl<T: AxisName> RopePolicy for T {
    default fn apply<const HALF: usize, O: M>(device: &mut Device,
        x: &Row<bf16, Self, HALF, m![{O}, Df]>, cos: &Row<bf16, Self, HALF, m![Df]>,
        sin: &Row<bf16, Self, HALF, m![Df]>) -> Row<bf16, Self, HALF, m![{O}, Df]> {
        rope_original::<Self, HALF, O>(device, x, cos, sin)
    }
}
macro_rules! measured_rope_policy {
    ($t:ty, $f:ident) => {
        impl RopePolicy for $t {
            fn apply<const HALF: usize, O: M>(device: &mut Device,
                x: &Row<bf16, Self, HALF, m![{O}, Df]>, cos: &Row<bf16, Self, HALF, m![Df]>,
                sin: &Row<bf16, Self, HALF, m![Df]>) -> Row<bf16, Self, HALF, m![{O}, Df]> {
                $f::<Self, HALF, O>(device, x, cos, sin)
            }
        }
    };
}
measured_rope_policy!(Tok512, rope_compact);
measured_rope_policy!(Tok1024, rope_split);

fn rope<T: AxisName, const HALF: usize, O: M>(device: &mut Device,
    x: &Row<bf16, T, HALF, m![{O}, Df]>, cos: &Row<bf16, T, HALF, m![Df]>,
    sin: &Row<bf16, T, HALF, m![Df]>) -> Row<bf16, T, HALF, m![{O}, Df]> {
    <T as RopePolicy>::apply::<HALF, O>(device, x, cos, sin)
}

/// `heads_pass` with the rows scattered from DM straight to their destinations: V and
/// K into the Full pools at the pass's `publish_f` rows, the query halves into the Full row space
/// `qin_f` at its `rows_f` (as `segments_full::full_prep`), so the stage stores no q/k/v to HBM and
/// reloads none. The same loads, arithmetic and order (v, k, q), so the same bits.
#[allow(clippy::too_many_arguments)]
fn heads_pass_rows<T: AxisName, const HALF: usize, const PS: usize, const P: usize, const PI: usize>(
    device: &mut Device,
    q_raw: &HbmTensor<bf16, Chip, m![T, Qf]>,
    k_raw: &HbmTensor<bf16, Chip, m![T, Pf]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
    rope_offsets: &HbmTensor<i32, Chip, m![T]>,
    cos: &HbmTensor<bf16, Chip, m![E, Df]>,
    sin: &HbmTensor<bf16, Chip, m![E, Df]>,
    rows_f: &HbmTensor<i32, Chip, m![T, Hh]>,
    publish_f: &HbmTensor<i32, Chip, m![T]>,
    qin_f: &mut HbmTensor<bf16, Chip, m![Rq, Hqf]>,
    k_n: &mut HbmTensor<f8e4m3, Chip, m![Rf, Df]>,
    v_n: &mut HbmTensor<f8e4m3, Chip, m![Rf, Df]>,
) {
    // The pass's scatter tables.
    let mut prow = HbmTensor::<i32, Chip, m![T % { PS }, Hh]>::new();
    rows_f
        .view()
        .tile::<m![T / { PS }], 1, m![T / { PS } = 1 # { P }, T % { PS }, Hh]>(PI)
        .to_hbm_view(&mut device.tdma, prow.view_mut());
    let mut ppub = HbmTensor::<i32, Chip, m![T % { PS }]>::new();
    publish_f
        .view()
        .tile::<m![T / { PS }], 1, m![T / { PS } = 1 # { P }, T % { PS }]>(PI)
        .to_hbm_view(&mut device.tdma, ppub.view_mut());
    // Loads on the base axes, then relabeled.
    let k: Row<bf16, T, HALF, m![Pf]> = k_raw.view().tile::<m![T / { PS }], 1, m![T / { PS } = 1 # { P }, T % { PS }, Pf]>(PI).to_dm(&mut device.tdma);
    let k: Row<bf16, T, HALF, m![{ KO }, Df]> = unsafe { k.reshape() };
    let q: Row<bf16, T, HALF, m![Qf]> = q_raw.view().tile::<m![T / { PS }], 1, m![T / { PS } = 1 # { P }, T % { PS }, Qf]>(PI).to_dm(&mut device.tdma);
    let q: Row<bf16, T, HALF, m![{ QO }, Df]> = unsafe { q.reshape() };
    let mut offsets = HbmTensor::<i32, Chip, m![T % { PS }]>::new();
    rope_offsets
        .view()
        .tile::<m![T / { PS }], 1, m![T / { PS } = 1 # { P }, T % { PS }]>(PI)
        .to_hbm_view(&mut device.tdma, offsets.view_mut());
    let c: Row<bf16, T, HALF, m![Df]> = cos.gather::<m![E], m![Df]>().by_byte_offsets(&offsets).to_dm(&mut device.tdma);
    let s: Row<bf16, T, HALF, m![Df]> = sin.gather::<m![E], m![Df]>().by_byte_offsets(&offsets).to_dm(&mut device.tdma);

    // V: the weightless norm of the raw K.
    let v = normalize_value::<T, HALF, KO>(device, &k);
    let v: Row<bf16, T, HALF, m![Df]> = unsafe { v.reshape() };
    let v = crate::ops_core::moves::e4m3_rows_cl::<TCl<T, HALF>, TSl<T, HALF>, m![Df], Df, m![Df / 16], m![Df / 8]>(device, &v);
    v_n.view_mut().scatter::<m![Rf], m![Df]>().by_byte_offsets(ppub.view()).from_dm_view(&mut device.tdma, v.view());

    // K: weighted norm, RoPE.
    let k = {
        let k_gamma = table_vrf::<T, HALF>(device, k_rms_weight);
        normalize_weighted::<T, HALF, KO>(device, &k, &k_gamma)
    };
    let k = rope::<T, HALF, KO>(device, &k, &c, &s);
    let k: Row<bf16, T, HALF, m![Df]> = unsafe { k.reshape() };
    let k = crate::ops_core::moves::e4m3_rows_cl::<TCl<T, HALF>, TSl<T, HALF>, m![Df], Df, m![Df / 16], m![Df / 8]>(device, &k);
    k_n.view_mut().scatter::<m![Rf], m![Df]>().by_byte_offsets(ppub.view()).from_dm_view(&mut device.tdma, k.view());

    // Q: weighted norm, RoPE, every head of the row; the two halves to their rows.
    let q = {
        let q_gamma = table_vrf::<T, HALF>(device, q_rms_weight);
        normalize_weighted::<T, HALF, QO>(device, &q, &q_gamma)
    };
    let q = rope::<T, HALF, QO>(device, &q, &c, &s);
    let q: Row<bf16, T, HALF, m![Hh, Hqf]> = unsafe { q.reshape() };
    qin_f.view_mut().scatter::<m![Rq], m![Hqf]>().by_byte_offsets(prow.view()).from_dm_view(&mut device.tdma, q.view());
}

/// The head phase with the moves fused in (`heads_pass_rows`), over all passes.
#[allow(clippy::too_many_arguments)]
pub(crate) trait FullHeadsRowsPasses {
    fn heads_rows<T: AxisName, const HALF: usize, const PS: usize>(
        device: &mut Device,
        q_raw: &HbmTensor<bf16, Chip, m![T, Qf]>,
        k_raw: &HbmTensor<bf16, Chip, m![T, Pf]>,
        q_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
        k_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
        rope_offsets: &HbmTensor<i32, Chip, m![T]>,
        cos: &HbmTensor<bf16, Chip, m![E, Df]>,
        sin: &HbmTensor<bf16, Chip, m![E, Df]>,
        rows_f: &HbmTensor<i32, Chip, m![T, Hh]>,
        publish_f: &HbmTensor<i32, Chip, m![T]>,
        qin_f: &mut HbmTensor<bf16, Chip, m![Rq, Hqf]>,
        k_n: &mut HbmTensor<f8e4m3, Chip, m![Rf, Df]>,
        v_n: &mut HbmTensor<f8e4m3, Chip, m![Rf, Df]>,
    );
}

macro_rules! full_heads_rows_passes_impl {
    ($ty:ty, $p:literal, [$($i:literal),*]) => {
        impl FullHeadsRowsPasses for $ty {
            fn heads_rows<T: AxisName, const HALF: usize, const PS: usize>(
                device: &mut Device,
                q_raw: &HbmTensor<bf16, Chip, m![T, Qf]>,
                k_raw: &HbmTensor<bf16, Chip, m![T, Pf]>,
                q_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
                k_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
                rope_offsets: &HbmTensor<i32, Chip, m![T]>,
                cos: &HbmTensor<bf16, Chip, m![E, Df]>,
                sin: &HbmTensor<bf16, Chip, m![E, Df]>,
                rows_f: &HbmTensor<i32, Chip, m![T, Hh]>,
                publish_f: &HbmTensor<i32, Chip, m![T]>,
                qin_f: &mut HbmTensor<bf16, Chip, m![Rq, Hqf]>,
                k_n: &mut HbmTensor<f8e4m3, Chip, m![Rf, Df]>,
                v_n: &mut HbmTensor<f8e4m3, Chip, m![Rf, Df]>,
            ) {
                $(heads_pass_rows::<T, HALF, PS, $p, $i>(
                    device, q_raw, k_raw, q_rms_weight, k_rms_weight, rope_offsets, cos, sin, rows_f, publish_f, qin_f, k_n, v_n,
                );)*
            }
        }
    };
}

full_heads_rows_passes_impl!(OnePass, 1, [0]);
full_heads_rows_passes_impl!(TwoPasses, 2, [0, 1]);

pub(crate) struct OneRowPerSlice;

/// The Full head phase with the moves fused in: `OneRowPerSlice`'s passes scatter V
/// and K into the Full pools and the query halves into the row space from DM
/// (`FullHeadsRowsPasses`), replacing the stage's q/k/v stores and `segments_full::full_prep`.
#[allow(clippy::too_many_arguments)]
pub(crate) trait FullHeadPhaseRows {
    fn run_rows<T: AxisName, const HALF: usize, const PS2: usize, PP: FullHeadsRowsPasses>(
        device: &mut Device,
        q_raw: &HbmTensor<bf16, Chip, m![T, Qf]>,
        k_raw: &HbmTensor<bf16, Chip, m![T, Pf]>,
        q_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
        k_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
        rope_offsets: &HbmTensor<i32, Chip, m![T]>,
        cos: &HbmTensor<bf16, Chip, m![E, Df]>,
        sin: &HbmTensor<bf16, Chip, m![E, Df]>,
        rows_f: &HbmTensor<i32, Chip, m![T, Hh]>,
        publish_f: &HbmTensor<i32, Chip, m![T]>,
        qin_f: &mut HbmTensor<bf16, Chip, m![Rq, Hqf]>,
        k_n: &mut HbmTensor<f8e4m3, Chip, m![Rf, Df]>,
        v_n: &mut HbmTensor<f8e4m3, Chip, m![Rf, Df]>,
    );
}

impl FullHeadPhaseRows for OneRowPerSlice {
    fn run_rows<T: AxisName, const HALF: usize, const PS2: usize, PP: FullHeadsRowsPasses>(
        device: &mut Device,
        q_raw: &HbmTensor<bf16, Chip, m![T, Qf]>,
        k_raw: &HbmTensor<bf16, Chip, m![T, Pf]>,
        q_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
        k_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
        rope_offsets: &HbmTensor<i32, Chip, m![T]>,
        cos: &HbmTensor<bf16, Chip, m![E, Df]>,
        sin: &HbmTensor<bf16, Chip, m![E, Df]>,
        rows_f: &HbmTensor<i32, Chip, m![T, Hh]>,
        publish_f: &HbmTensor<i32, Chip, m![T]>,
        qin_f: &mut HbmTensor<bf16, Chip, m![Rq, Hqf]>,
        k_n: &mut HbmTensor<f8e4m3, Chip, m![Rf, Df]>,
        v_n: &mut HbmTensor<f8e4m3, Chip, m![Rf, Df]>,
    ) {
        PP::heads_rows::<T, HALF, PS2>(device, q_raw, k_raw, q_rms_weight, k_rms_weight, rope_offsets, cos, sin, rows_f, publish_f, qin_f, k_n, v_n);
    }
}

/// Use the compact Full RoPE arithmetic from the 512-row bucket for the 128-row
/// Sliding-to-Full segment. The 128-row bucket preserves the 512-row numerics.
#[allow(clippy::too_many_arguments)]
pub(crate) fn heads_rows_compact_one_pass<T: AxisName, const HALF: usize, const PS: usize>(
    device: &mut Device,
    q_raw: &HbmTensor<bf16, Chip, m![T, Qf]>,
    k_raw: &HbmTensor<bf16, Chip, m![T, Pf]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
    rope_offsets: &HbmTensor<i32, Chip, m![T]>,
    cos: &HbmTensor<bf16, Chip, m![E, Df]>,
    sin: &HbmTensor<bf16, Chip, m![E, Df]>,
    rows_f: &HbmTensor<i32, Chip, m![T, Hh]>,
    publish_f: &HbmTensor<i32, Chip, m![T]>,
    qin_f: &mut HbmTensor<bf16, Chip, m![Rq, Hqf]>,
    k_n: &mut HbmTensor<f8e4m3, Chip, m![Rf, Df]>,
    v_n: &mut HbmTensor<f8e4m3, Chip, m![Rf, Df]>,
) {
    // The pass's scatter tables.
    let mut prow = HbmTensor::<i32, Chip, m![T % { PS }, Hh]>::new();
    rows_f
        .view()
        .tile::<m![T / { PS }], 1, m![T / { PS } = 1 # 1, T % { PS }, Hh]>(0)
        .to_hbm_view(&mut device.tdma, prow.view_mut());
    let mut ppub = HbmTensor::<i32, Chip, m![T % { PS }]>::new();
    publish_f
        .view()
        .tile::<m![T / { PS }], 1, m![T / { PS } = 1 # 1, T % { PS }]>(0)
        .to_hbm_view(&mut device.tdma, ppub.view_mut());
    // Loads on the base axes, then relabeled.
    let k: Row<bf16, T, HALF, m![Pf]> = k_raw.view().tile::<m![T / { PS }], 1, m![T / { PS } = 1 # 1, T % { PS }, Pf]>(0).to_dm(&mut device.tdma);
    let k: Row<bf16, T, HALF, m![{ KO }, Df]> = unsafe { k.reshape() };
    let q: Row<bf16, T, HALF, m![Qf]> = q_raw.view().tile::<m![T / { PS }], 1, m![T / { PS } = 1 # 1, T % { PS }, Qf]>(0).to_dm(&mut device.tdma);
    let q: Row<bf16, T, HALF, m![{ QO }, Df]> = unsafe { q.reshape() };
    let mut offsets = HbmTensor::<i32, Chip, m![T % { PS }]>::new();
    rope_offsets
        .view()
        .tile::<m![T / { PS }], 1, m![T / { PS } = 1 # 1, T % { PS }]>(0)
        .to_hbm_view(&mut device.tdma, offsets.view_mut());
    let c: Row<bf16, T, HALF, m![Df]> = cos.gather::<m![E], m![Df]>().by_byte_offsets(&offsets).to_dm(&mut device.tdma);
    let s: Row<bf16, T, HALF, m![Df]> = sin.gather::<m![E], m![Df]>().by_byte_offsets(&offsets).to_dm(&mut device.tdma);

    // V: the weightless norm of the raw K.
    let v = normalize_value::<T, HALF, KO>(device, &k);
    let v: Row<bf16, T, HALF, m![Df]> = unsafe { v.reshape() };
    let v = crate::ops_core::moves::e4m3_rows_cl::<TCl<T, HALF>, TSl<T, HALF>, m![Df], Df, m![Df / 16], m![Df / 8]>(device, &v);
    v_n.view_mut().scatter::<m![Rf], m![Df]>().by_byte_offsets(ppub.view()).from_dm_view(&mut device.tdma, v.view());

    // K: weighted norm, RoPE.
    let k = {
        let k_gamma = table_vrf::<T, HALF>(device, k_rms_weight);
        normalize_weighted::<T, HALF, KO>(device, &k, &k_gamma)
    };
    let k = rope_compact::<T, HALF, KO>(device, &k, &c, &s);
    let k: Row<bf16, T, HALF, m![Df]> = unsafe { k.reshape() };
    let k = crate::ops_core::moves::e4m3_rows_cl::<TCl<T, HALF>, TSl<T, HALF>, m![Df], Df, m![Df / 16], m![Df / 8]>(device, &k);
    k_n.view_mut().scatter::<m![Rf], m![Df]>().by_byte_offsets(ppub.view()).from_dm_view(&mut device.tdma, k.view());

    // Q: weighted norm, RoPE, every head of the row; the two halves to their rows.
    let q = {
        let q_gamma = table_vrf::<T, HALF>(device, q_rms_weight);
        normalize_weighted::<T, HALF, QO>(device, &q, &q_gamma)
    };
    let q = rope_compact::<T, HALF, QO>(device, &q, &c, &s);
    let q: Row<bf16, T, HALF, m![Hh, Hqf]> = unsafe { q.reshape() };
    qin_f.view_mut().scatter::<m![Rq], m![Hqf]>().by_byte_offsets(prow.view()).from_dm_view(&mut device.tdma, q.view());
}

#[cfg(feature = "compile-carried")]
use crate::device::tokenwise_int8::passes::ThreePasses;

#[cfg(feature = "compile-carried")]
full_heads_rows_passes_impl!(ThreePasses, 3, [0, 1, 2]);
