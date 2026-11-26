//! Weight preparation from supplied quantized tensors and precomputed smoothing factors.
use furiosa_opt_std::prelude::*;

use super::{down, ffn_ug::{self, Gt}, heads, hidden, WugRows};
use hidden::{Blk, Hw, Row};
use crate::Chip;
use crate::axes::{H, L, Qf};
use crate::device::tokenwise_int8::gemm::Pi;
use crate::device::tokenwise_int8::passes::OnePass;
use crate::device::tokenwise_int8::qkv::Nq;
use crate::device::tokenwise_int8::rotate::{self, Kh, Rc, Sgi, Sl8, TCl, TSl};
use crate::device::tokenwise_int8::rotate_heads;

axes![ScalePad = 64];

/// Scale replicas make each per-slice BF16 DMA 128 bytes, rather than an illegal two-byte gather.
fn scale_vrf(device: &mut Device, scale: &Row<bf16, Hw, 256, m![ScalePad]>) -> VrfTensor<f32, Chip, TCl<Hw, 256>, TSl<Hw, 256>, m![1 # 8]> {
    let value: Row<f32, Hw, 256, m![1 # 8]> = device.main.begin(scale.view())
        .fetch::<m![ScalePad / 16], m![ScalePad % 16]>().fetch_cast::<f32>()
        .collect::<m![ScalePad / 8], m![ScalePad % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![ScalePad / 4], m![ScalePad % 4]>()
        // All 64 entries are the same positive source row scale; this keeps its exact f32 bits.
        .vector_intra_slice_reduce::<ScalePad, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>().vector_final().commit_trim::<m![1 # 8]>().commit();
    device.sub.begin(value.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf()
}

/// Supplied attention matrix, one row per slice; preserve the BF16 expansion rounding before any rescale.
fn expand_fp8<Q: AxisName>(device: &mut Device, codes: &Row<f8e4m3, Hw, 256, m![Q]>, scales: &Row<bf16, Hw, 256, m![ScalePad]>) -> Row<bf16, Hw, 256, m![Q]> {
    let sv = scale_vrf(device, scales);
    device.main.begin(codes.view())
        .fetch::<m![Q / 32], m![Q % 32]>().fetch_table_lookup::<bf16>().fetch_cast::<f32>()
        .collect::<m![Q / 8], m![Q % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Q / 4], m![Q % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &sv)
        .vector_widen_concat::<m![Q / 8], m![Q % 8]>().vector_final()
        .cast::<bf16, m![Q % 8 # 16]>().commit_trim::<m![Q % 8]>().commit()
}

fn smooth<T: AxisName, Q: AxisName>(device: &mut Device, x: &Row<bf16, T, 256, m![Q]>, factors: &Row<bf16, T, 256, m![Q]>) -> Row<f32, T, 256, m![Q]> {
    // Keep f32 through weight quantization, exactly as the GPU carrier (no extra BF16 underflow boundary).
    device.main.begin_interleaved::<Pi, _, _, _, _, _>(x.view(), factors.view())
        .fetch::<m![Q / 8, Pi], m![Q % 8]>().fetch_cast::<f32>()
        .collect::<m![Q / 8, Pi], m![Q % 8]>()
        .vector_init().vector_intra_slice_unzip::<Pi, m![Q / 8]>()
        .vector_narrow_split::<m![Q / 4], m![Q % 4]>()
        .vector_fp_zip(FpBinaryOp::MulF(FpMulAlu::Mul0))
        .vector_widen_concat::<m![Q / 8], m![Q % 8]>().vector_final()
        .commit_trim::<m![Q % 8]>().commit()
}

/// Per-row INT8 from f32 weights, preserving the screen's smoothing arithmetic.
fn quantize<T: AxisName, Q: AxisName>(device: &mut Device, y: &Row<f32, T, 256, m![Q]>) -> (Row<f32, T, 256, m![1 # 8]>, Row<i8, T, 256, m![Q]>) {
    let s: Row<f32, T, 256, m![1 # 8]> = device
        .main
        .begin(y.view())
        .fetch::<m![Q / 8], m![Q % 8]>()
        .collect::<m![Q / 8], m![Q % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, i32::MAX)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![Q / 4], m![Q % 4]>()
        .vector_intra_slice_reduce::<Q, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_fp_div(127f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1e-30f32)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let sv: VrfTensor<f32, Chip, TCl<T, 256>, TSl<T, 256>, m![1 # 8]> =
        device.sub.begin(s.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let q: Row<i8, T, 256, m![Q]> = device
        .main
        .begin(y.view())
        .fetch::<m![Q / 8], m![Q % 8]>()
        .collect::<m![Q / 8], m![Q % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Q / 4], m![Q % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &sv)
        .vector_fp_binary(FpBinaryOp::AddF, 12_582_912.0f32)
        .vector_widen_concat::<m![Q / 8], m![Q % 8]>()
        .vector_fp_to_fxp(31)
        .vector_clip(ClipBinaryOpI32::AddFxp, -12_582_912)
        .vector_clip(ClipBinaryOpI32::Min, 127)
        .vector_clip(ClipBinaryOpI32::Max, -127)
        .vector_final()
        .cast::<i8, m![Q % 8 # 32]>()
        .commit_trim::<m![Q % 8]>()
        .commit();
    (s, q)
}

fn qkv<N: AxisName, const P: usize>(device: &mut Device, codes: &HbmTensor<f8e4m3, Chip, m![N, H]>, scales: &HbmTensor<bf16, Chip, m![N, ScalePad]>, factors: &HbmTensor<bf16, Chip, m![Rc, H]>, eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>) -> (HbmTensor<i8, Chip, m![N, H]>, HbmTensor<f32, Chip, m![N]>) {
    let f = rotate_heads::signs::<Hw, 256, H, OnePass>(device, factors);
    let onehot = rotate::onehot::<N, 256>(device, eye);
    let mut out = HbmTensor::<i8, Chip, m![N, H]>::new();
    let mut ws = HbmTensor::<f32, Chip, m![N]>::new();
    for p in 0..P {
        let c: Row<f8e4m3, N, 256, m![H]> = codes.view().tile::<m![N / 512], 1, m![N / 512 = 1 # { P }, N % 512, H]>(p).to_dm(&mut device.tdma);
        let c: Row<f8e4m3, Hw, 256, m![H]> = unsafe { c.reshape() };
        let s: Row<bf16, N, 256, m![ScalePad]> = scales.view().tile::<m![N / 512], 1, m![N / 512 = 1 # { P }, N % 512, ScalePad]>(p).to_dm(&mut device.tdma);
        let s: Row<bf16, Hw, 256, m![ScalePad]> = unsafe { s.reshape() };
        let x = expand_fp8::<H>(device, &c, &s);
        let x = smooth::<Hw, H>(device, &x, &f);
        let (s, q) = quantize::<Hw, H>(device, &x);
        let q: Row<i8, N, 256, m![H]> = unsafe { q.reshape() };
        q.view().to_hbm_view(&mut device.tdma, out.view_mut().tile::<m![N / 512], 1, m![N / 512 = 1 #{!} { P }, N % 512, H]>(p));
        rotate::store_scales::<N, 256, m![N / 512 = 1 #{!} { P }, N % 512]>(device, unsafe { s.reshape() }, &onehot,
            ws.view_mut().tile::<m![N / 512], 1, m![N / 512 = 1 #{!} { P }, N % 512]>(p));
    }
    (out, ws)
}


macro_rules! qkv_entry {
    ($name:ident, $n:ident, $p:expr) => {
        #[device(chip = 1)]
        pub fn $name(device: &mut Device, codes: &HbmTensor<f8e4m3, Chip, m![$n, H]>, scales: &HbmTensor<bf16, Chip, m![$n, ScalePad]>, factors: &HbmTensor<bf16, Chip, m![Rc, H]>, eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>) -> (HbmTensor<i8, Chip, m![$n, H]>, HbmTensor<f32, Chip, m![$n]>) {
            qkv::<$n, $p>(device, codes, scales, factors, eye)
        }
    };
}
qkv_entry!(prep_qkv, Nq, 16);


fn o<Q: AxisName>(device: &mut Device, codes: &HbmTensor<f8e4m3, Chip, m![H, Q]>, scales: &HbmTensor<bf16, Chip, m![H, ScalePad]>, factors: &HbmTensor<bf16, Chip, m![Rc, Q]>, eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>) -> (HbmTensor<i8, Chip, m![H, Q]>, HbmTensor<f32, Chip, m![H]>) {
    let f = rotate_heads::signs::<Hw, 256, Q, OnePass>(device, factors);
    let onehot = rotate::onehot::<Hw, 256>(device, eye);
    let mut out = HbmTensor::<i8, Chip, m![H, Q]>::new();
    let mut ws = HbmTensor::<f32, Chip, m![H]>::new();
    // H has fifteen 256-row blocks: the last pair overlaps block13 (identical deterministic output).
    for i in 0..8 {
        let b = if i < 7 { 2 * i } else { 13 };
        let c = hidden::load_blocks::<f8e4m3, m![Q]>(device, codes, b);
        let s = hidden::load_blocks::<bf16, m![ScalePad]>(device, scales, b);
        let x = expand_fp8::<Q>(device, &c, &s);
        let x = smooth::<Hw, Q>(device, &x, &f);
        let (s, q) = quantize::<Hw, Q>(device, &x);
        let q: Blk<i8, m![Q]> = unsafe { q.reshape() };
        q.view().to_hbm_view(&mut device.tdma, out.view_mut().tile::<m![H / 256], 2, m![H / 256 = 2 #{!} 15, H % 256, Q]>(b));
        heads::store_ws(device, s, &onehot, b, &mut ws);
    }
    (out, ws)
}

macro_rules! o_entry {
    ($name:ident, $q:ident) => {
        #[device(chip = 1)]
        pub fn $name(device: &mut Device, codes: &HbmTensor<f8e4m3, Chip, m![H, $q]>, scales: &HbmTensor<bf16, Chip, m![H, ScalePad]>, factors: &HbmTensor<bf16, Chip, m![Rc, $q]>, eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>) -> (HbmTensor<i8, Chip, m![H, $q]>, HbmTensor<f32, Chip, m![H]>) {
            o::<$q>(device, codes, scales, factors, eye)
        }
    };
}
o_entry!(prep_o_full, Qf);







// Apply the per-row calibration ratio during FP8 weight quantization.
fn scale_vrf_mse<T: AxisName>(device: &mut Device, scale: &Row<bf16, T, 256, m![ScalePad]>) -> VrfTensor<f32, Chip, TCl<T, 256>, TSl<T, 256>, m![1 # 8]> {
    let value: Row<f32, T, 256, m![1 # 8]> = device.main.begin(scale.view())
        .fetch::<m![ScalePad / 16], m![ScalePad % 16]>().fetch_cast::<f32>()
        .collect::<m![ScalePad / 8], m![ScalePad % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![ScalePad / 4], m![ScalePad % 4]>()
        // All 64 entries are the same positive source row scale; this keeps its exact f32 bits.
        .vector_intra_slice_reduce::<ScalePad, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>().vector_final().commit_trim::<m![1 # 8]>().commit();
    device.sub.begin(value.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf()
}

fn quantize_fp8_mse<T: AxisName, Q: AxisName>(device: &mut Device, y: &Row<f32, T, 256, m![Q]>, ratio: &Row<bf16, T, 256, m![ScalePad]>) -> (Row<f32, T, 256, m![1 # 8]>, Row<f8e4m3, T, 256, m![Q]>) {
    let s: Row<f32, T, 256, m![1 # 8]> = device
        .main
        .begin(y.view())
        .fetch::<m![Q / 8], m![Q % 8]>()
        .collect::<m![Q / 8], m![Q % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, i32::MAX)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![Q / 4], m![Q % 4]>()
        .vector_intra_slice_reduce::<Q, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_fp_div(256f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1e-30f32)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rv = scale_vrf_mse(device, ratio);
    let s: Row<f32, T, 256, m![1 # 8]> = device.main.begin(s.view())
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &rv)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    let sv: VrfTensor<f32, Chip, TCl<T, 256>, TSl<T, 256>, m![1 # 8]> =
        device.sub.begin(s.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let q: Row<f8e4m3, T, 256, m![Q]> = device
        .main
        .begin(y.view())
        .fetch::<m![Q / 8], m![Q % 8]>()
        .collect::<m![Q / 8], m![Q % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Q / 4], m![Q % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &sv)
        .vector_widen_concat::<m![Q / 8], m![Q % 8]>()
        .vector_clip(ClipBinaryOpF32::Min, 448f32)
        .vector_clip(ClipBinaryOpF32::Max, -448f32)
        .vector_final()
        .cast::<f8e4m3, m![Q % 8 # 32]>()
        .commit_trim::<m![Q % 8]>()
        .commit();
    (s, q)
}
fn ug_pass_fp8_mse(device: &mut Device, packed: &HbmTensor<f4e2m1, Chip, m![L, H]>, blocks: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>, g: &VrfTensor<f32, Chip, TCl<L, 256>, m![1 # 256], m![1 # 8]>, row_factor: &HbmTensor<f32, Chip, m![L]>, factors: &Row<bf16, L, 256, m![H]>, onehot: &VrfTensor<f32, Chip, TCl<L, 256>, m![1 # 256], m![Sgi, Sl8]>, out: &mut HbmTensor<f8e4m3, Chip, m![{ WugRows }, H]>, ws: &mut HbmTensor<f32, Chip, WugRows>, p: usize, gate: usize, ratio: &HbmTensor<bf16, Chip, m![L, ScalePad]>) {
    let c: Row<f4e2m1, L, 256, m![H]> = packed.view().tile::<m![L / 512], 1, m![L / 512 = 1 # 30, L % 512, H]>(p).to_dm(&mut device.tdma);
    let b: Row<f8e4m3, L, 256, m![H / 16]> = blocks.view().tile::<m![L / 512], 1, m![L / 512 = 1 # 30, L % 512, H / 16]>(p).to_dm(&mut device.tdma);
    let c: Row<f4e2m1, L, 256, m![Kh]> = unsafe { c.reshape() };
    let b: Row<f8e4m3, L, 256, m![Kh / 16]> = unsafe { b.reshape() };
    let x = ffn_ug::fp4_decode::<L, 256>(device, &c, &b);
    let x: Row<bf16, L, 256, m![H]> = unsafe { x.reshape() };
    let x = smooth::<L, H>(device, &x, factors);
    let ratio: Row<bf16, L, 256, m![ScalePad]> = ratio.view()
        .tile::<m![L / 512], 1, m![L / 512 = 1 # 30, L % 512, ScalePad]>(p).to_dm(&mut device.tdma);
    let (s, q) = quantize_fp8_mse::<L, H>(device, &x, &ratio);
    // Tile the original view directly because the translator cannot tile a reshaped view.
    q.view().to_hbm_view(&mut device.tdma, out.view_mut()
        .tile::<m![L / 512], 1, m![L / 512 = 1 #{!} 30, L / 256 % 2, L / 16 % 8, L / 128 % 2, L / 8 % 2, Gt, L % 8, H]>(p)
        .tile::<m![Gt], 1, m![L / 512 = 1 #{!} 30, L / 256 % 2, L / 16 % 8, L / 128 % 2, L / 8 % 2, Gt = 1 #{!} 2, L % 8, H]>(gate));
    ffn_ug::store_ws(device, s, onehot, g, row_factor, p, ws.view_mut()
        .tile::<m![L / 512], 1, m![L / 512 = 1 #{!} 30, L / 256 % 2, L / 16 % 8, L / 128 % 2, L / 8 % 2, Gt, L % 8]>(p)
        .tile::<m![Gt], 1, m![L / 512 = 1 #{!} 30, L / 256 % 2, L / 16 % 8, L / 128 % 2, L / 8 % 2, Gt = 1 #{!} 2, L % 8]>(gate));
}

#[device(chip = 1)]
pub fn prep_ug_fp8_mse(device: &mut Device, up: &HbmTensor<f4e2m1, Chip, m![L, H]>, gate: &HbmTensor<f4e2m1, Chip, m![L, H]>, up_blocks: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>, gate_blocks: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>, up_global: &HbmTensor<f32, Chip, m![1]>, gate_global: &HbmTensor<f32, Chip, m![1]>, factors: &HbmTensor<bf16, Chip, m![Rc, H]>, inv_down: &HbmTensor<f32, Chip, m![L]>, ones: &HbmTensor<f32, Chip, m![L]>, eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>, up_ratio: &HbmTensor<bf16, Chip, m![L, ScalePad]>, gate_ratio: &HbmTensor<bf16, Chip, m![L, ScalePad]>) -> (HbmTensor<f8e4m3, Chip, m![{ WugRows }, H]>, HbmTensor<f32, Chip, WugRows>) {
    let f = rotate_heads::signs::<L, 256, H, OnePass>(device, factors);
    let onehot = rotate::onehot::<L, 256>(device, eye);
    let gu = ffn_ug::scalar0(device, up_global);
    let gg = ffn_ug::scalar0(device, gate_global);
    let mut out = HbmTensor::new();
    let mut ws = HbmTensor::new();
    for p in 0..30 {
        ug_pass_fp8_mse(device, up, up_blocks, &gu, inv_down, &f, &onehot, &mut out, &mut ws, p, 0, up_ratio);
        ug_pass_fp8_mse(device, gate, gate_blocks, &gg, ones, &f, &onehot, &mut out, &mut ws, p, 1, gate_ratio);
    }
    (out, ws)
}

#[device(chip = 1)]
pub fn prep_down_fp8_mse(device: &mut Device, packed: &HbmTensor<f4e2m1, Chip, m![H, L]>, blocks: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>, global: &HbmTensor<f32, Chip, m![1]>, factors: &HbmTensor<bf16, Chip, m![Rc, L]>, eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>, ratio: &HbmTensor<bf16, Chip, m![H, ScalePad]>) -> (HbmTensor<f8e4m3, Chip, m![L / 128 % 4, H, L / 512, L % 128]>, HbmTensor<f32, Chip, m![H]>) {
    let f = rotate_heads::signs::<Hw, 256, L, OnePass>(device, factors);
    let onehot = rotate::onehot::<Hw, 256>(device, eye);
    let g = down::scalar0::<Hw, 256>(device, global);
    let mut out = HbmTensor::<f8e4m3, Chip, m![L / 128 % 4, H, L / 512, L % 128]>::new();
    let mut ws = HbmTensor::<f32, Chip, m![H]>::new();
    for i in 0..8 {
        let b = if i < 7 { 2 * i } else { 13 };
        let c = hidden::load_blocks::<f4e2m1, m![L]>(device, packed, b);
        let bs = hidden::load_blocks::<f8e4m3, m![L / 16]>(device, blocks, b);
        let x = down::fp4_decode::<Hw, 256>(device, &c, &bs);
        let x = smooth::<Hw, L>(device, &x, &f);
        let ratio = hidden::load_blocks::<bf16, m![ScalePad]>(device, ratio, b);
        let (s, q) = quantize_fp8_mse::<Hw, L>(device, &x, &ratio);
        // Natural l -> quarter l/128%4, kd = 128*(l/512)+l%128 (CPU-checked fp8c Down layout). The
        // output is declared in these digits so view_mut() tiles directly (the MIR translator rejects
        //.tile() on a reshaped view_mut); the wire order equals [Kq4, H, Kd] -- the host re-spells it
        // on arrival -- so the op's bytes are unchanged.
        let q: Blk<f8e4m3, m![L]> = unsafe { q.reshape() };
        q.view().to_hbm_view(&mut device.tdma, out.view_mut()
            .tile::<m![H / 256], 2, m![L / 128 % 4, H / 256 = 2 #{!} 15, H % 256, L / 512, L % 128]>(b));
        let s = down::compact_ws::<Hw, 256>(device, s, &onehot, &g);
        hidden::store_scale_blocks(device, s, &mut ws, b);
    }
    (out, ws)
}
