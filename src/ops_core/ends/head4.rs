//! W4A8 vocabulary head. Each signed nibble `c` encodes the odd weight `2c + 1`.
//! Each INT8 activation `q` is represented as `16 hi + lo` with two signed nibbles.
//! The dot product is `2 * (16 * dot(hi, c) + dot(lo, c)) + sum(q)`. The device
//! applies the per-row scales and softcap after the integer contraction.

use furiosa_opt_std::prelude::*;
use crate::axes::{H, W, Rc};
use crate::device::tokenwise_int8::rotate::{Ob,Kb,H60Hbm,Sgi,Sl8};
use crate::device::tokenwise_int8::{gemm, gemm16};
use gemm::Pi;
use crate::device::tokenwise_int8::ffn_fast::ug4::Hb;
use crate::Chip;
use super::{head16, HRing, HXs, HYs};

pub mod prep;

type VCl = m![W / 4096 % 2];
type Slices = m![{ HRing }, { HXs }, { HYs }];
type Acc<T> = DmTensor<i32, Chip, VCl, Slices, m![T / 8, T % 8, W / 8 % 2, W % 8]>;
type Float<T> = DmTensor<f32, Chip, VCl, Slices, m![T / 8, T % 8, W / 8 % 2, W % 8]>;
type Rows<T> = VrfTensor<f32, Chip, VCl, Slices, m![T]>;
type WScale = VrfTensor<f32, Chip, VCl, Slices, m![W / 8 % 2, W % 8]>;
type Chunk<K> = DmTensor<i8, Chip, VCl, Slices, m![W % 16, K]>;

pub(super) fn load_chunk<K: AxisName>(device: &mut Device, codes: &HbmTensor<i8, Chip, m![W, K]>, c: usize) -> Chunk<K> {
    codes.view().tile::<m![W / 8192], 1, m![W / 8192 = 1 # 32, W % 8192, K]>(c).to_dm(&mut device.tdma)
}

/// Keep the two plane sums in i32 until both have been combined. The bound
/// 3840*119*15=6,854,400 is below 2^23, so f32 conversion is exact as well.
fn combine<T: AxisName>(device: &mut Device, hi: &Acc<T>, lo: &Acc<T>, sum_q: &Rows<T>) -> Float<T> {
    let high: Acc<T> = device.main.begin(hi.view())
        .fetch::<m![T / 8, T % 8, W / 8 % 2], m![W % 8]>()
        .collect::<m![T / 8, T % 8, W / 8 % 2], m![W % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::LeftShift, 4)
        .vector_final().commit_trim::<m![W % 8]>().commit();
    device.main.begin_interleaved::<Pi, _, _, _, _, _>(high.view(), lo.view())
        .fetch::<m![T / 8, Pi, T % 8, W / 8 % 2], m![W % 8]>()
        .collect::<m![T / 8, Pi, T % 8, W / 8 % 2], m![W % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Pi, m![T / 8, T % 8, W / 8 % 2]>()
        .vector_fxp_zip(FxpBinaryOp::AddFxp)
        .vector_fxp(FxpBinaryOp::LeftShift, 1)
        .vector_fxp_to_fp(31)
        .vector_narrow_split::<m![T / 8, T % 8, W / 8 % 2, W / 4 % 2], m![W % 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, sum_q)
        .vector_widen_concat::<m![T / 8, T % 8, W / 8 % 2], m![W % 8]>()
        .vector_final().commit_trim::<m![W % 8]>().commit()
}

fn chunk<T: AxisName, K: AxisName, const F: usize, const CH: usize, const CH2: usize>(
    device: &mut Device,
    hi: &gemm::XRing<VCl, HRing, m![T / 8], m![T % 8], K, CH>,
    lo: &gemm::XRing<VCl, HRing, m![T / 8], m![T % 8], K, CH>,
    wd: &Chunk<K>, scales: &head16::Scales8<HRing, HXs, HYs>,
    sx: &Rows<T>, sum_q: &Rows<T>, c: usize, out: &mut HbmTensor<bf16, Chip, m![T, W]>,
) {
    // A single full-K packed fill per chunk, shared by both activation planes.
    let wt = gemm::to_trf::<VCl, Slices, W, K, F, CH, CH2, m![W % 16, K]>(device, wd);
    let high = gemm::contract_i32::<VCl, HRing, HXs, HYs, m![T / 8], m![T % 8], W, K, 8, F, CH, CH2>(device, hi, &wt);
    let low = gemm::contract_i32::<VCl, HRing, HXs, HYs, m![T / 8], m![T % 8], W, K, 8, F, CH, CH2>(device, lo, &wt);
    let dot = combine::<T>(device, &high, &low, sum_q);
    // Round to BF16 separately after the activation and weight scales.
    let y: gemm16::Tile<VCl, Slices, m![T / 8], m![T % 8], W> = device.main.begin(dot.view())
        .fetch::<m![T / 8, T % 8, W / 8 % 2], m![W % 8]>()
        .collect::<m![T / 8, T % 8, W / 8 % 2], m![W % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![T / 8, T % 8, W / 8 % 2, W / 4 % 2], m![W % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), sx)
        .vector_widen_concat::<m![T / 8, T % 8, W / 8 % 2], m![W % 8]>()
        .vector_final().cast::<bf16, m![W % 8 # 16]>().commit_trim::<m![W % 8]>().commit();
    let ws: WScale = device.sub.begin(scales.view().tile::<m![W / 8192], 1, m![W / 8192 = 1 # 32, W / 8 % 2, W % 8]>(c))
        .fetch::<m![W / 8 % 2], m![W % 8]>().fetch_cast::<f32>()
        .collect::<m![W / 8 % 2], m![W % 8]>().to_vrf();
    let y = gemm16::scale_cols::<VCl, Slices, m![T / 8], m![T % 8], W>(device, &y, &ws);
    head16::store_chunk::<T, HRing, HXs, HYs, m![T / 8], m![T % 8]>(device, &y, c, out);
}

pub(super) fn stream_prefetched<T:AxisName,K:AxisName,const F:usize,const CH:usize,const CH2:usize>(
    device:&mut Device,xhi:&HbmTensor<i8,Chip,m![T,K]>,xlo:&HbmTensor<i8,Chip,m![T,K]>,
    sx:&HbmTensor<f32,Chip,m![T]>,sum_q:&HbmTensor<f32,Chip,m![T]>,
    codes:&HbmTensor<i8,Chip,m![W,K]>,scales:&head16::Scales8<HRing,HXs,HYs>,first:Chunk<K>,
    out:&mut HbmTensor<bf16,Chip,m![T,W]>,
) {
    let hi = gemm::load_codes::<VCl, HRing, m![T / 8], m![T % 8], K, CH, m![T, K]>(device, xhi.view());
    let lo = gemm::load_codes::<VCl, HRing, m![T / 8], m![T % 8], K, CH, m![T, K]>(device, xlo.view());
    let sx = gemm::row_scales_bcast::<T, VCl, Slices>(device, sx);
    let sum_q = gemm::row_scales_bcast::<T, VCl, Slices>(device, sum_q);
    macro_rules! pipeline {
        ($(($prev:ident, $next:ident, $c:literal)),* ; $last:ident) => {
            $(let $next = load_chunk::<K>(device, codes, $c);
            chunk::<T,K,F,CH,CH2>(device, &hi, &lo, &$prev, scales, &sx, &sum_q, $c-1, out);)*
            chunk::<T,K,F,CH,CH2>(device, &hi, &lo, &$last, scales, &sx, &sum_q, 31, out);
        };
    }
    pipeline!((first,c1,1),(c1,c2,2),(c2,c3,3),(c3,c4,4),(c4,c5,5),(c5,c6,6),(c6,c7,7),
        (c7,c8,8),(c8,c9,9),(c9,c10,10),(c10,c11,11),(c11,c12,12),(c12,c13,13),(c13,c14,14),
        (c14,c15,15),(c15,c16,16),(c16,c17,17),(c17,c18,18),(c18,c19,19),(c19,c20,20),
        (c20,c21,21),(c21,c22,22),(c22,c23,23),(c23,c24,24),(c24,c25,25),(c25,c26,26),
        (c26,c27,27),(c27,c28,28),(c28,c29,29),(c29,c30,30),(c30,c31,31);c31);
}

pub(super) fn logits<T:AxisName,const HALF:usize,const PS:usize>(
    device:&mut Device,h:&HbmTensor<bf16,Chip,m![T,H]>,
    codes:&HbmTensor<i8,Chip,m![W,Hb]>,ws:&head16::Scales8<HRing,HXs,HYs>,first:Chunk<Hb>,
    h64:&HbmTensor<bf16,Chip,m![Rc,Ob,Kb]>,
    h60:&HbmTensor<bf16,Chip,m![Rc,{H60Hbm}]>,
    eye:&HbmTensor<f32,Chip,m![Sgi,Sl8]>,
    logits:&mut HbmTensor<bf16,Chip,m![T,W]>,
) {
    let mut hi=HbmTensor::<i8,Chip,m![T,Hb]>::new();let mut lo=HbmTensor::<i8,Chip,m![T,Hb]>::new();
    let mut sx=HbmTensor::<f32,Chip,m![T]>::new();let mut sums=HbmTensor::<f32,Chip,m![T]>::new();
    prep::prepare::<T,HALF,PS>(device,h,h64,h60,eye,&mut hi,&mut lo,&mut sx,&mut sums);
    stream_prefetched::<T,Hb,15,480,960>(device,&hi,&lo,&sx,&sums,codes,ws,first,logits);
}
