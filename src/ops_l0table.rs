//! Weight-only layer-0 table preparation. No activation or weight re-quantization:
//! FP32 inputs arrive as three BF16 terms and contractions retain FP32 sums.
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{H, Tok1024};
use crate::device::tokenwise_int8::gemm16::{self, Acc};

use crate::device::tokenwise_int8::qkv::{Nq, QkvCl};

pub mod post;
pub mod serve;

type Sl = m![1, Nq / 64 % 64, Nq / 16 % 4];
type Sums = Acc<QkvCl, Sl, m![Tok1024 / 64], m![Tok1024 % 64], Nq>;

fn half(
    device: &mut Device,
    hi: &HbmTensor<bf16, Chip, m![Tok1024, H]>,
    mid: &HbmTensor<bf16, Chip, m![Tok1024, H]>,
    lo: &HbmTensor<bf16, Chip, m![Tok1024, H]>,
    weight: &HbmTensor<bf16, Chip, m![Nq, H]>,
    p: usize,
) -> Sums {
    let wd: DmTensor<bf16, Chip, QkvCl, Sl, m![Nq % 16, H % 1920]> = weight
        .view()
        .tile::<m![H / 1920], 1, m![Nq, H / 1920 = 1 # 2, H % 1920]>(p)
        .to_dm(&mut device.tdma);
    let w = gemm16::to_trf::<QkvCl, Sl, Nq, H, 30, 480, 960, m![Nq % 16, H % 1920]>(device, &wd);
    let xh: gemm16::XRing16<QkvCl, m![1], m![Tok1024 / 64], m![Tok1024 % 64], H, 480> = hi
        .view()
        .tile::<m![H / 1920], 1, m![Tok1024, H / 1920 = 1 # 2, H % 1920]>(p)
        .to_dm(&mut device.tdma);
    let a = gemm16::contract::<
        QkvCl,
        m![1],
        m![Nq / 64 % 64],
        m![Nq / 16 % 4],
        m![Tok1024 / 64],
        m![Tok1024 % 64],
        Nq,
        H,
        64,
        30,
        480,
        960,
    >(device, &xh, &w);
    let xm: gemm16::XRing16<QkvCl, m![1], m![Tok1024 / 64], m![Tok1024 % 64], H, 480> = mid
        .view()
        .tile::<m![H / 1920], 1, m![Tok1024, H / 1920 = 1 # 2, H % 1920]>(p)
        .to_dm(&mut device.tdma);
    let b = gemm16::contract::<
        QkvCl,
        m![1],
        m![Nq / 64 % 64],
        m![Nq / 16 % 4],
        m![Tok1024 / 64],
        m![Tok1024 % 64],
        Nq,
        H,
        64,
        30,
        480,
        960,
    >(device, &xm, &w);
    let ab = gemm16::add::<QkvCl, Sl, m![Tok1024 / 64], m![Tok1024 % 64], Nq>(device, &a, &b);
    let xl: gemm16::XRing16<QkvCl, m![1], m![Tok1024 / 64], m![Tok1024 % 64], H, 480> = lo
        .view()
        .tile::<m![H / 1920], 1, m![Tok1024, H / 1920 = 1 # 2, H % 1920]>(p)
        .to_dm(&mut device.tdma);
    let c = gemm16::contract::<
        QkvCl,
        m![1],
        m![Nq / 64 % 64],
        m![Nq / 16 % 4],
        m![Tok1024 / 64],
        m![Tok1024 % 64],
        Nq,
        H,
        64,
        30,
        480,
        960,
    >(device, &xl, &w);
    gemm16::add::<QkvCl, Sl, m![Tok1024 / 64], m![Tok1024 % 64], Nq>(device, &ab, &c)
}

#[device(chip = 1)]
pub fn prepare_project_t1024(
    device: &mut Device,
    hi: &HbmTensor<bf16, Chip, m![Tok1024, H]>,
    mid: &HbmTensor<bf16, Chip, m![Tok1024, H]>,
    lo: &HbmTensor<bf16, Chip, m![Tok1024, H]>,
    weight: &HbmTensor<bf16, Chip, m![Nq, H]>,
    ws: &HbmTensor<bf16, Chip, m![Nq]>,
    raw: &mut HbmTensor<f32, Chip, m![Tok1024, Nq]>,
) {
    let a = half(device, hi, mid, lo, weight, 0);
    let b = half(device, hi, mid, lo, weight, 1);
    let sum = gemm16::add::<QkvCl, Sl, m![Tok1024 / 64], m![Tok1024 % 64], Nq>(device, &a, &b);
    let scale = gemm16::col_scales::<QkvCl, Sl, Nq, m![Nq]>(device, ws.view());
    let sum: Sums = device
        .main
        .begin(sum.view())
        .fetch::<m![Tok1024 / 64, Tok1024 % 64, Nq / 8 % 2], m![Nq % 8]>()
        .collect::<m![Tok1024 / 64, Tok1024 % 64, Nq / 8 % 2], m![Nq % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Tok1024 / 64, Tok1024 % 64, Nq / 8 % 2, Nq / 4 % 2], m![Nq % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_widen_concat::<m![Tok1024 / 64, Tok1024 % 64, Nq / 8 % 2], m![Nq % 8]>()
        .vector_final()
        .commit_trim::<m![Nq % 8]>()
        .commit();
    sum.view().to_hbm_view(&mut device.tdma, raw.view_mut());
}
