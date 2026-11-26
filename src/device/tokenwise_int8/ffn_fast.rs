//! The INT8 FFN stage restructured for the large buckets (Tok512, Tok1024): a drop-in for
//! `ffn::ffn_stage` (same arguments, same weight layouts, same output layout).
//!
//! Changes against `ffn::ffn_stage`, each aimed at the Main context (the stage's bottleneck) or at
//! time Main spends waiting:
//! 1. The rotations' Hadamard constants, the signed gamma and the row scales arrive as all-gathers
//!    (a sixteenth per slice from HBM, a sixteen-slice `Broadcast1`) instead of replicated DMAs.
//! 2. Up/Gate keeps four full-K passes; each pass pair's GeGLU output goes token-major through an
//!    8-ring transpose and one HBM store, and pair 0's store overlaps passes 2-3.
//! 3. The rotations use two-term splits, and the Down codes are stored per pass.
//!
//! Arithmetic: `ffn::ffn_stage` bit for bit. Every i32 sum is exact, so splitting the Down K across
//! the clusters changes nothing.
use furiosa_opt_std::prelude::*;

use super::ffn::{Lu, UCl};
use super::gemm::{self, Pi};
use super::passes::{OnePass, Passes, TwoPasses};
use super::post;
use super::post2;
use super::rotate::{self, H60Hbm, Kb, Kh, Ob, Ra, Rc, Sgi, Sl8, TCl, TSl};
use super::rotate2::{self, DmConsts};
use super::rotate_ff::{Kd, Kq4};
use crate::Chip;
use crate::axes::{Dummy2, H};

const INVSQRT2: f32 = 0.70710678118f32;

pub(super) type Row<D, T, const HALF: usize, E> = DmTensor<D, Chip, TCl<T, HALF>, TSl<T, HALF>, E>;
pub(super) type USl<URING, UXS, UYS> = m![{ URING }, { UXS }, { UYS }];
pub(super) type Onehot<T, const HALF: usize> = VrfTensor<f32, Chip, TCl<T, HALF>, m![1 # 256], m![Sgi, Sl8]>;

pub(crate) trait FastPasses: Passes {}
impl FastPasses for OnePass {}
impl FastPasses for TwoPasses {}

/// Down rows' slice axes.
pub(super) type DSl<DRING, DXS, DYS> = m![{ DRING }, { DXS }, { DYS }];

axes![X16s = 16];

/// The row scales `sx [T]` in every slice's VRF as an all-gather instead of one 2 KiB replicated DMA
/// (9 us less at Tok512 on the device): slice `16 g + j` reads piece `j =
/// [T / 64 % 2, T % 64 / 8]` (`T / 128` runs of eight scales), and a sixteen-slice `Broadcast1`
/// lays the pieces out in natural order. `T` a multiple of 128.
pub(crate) fn row_scales_ag<T: AxisName, CL: M, SL: M>(device: &mut Device, sx: &HbmTensor<f32, Chip, m![T]>) -> VrfTensor<f32, Chip, CL, SL, m![T]> {
    let p: DmTensor<f32, Chip, m![Dummy2], m![Rc, T / 64 % 2, T % 64 / 8], m![T / 128, T % 8]> = sx.to_dm(&mut device.tdma);
    let p: DmTensor<f32, Chip, CL, m![Rc, T / 64 % 2, T % 64 / 8], m![T / 128, T % 8]> = unsafe { p.reshape() };
    let g: DmTensor<f32, Chip, CL, m![Rc, X16s], m![T / 128, T / 64 % 2, T % 64 / 8, T % 8]> = device
        .main
        .begin(p.view())
        .fetch::<m![T / 128], m![T % 8]>()
        .switch::<m![Rc, X16s], m![T / 128, T / 64 % 2, T % 64 / 8]>(SwitchConfig::Broadcast1 { slice1: 16, slice0: 1 })
        .collect::<m![T / 128, T / 64 % 2, T % 64 / 8], m![T % 8]>()
        .commit_trim::<m![T % 8]>()
        .commit();
    let g: DmTensor<f32, Chip, CL, SL, m![T]> = unsafe { g.reshape() };
    device.sub.begin(g.view()).fetch::<m![T / 8], m![T % 8]>().collect::<m![T / 8], m![T % 8]>().to_vrf()
}

pub(crate) trait DownForm {}

/// GL: the Down input rotated and quantized in the GEMM layout (`ffn_core_gl`).
pub(crate) struct Gl;
/// GL with the quarter-owner Down tail (`QuarterDma`; Tok1024: `T = 2 PS`). Bits equal to `Gl`'s.
pub(crate) struct GlQ;
/// The GL Down form of a bucket: `GlQ` where the quarter-owner tail applies (`T = 2 PS`, Tok1024), else `Gl`.
pub(crate) trait GlPick {
    type F: DownForm;
}
impl GlPick for crate::axes::Tok1024 {
    type F = GlQ;
}
/// `GlPick`'s form for token axis `T`.
pub(crate) type GlOf<T> = <T as GlPick>::F;

impl DownForm for Gl {}
impl DownForm for GlQ {}

axes![Gs16 = 16, Gpp = 2];

struct AllReduce;

type GSl = m![Gs16, Lu / 1024 % 2, Lu / 256 % 2, Kb / 16];
/// The ring-transpose slices: ring pair, token `T % 16` (256-byte code runs: no HBM read-modify-write).
type GItSl<T> = m![Gs16, T % 16];
/// The Up rows' scales of all four passes with zeros in the padded ring pair: a pass zeroes every
/// slice's copy (`AND 0`, whatever the DM holds), then one DMA writes the 30 rings. Up/Gate then
/// gives `acc * sx * 0 = 0` there, so the padded slices add zeros to every quarter's amax (with the
/// DM's leftovers there instead, the scales and outputs changed).
fn gl_up_scales<URING: M, UXS: M, UYS: M>(device: &mut Device, wug_scale: &HbmTensor<f32, Chip, m![Lu]>) -> DmTensor<f32, Chip, UCl, USl<URING, UXS, UYS>, m![Lu / 16 % 4, Lu / 8 % 2, Lu % 8]> {
    let junk: DmTensor<f32, Chip, UCl, GSl, m![Lu / 16 % 4, Lu / 8 % 2, Lu % 8]> = DmTensor::new();
    let z: DmTensor<f32, Chip, UCl, GSl, m![Lu / 16 % 4, Lu / 8 % 2, Lu % 8]> = device
        .main
        .begin(junk.view())
        .fetch::<m![Lu / 16 % 4, Lu / 8 % 2], m![Lu % 8]>()
        .collect::<m![Lu / 16 % 4, Lu / 8 % 2], m![Lu % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0)
        .vector_reinterpret::<f32>()
        .vector_final()
        .commit_trim::<m![Lu % 8]>()
        .commit();
    let mut z: DmTensor<f32, Chip, UCl, USl<URING, UXS, UYS>, m![Lu / 16 % 4, Lu / 8 % 2, Lu % 8]> = unsafe { z.reshape() };
    wug_scale.view().to_dm_view(&mut device.tdma, z.view_mut());
    z
}

/// `geglu` into pass `p`'s tile of one act tensor for all four passes (GL reads it once all four
/// are written: the scheduler tracks hazards per tensor, so no GL table sits in the TRF beside an
/// Up/Gate weight pass).
#[allow(clippy::type_complexity)]
fn gl_geglu<TT: M, TR: M, URING: M, UXS: M, UYS: M>(
    device: &mut Device,
    y: &DmTensor<bf16, Chip, UCl, USl<URING, UXS, UYS>, m![{ TT }, { TR }, Lu / 8 % 2, Lu % 8]>,
    p: usize,
    act: &mut DmTensor<bf16, Chip, UCl, USl<URING, UXS, UYS>, m![Lu / 32 % 2, { TT }, { TR }, Lu / 16 % 2, Lu % 8]>,
) {
    let gate = y.view().tile::<m![Lu / 8 % 2], 1, m![{ TT }, { TR }, Lu / 8 % 2 = 1 # 2, Lu % 8]>(1);
    let up = y.view().tile::<m![Lu / 8 % 2], 1, m![{ TT }, { TR }, Lu / 8 % 2 = 1 # 2, Lu % 8]>(0);
    let gl: DmTensor<f32, Chip, UCl, USl<URING, UXS, UYS>, m![{ TT }, { TR }, Lu % 8]> = device
        .main
        .begin(gate)
        .fetch::<m![{ TT }, { TR }], m![Lu % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![{ TT }, { TR }], m![Lu % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{ TT }, { TR }, Lu / 4 % 2], m![Lu % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), INVSQRT2)
        .vector_fp_unary(FpUnaryOp::Erf)
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_widen_concat::<m![{ TT }, { TR }], m![Lu % 8]>()
        .vector_final()
        .commit_trim::<m![Lu % 8]>()
        .commit();
    let u: DmTensor<f32, Chip, UCl, USl<URING, UXS, UYS>, m![{ TT }, { TR }, Lu % 8]> = device
        .main
        .begin(up)
        .fetch::<m![{ TT }, { TR }], m![Lu % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![{ TT }, { TR }], m![Lu % 8]>()
        .commit_trim::<m![Lu % 8]>()
        .commit();
    device
        .main
        .begin_interleaved::<Pi, _, _, _, _, _>(u.view(), gl.view())
        .fetch::<m![{ TT }, { TR } / 16, Pi, { TR } % 16], m![Lu % 8]>()
        .collect::<m![{ TT }, { TR } / 16, Pi, { TR } % 16], m![Lu % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Pi, m![{ TT }, { TR }]>()
        .vector_narrow_split::<m![{ TT }, { TR }, Lu / 4 % 2], m![Lu % 4]>()
        .vector_fp_zip(FpBinaryOp::MulF(FpMulAlu::Mul0))
        .vector_fp_div(2f32)
        .vector_widen_concat::<m![{ TT }, { TR }], m![Lu % 8]>()
        .vector_final()
        .cast::<bf16, m![Lu % 8 # 16]>()
        .commit_trim::<m![Lu % 8]>()
        .commit_view(
            act.view_mut()
                .tile::<m![Lu / 32 % 2], 1, m![Lu / 32 % 2 = 1 #{!} 2, { TT }, { TR }, Lu / 16 % 2, Lu % 8]>(p / 2)
                .tile::<m![Lu / 16 % 2], 1, m![Lu / 32 % 2 = 1 #{!} 2, { TT }, { TR }, Lu / 16 % 2 = 1 #{!} 2, Lu % 8]>(p % 2),
        );
}

#[path = "ffn_fast_fp8c.rs"]
pub(crate) mod fp8c;

#[path = "rowdown.rs"]
mod rowdown;

#[path = "rowdown_f8.rs"]
mod rowdown_f8;

#[cfg(feature = "compile-carried")]
#[path = "rowdown_dm8_f8.rs"]
mod rowdown_dm8_f8;

#[path = "rowdown_slices256c2_f8.rs"]
mod rowdown_slices256c2_f8;
/// Odd-grid Q4 Gate/Up using two-flit packing.
pub(crate) mod ug4;

#[cfg(feature = "compile-carried")]
use super::passes::ThreePasses;

#[cfg(feature = "compile-carried")]
impl FastPasses for ThreePasses {}

#[cfg(feature = "compile-carried")]
impl GlPick for crate::axes::Tok1536 { type F = Gl; }
