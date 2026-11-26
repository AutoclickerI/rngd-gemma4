//! The Sliding head phase (q/k head norms with their gammas, the weightless v norm, RoPE) in
//! passes of `PS = 2 HALF = 512` rows with one row per slice and all 32 heads of the row in its
//! slice (the per-token phases' `TCl`/`TSl` layout).
//!
//! `device::tokenwise_int8::heads` puts one (row, KV head) per slice, `PS = 64` rows per pass, so
//! a Tok512 stage runs 8 passes of about 13 small DMA commands each and the pass is a serial chain
//! of their latencies (110k cycles of the 245k QKV stage). Here a pass is three loads (V, K, Q
//! parts of the projection rows, 4-8 KB per slice), one RoPE gather per table (one row per slice,
//! no replication copy) and three stores. Every per-element instruction keeps that module's
//! arithmetic: the same reduction (`Ds` in the same split, the outer head axes only added), the
//! same division and multiply units and roundings; the RoPE sum `q cos + rot(q) sin` adds the same
//! two f32 products in pair mode (the two halves of each head read interleaved, the tables as
//! half-row VRF operands), and an f32 add is commutative, so the output bits are unchanged.
use furiosa_opt_std::prelude::*;

use crate::axes::{Ds, E, Gs, Hh, Hqs, Ns, Ps, Qs, Rq, Rs};
use crate::device::tokenwise_int8::passes::{OnePass, TwoPasses};
use crate::device::tokenwise_int8::qkv::Nq;
use crate::device::tokenwise_int8::rotate::{TCl, TSl};
use crate::{Chip, EPS};

axes![Pz = 2];

const DS_F32: f32 = Ds::SIZE as f32;

type Row<D, T, const HALF: usize, EL> = DmTensor<D, Chip, TCl<T, HALF>, TSl<T, HALF>, EL>;
pub(crate) type Vrf<T, const HALF: usize, EL> = VrfTensor<f32, Chip, TCl<T, HALF>, TSl<T, HALF>, EL>;
/// The query heads of a row, `[Ns, Gs]`, as the outer axes of `[.., Ds]`.
type QO = m![Ns, Gs];

/// A `[Ds]` bf16 row as an f32 VRF operand.
fn vrf_ds<T: AxisName, const HALF: usize>(device: &mut Device, x: &Row<bf16, T, HALF, m![Ds]>) -> Vrf<T, HALF, m![Ds]> {
    device
        .sub
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf()
}

/// A `[Ds]` gamma in every slice's VRF.
pub(crate) fn table_vrf<T: AxisName, const HALF: usize>(device: &mut Device, w: &HbmTensor<bf16, Chip, m![Ds]>) -> Vrf<T, HALF, m![Ds]> {
    let w: Row<bf16, T, HALF, m![Ds]> = w.to_dm(&mut device.tdma);
    vrf_ds::<T, HALF>(device, &w)
}

/// `rms = sqrt(fold(x^2) / 256 + eps)` of every head (outer axes `O`), as `sliding::rmsnorm`.
fn rms<T: AxisName, const HALF: usize, O: M>(device: &mut Device, x: &Row<bf16, T, HALF, m![{ O }, Ds]>) -> Vrf<T, HALF, m![{ O }, 1 # 8]> {
    let mean_square: Row<f32, T, HALF, m![{ O }, 1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![{ O }, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![{ O }, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{ O }, Ds / 4], m![Ds % 4]>()
        .vector_stash()
        // Delay the stash read by two vector stages to keep the square pipelined.
        .vector_fp_binary(FpBinaryOp::AddF, -0f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<Ds, m![{ O }], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: Row<f32, T, HALF, m![{ O }, 1 # 8]> = device
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
    device.sub.begin(rms.view()).fetch::<m![{ O }], m![1 # 8]>().collect::<m![{ O }], m![1 # 8]>().to_vrf()
}

/// `bf16(x / rms * gamma)` per head (`normalize_query` / `normalize_key`).
fn normalize<T: AxisName, const HALF: usize, O: M>(
    device: &mut Device,
    x: &Row<bf16, T, HALF, m![{ O }, Ds]>,
    w: &Vrf<T, HALF, m![Ds]>,
) -> Row<bf16, T, HALF, m![{ O }, Ds]> {
    let r = rms::<T, HALF, O>(device, x);
    device
        .main
        .begin(x.view())
        .fetch::<m![{ O }, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![{ O }, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{ O }, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &r)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), w)
        .vector_widen_concat::<m![{ O }, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

/// `bf16(x / rms)` per head, no gamma (`normalize_value`).
fn normalize_value<T: AxisName, const HALF: usize, O: M>(device: &mut Device, x: &Row<bf16, T, HALF, m![{ O }, Ds]>) -> Row<bf16, T, HALF, m![{ O }, Ds]> {
    let r = rms::<T, HALF, O>(device, x);
    device
        .main
        .begin(x.view())
        .fetch::<m![{ O }, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![{ O }, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{ O }, Ds / 4], m![Ds % 4]>()
        .vector_fp_div(&r)
        .vector_widen_concat::<m![{ O }, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

/// Half `j` (`Ds / 128 = j`) of a `[Ds]` bf16 row as an f32 VRF operand.
fn vrf_half<T: AxisName, const HALF: usize>(device: &mut Device, x: &Row<bf16, T, HALF, m![Ds]>, j: usize) -> Vrf<T, HALF, m![Ds % 128]> {
    device
        .sub
        .begin(x.view().tile::<m![Ds / 128], 1, m![Ds / 128 = 1 # 2, Ds % 128]>(j))
        .fetch::<m![Ds % 128 / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds % 128 / 8], m![Ds % 8]>()
        .to_vrf()
}

/// The RoPE tables of a row as half operands: `(cos half 0, cos half 1, sin half 0, sin half 1)`.
pub(crate) type RopeHalves<T, const HALF: usize> =
    (Vrf<T, HALF, m![Ds % 128]>, Vrf<T, HALF, m![Ds % 128]>, Vrf<T, HALF, m![Ds % 128]>, Vrf<T, HALF, m![Ds % 128]>);

pub(crate) fn rope_halves<T: AxisName, const HALF: usize>(device: &mut Device, c: &Row<bf16, T, HALF, m![Ds]>, s: &Row<bf16, T, HALF, m![Ds]>) -> RopeHalves<T, HALF> {
    (vrf_half::<T, HALF>(device, c, 0), vrf_half::<T, HALF>(device, c, 1), vrf_half::<T, HALF>(device, s, 0), vrf_half::<T, HALF>(device, s, 1))
}

/// Output half `j` of `rope_pair`: `bf16(f32(x_j) cos_j + f32(x_{1-j}) sin_j)`, the two halves of
/// every head read as one interleaved pair.
fn rope_pair_half<T: AxisName, const HALF: usize, O: M>(
    device: &mut Device,
    x: &Row<bf16, T, HALF, m![{ O }, Ds]>,
    cos: &Vrf<T, HALF, m![Ds % 128]>,
    sin: &Vrf<T, HALF, m![Ds % 128]>,
    j: usize,
    out: &mut Row<bf16, T, HALF, m![{ O }, Ds]>,
) {
    device
        .main
        .begin_interleaved::<Pz, _, _, _, _, _>(
            x.view().tile::<m![Ds / 128], 1, m![{ O }, Ds / 128 = 1 # 2, Ds % 128]>(j),
            x.view().tile::<m![Ds / 128], 1, m![{ O }, Ds / 128 = 1 # 2, Ds % 128]>(1 - j),
        )
        .fetch::<m![{ O }, Ds % 128 / 8, Pz], m![Ds % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![{ O }, Ds % 128 / 8, Pz], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Pz, m![{ O }, Ds % 128 / 8]>()
        .vector_narrow_split::<m![{ O }, Ds % 128 / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), cos, ())
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), (), sin)
        .vector_widen_concat::<m![{ O }, Ds % 128 / 8], m![Ds % 8]>()
        .vector_clip_zip(ClipBinaryOpF32::Add)
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit_view(out.view_mut().tile::<m![Ds / 128], 1, m![{ O }, Ds / 128 = 1 #{!} 2, Ds % 128]>(j));
}

/// `sliding::rope::apply_rope` on every head, `bf16(x cos + rotate_half(x) sin)`, in two pair-mode
/// passes: output half `j` reads halves `j` and `1 - j` of `x` interleaved and multiplies them by
/// `cos_j` and `sin_j` (the tables carry `rotate_half`'s sign), then adds the two f32 products
/// (`ClipBinaryOpF32::Add`) and rounds once, without materializing the product rows.
pub(crate) fn rope_pair<T: AxisName, const HALF: usize, O: M>(device: &mut Device, x: &Row<bf16, T, HALF, m![{ O }, Ds]>, t: &RopeHalves<T, HALF>) -> Row<bf16, T, HALF, m![{ O }, Ds]> {
    let mut out: Row<bf16, T, HALF, m![{ O }, Ds]> = DmTensor::new();
    rope_pair_half::<T, HALF, O>(device, x, &t.0, &t.2, 0, &mut out);
    rope_pair_half::<T, HALF, O>(device, x, &t.1, &t.3, 1, &mut out);
    out
}

/// The loads and arithmetic of one pass (`PI` of `P`) of the head phase on the projection rows
/// `raw = [T, Q (4,096) | K (2,048) | V (2,048)]` (bf16, scaled), `PS = 2 HALF` rows: the pass's
/// q, k, v rows, one per slice. No loop: a rolled loop disables the compiler's register-file
/// reuse for the whole kernel and repeats one schedule per iteration.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn heads_compute<T: AxisName, const HALF: usize, const PS: usize, const P: usize, const PI: usize>(
    device: &mut Device,
    raw: &HbmTensor<bf16, Chip, m![T, Nq]>,
    q_gamma: &Vrf<T, HALF, m![Ds]>,
    k_gamma: &Vrf<T, HALF, m![Ds]>,
    rope_offsets: &HbmTensor<i32, Chip, m![T]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
) -> (Row<bf16, T, HALF, m![Qs]>, Row<bf16, T, HALF, m![Ps]>, Row<bf16, T, HALF, m![Ps]>) {
    // Parts are tiled on the base axis `Nq` and loaded in `Nq` digits, then the DM tensor is
    // relabeled to the head axes.
    let v: Row<bf16, T, HALF, m![Nq / 256 % 8, Nq % 256]> = raw
        .view()
        .tile::<m![T / { PS }], 1, m![T / { PS } = 1 # { P }, T % { PS }, Nq]>(PI)
        .tile::<m![Nq / 2048], 1, m![T / { PS } = 1 # { P }, T % { PS }, Nq / 2048 = 1 # 4, Nq % 2048]>(3)
        .to_dm(&mut device.tdma);
    let v: Row<bf16, T, HALF, m![Ns, Ds]> = unsafe { v.reshape() };
    let k: Row<bf16, T, HALF, m![Nq / 256 % 8, Nq % 256]> = raw
        .view()
        .tile::<m![T / { PS }], 1, m![T / { PS } = 1 # { P }, T % { PS }, Nq]>(PI)
        .tile::<m![Nq / 2048], 1, m![T / { PS } = 1 # { P }, T % { PS }, Nq / 2048 = 1 # 4, Nq % 2048]>(2)
        .to_dm(&mut device.tdma);
    let k: Row<bf16, T, HALF, m![Ns, Ds]> = unsafe { k.reshape() };
    let mut offsets = HbmTensor::<i32, Chip, m![T % { PS }]>::new();
    rope_offsets
        .view()
        .tile::<m![T / { PS }], 1, m![T / { PS } = 1 # { P }, T % { PS }]>(PI)
        .to_hbm_view(&mut device.tdma, offsets.view_mut());
    let c: Row<bf16, T, HALF, m![Ds]> = cos.gather::<m![E], m![Ds]>().by_byte_offsets(&offsets).to_dm(&mut device.tdma);
    let s: Row<bf16, T, HALF, m![Ds]> = sin.gather::<m![E], m![Ds]>().by_byte_offsets(&offsets).to_dm(&mut device.tdma);
    let q: Row<bf16, T, HALF, m![Nq / 512 % 8, Nq / 256 % 2, Nq % 256]> = raw
        .view()
        .tile::<m![T / { PS }], 1, m![T / { PS } = 1 # { P }, T % { PS }, Nq]>(PI)
        .tile::<m![Nq / 4096], 1, m![T / { PS } = 1 # { P }, T % { PS }, Nq / 4096 = 1 # 2, Nq % 4096]>(0)
        .to_dm(&mut device.tdma);
    let q: Row<bf16, T, HALF, m![{ QO }, Ds]> = unsafe { q.reshape() };

    let v = normalize_value::<T, HALF, m![Ns]>(device, &v);
    let t = rope_halves::<T, HALF>(device, &c, &s);
    let k = normalize::<T, HALF, m![Ns]>(device, &k, k_gamma);
    let k = rope_pair::<T, HALF, m![Ns]>(device, &k, &t);
    let q = normalize::<T, HALF, QO>(device, &q, q_gamma);
    let q = rope_pair::<T, HALF, QO>(device, &q, &t);
    (unsafe { q.reshape() }, unsafe { k.reshape() }, unsafe { v.reshape() })
}

/// One pass of the head phase with the rows stored to the dense outputs (the stage contract).
#[allow(clippy::too_many_arguments)]
pub(crate) fn heads_pass<T: AxisName, const HALF: usize, const PS: usize, const P: usize, const PI: usize>(
    device: &mut Device,
    raw: &HbmTensor<bf16, Chip, m![T, Nq]>,
    q_gamma: &Vrf<T, HALF, m![Ds]>,
    k_gamma: &Vrf<T, HALF, m![Ds]>,
    rope_offsets: &HbmTensor<i32, Chip, m![T]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
    q_out: &mut HbmTensor<bf16, Chip, m![T, Qs]>,
    k_out: &mut HbmTensor<bf16, Chip, m![T, Ps]>,
    v_out: &mut HbmTensor<bf16, Chip, m![T, Ps]>,
) {
    let (q, k, v) = heads_compute::<T, HALF, PS, P, PI>(device, raw, q_gamma, k_gamma, rope_offsets, cos, sin);
    v.view().to_hbm_view(
        &mut device.tdma,
        v_out.view_mut().tile::<m![T / { PS }], 1, m![T / { PS } = 1 #{!} { P }, T % { PS }, Ps]>(PI),
    );
    k.view().to_hbm_view(
        &mut device.tdma,
        k_out.view_mut().tile::<m![T / { PS }], 1, m![T / { PS } = 1 #{!} { P }, T % { PS }, Ps]>(PI),
    );
    q.view().to_hbm_view(
        &mut device.tdma,
        q_out.view_mut().tile::<m![T / { PS }], 1, m![T / { PS } = 1 #{!} { P }, T % { PS }, Qs]>(PI),
    );
}

/// Run head normalization and RoPE for each token pass.
#[allow(clippy::too_many_arguments)]
pub(crate) trait HeadsPasses {
    fn heads<T: AxisName, const HALF: usize, const PS: usize>(
        device: &mut Device,
        raw: &HbmTensor<bf16, Chip, m![T, Nq]>,
        q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
        k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
        rope_offsets: &HbmTensor<i32, Chip, m![T]>,
        cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
        sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
        q_out: &mut HbmTensor<bf16, Chip, m![T, Qs]>,
        k_out: &mut HbmTensor<bf16, Chip, m![T, Ps]>,
        v_out: &mut HbmTensor<bf16, Chip, m![T, Ps]>,
    );
}

macro_rules! heads_passes_impl {
    ($ty:ty, $p:literal, [$($i:literal),*]) => {
        impl HeadsPasses for $ty {
            fn heads<T: AxisName, const HALF: usize, const PS: usize>(
                device: &mut Device,
                raw: &HbmTensor<bf16, Chip, m![T, Nq]>,
                q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
                k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
                rope_offsets: &HbmTensor<i32, Chip, m![T]>,
                cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
                sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
                q_out: &mut HbmTensor<bf16, Chip, m![T, Qs]>,
                k_out: &mut HbmTensor<bf16, Chip, m![T, Ps]>,
                v_out: &mut HbmTensor<bf16, Chip, m![T, Ps]>,
            ) {
                let k_gamma = table_vrf::<T, HALF>(device, k_rms_weight);
                let q_gamma = table_vrf::<T, HALF>(device, q_rms_weight);
                $(heads_pass::<T, HALF, PS, $p, $i>(device, raw, &q_gamma, &k_gamma, rope_offsets, cos, sin, q_out, k_out, v_out);)*
            }
        }
    };
}

heads_passes_impl!(OnePass, 1, [0]);
heads_passes_impl!(TwoPasses, 2, [0, 1]);

/// Encode FP8 KV as `e4m3(clamp(f32(x),
/// -448, 448))` per element of every head, round-to-nearest-even in the Cast Engine, no scale.
fn e4m3_heads<T: AxisName, const HALF: usize, O: M>(device: &mut Device, x: &Row<bf16, T, HALF, m![{ O }, Ds]>) -> Row<f8e4m3, T, HALF, m![{ O }, Ds]> {
    device
        .main
        .begin(x.view())
        .fetch::<m![{ O }, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![{ O }, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Min, 448f32)
        .vector_clip(ClipBinaryOpF32::Max, -448f32)
        .vector_final()
        .cast::<f8e4m3, m![Ds % 8 # 32]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

/// `heads_pass_rows_early` into FP8 (E4M3) Sliding pools: the normalized v and the roped k of
/// each row narrowed by `e4m3_heads` before their
/// scatters; `stage` holds the E4M3 pools' byte offsets (half the bf16 ones). q stays bf16.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "compile-carried")]
pub(crate) fn heads_pass_rows_early8<T: AxisName, const HALF: usize, const PS: usize, const P: usize, const PI: usize>(
    device: &mut Device,
    raw: &HbmTensor<bf16, Chip, m![T, Nq]>,
    q_gamma: &Vrf<T, HALF, m![Ds]>,
    k_gamma: &Vrf<T, HALF, m![Ds]>,
    rope_offsets: &HbmTensor<i32, Chip, m![T]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
    rows: &HbmTensor<i32, Chip, m![T, Hh]>,
    stage: &HbmTensor<i32, Chip, m![T]>,
    qin: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
    k_pool: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
    v_pool: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
) {
    let mut prow = HbmTensor::<i32, Chip, m![T % { PS }, Hh]>::new();
    rows.view()
        .tile::<m![T / { PS }], 1, m![T / { PS } = 1 # { P }, T % { PS }, Hh]>(PI)
        .to_hbm_view(&mut device.tdma, prow.view_mut());
    let mut pstage = HbmTensor::<i32, Chip, m![T % { PS }]>::new();
    stage
        .view()
        .tile::<m![T / { PS }], 1, m![T / { PS } = 1 # { P }, T % { PS }]>(PI)
        .to_hbm_view(&mut device.tdma, pstage.view_mut());
    let v: Row<bf16, T, HALF, m![Nq / 256 % 8, Nq % 256]> = raw
        .view()
        .tile::<m![T / { PS }], 1, m![T / { PS } = 1 # { P }, T % { PS }, Nq]>(PI)
        .tile::<m![Nq / 2048], 1, m![T / { PS } = 1 # { P }, T % { PS }, Nq / 2048 = 1 # 4, Nq % 2048]>(3)
        .to_dm(&mut device.tdma);
    let v: Row<bf16, T, HALF, m![Ns, Ds]> = unsafe { v.reshape() };
    let k: Row<bf16, T, HALF, m![Nq / 256 % 8, Nq % 256]> = raw
        .view()
        .tile::<m![T / { PS }], 1, m![T / { PS } = 1 # { P }, T % { PS }, Nq]>(PI)
        .tile::<m![Nq / 2048], 1, m![T / { PS } = 1 # { P }, T % { PS }, Nq / 2048 = 1 # 4, Nq % 2048]>(2)
        .to_dm(&mut device.tdma);
    let k: Row<bf16, T, HALF, m![Ns, Ds]> = unsafe { k.reshape() };
    let mut offsets = HbmTensor::<i32, Chip, m![T % { PS }]>::new();
    rope_offsets
        .view()
        .tile::<m![T / { PS }], 1, m![T / { PS } = 1 # { P }, T % { PS }]>(PI)
        .to_hbm_view(&mut device.tdma, offsets.view_mut());
    let c: Row<bf16, T, HALF, m![Ds]> = cos.gather::<m![E], m![Ds]>().by_byte_offsets(&offsets).to_dm(&mut device.tdma);
    let s: Row<bf16, T, HALF, m![Ds]> = sin.gather::<m![E], m![Ds]>().by_byte_offsets(&offsets).to_dm(&mut device.tdma);
    let q: Row<bf16, T, HALF, m![Nq / 512 % 8, Nq / 256 % 2, Nq % 256]> = raw
        .view()
        .tile::<m![T / { PS }], 1, m![T / { PS } = 1 # { P }, T % { PS }, Nq]>(PI)
        .tile::<m![Nq / 4096], 1, m![T / { PS } = 1 # { P }, T % { PS }, Nq / 4096 = 1 # 2, Nq % 4096]>(0)
        .to_dm(&mut device.tdma);
    let q: Row<bf16, T, HALF, m![{ QO }, Ds]> = unsafe { q.reshape() };

    let v = normalize_value::<T, HALF, m![Ns]>(device, &v);
    let v = e4m3_heads::<T, HALF, m![Ns]>(device, &v);
    v_pool.view_mut().scatter::<m![Rs], m![Ns, Ds]>().by_byte_offsets(pstage.view()).from_dm_view(&mut device.tdma, v.view());
    let t = rope_halves::<T, HALF>(device, &c, &s);
    let k = normalize::<T, HALF, m![Ns]>(device, &k, k_gamma);
    let k = rope_pair::<T, HALF, m![Ns]>(device, &k, &t);
    let k = e4m3_heads::<T, HALF, m![Ns]>(device, &k);
    k_pool.view_mut().scatter::<m![Rs], m![Ns, Ds]>().by_byte_offsets(pstage.view()).from_dm_view(&mut device.tdma, k.view());
    let q = normalize::<T, HALF, QO>(device, &q, q_gamma);
    let q = rope_pair::<T, HALF, QO>(device, &q, &t);
    let q: Row<bf16, T, HALF, m![Hh, Hqs]> = unsafe { q.reshape() };
    qin.view_mut().scatter::<m![Rq], m![Hqs]>().by_byte_offsets(prow.view()).from_dm_view(&mut device.tdma, q.view());
}

/// The head phase with early scatters into FP8 (E4M3) Sliding pools, over all passes.
#[allow(clippy::too_many_arguments)]
pub(crate) trait HeadsRowsEarlyPasses8 {
    #[cfg(feature = "compile-carried")]
    fn heads_rows_early8<T: AxisName, const HALF: usize, const PS: usize>(
        device: &mut Device,
        raw: &HbmTensor<bf16, Chip, m![T, Nq]>,
        q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
        k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
        rope_offsets: &HbmTensor<i32, Chip, m![T]>,
        cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
        sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
        rows: &HbmTensor<i32, Chip, m![T, Hh]>,
        stage: &HbmTensor<i32, Chip, m![T]>,
        qin: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
        k_pool: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
        v_pool: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
    );
}

macro_rules! heads_rows_early8_impl {
    ($ty:ty, $p:literal, [$($i:literal),*]) => {
        impl HeadsRowsEarlyPasses8 for $ty {
            #[cfg(feature = "compile-carried")]
            fn heads_rows_early8<T: AxisName, const HALF: usize, const PS: usize>(
                device: &mut Device,
                raw: &HbmTensor<bf16, Chip, m![T, Nq]>,
                q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
                k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
                rope_offsets: &HbmTensor<i32, Chip, m![T]>,
                cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
                sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
                rows: &HbmTensor<i32, Chip, m![T, Hh]>,
                stage: &HbmTensor<i32, Chip, m![T]>,
                qin: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
                k_pool: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
                v_pool: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
            ) {
                let k_gamma = table_vrf::<T, HALF>(device, k_rms_weight);
                let q_gamma = table_vrf::<T, HALF>(device, q_rms_weight);
                $(heads_pass_rows_early8::<T, HALF, PS, $p, $i>(device, raw, &q_gamma, &k_gamma, rope_offsets, cos, sin, rows, stage, qin, k_pool, v_pool);)*
            }
        }
    };
}

heads_rows_early8_impl!(OnePass, 1, [0]);
heads_rows_early8_impl!(TwoPasses, 2, [0, 1]);

/// Pass-local raw storage; identical current head arithmetic and scatter tables.
pub(crate) fn heads_pass_rows_early8_raw<T: AxisName, const HALF: usize, const PS: usize, const P: usize, const PI: usize>(
    device: &mut Device,
    raw: &HbmTensor<bf16, Chip, m![T % { PS }, Nq]>,
    q_gamma: &Vrf<T, HALF, m![Ds]>,
    k_gamma: &Vrf<T, HALF, m![Ds]>,
    rope_offsets: &HbmTensor<i32, Chip, m![T]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
    rows: &HbmTensor<i32, Chip, m![T, Hh]>,
    stage: &HbmTensor<i32, Chip, m![T]>,
    qin: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
    k_pool: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
    v_pool: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
) {
    let mut prow = HbmTensor::<i32, Chip, m![T % { PS }, Hh]>::new();
    rows.view()
        .tile::<m![T / { PS }], 1, m![T / { PS } = 1 # { P }, T % { PS }, Hh]>(PI)
        .to_hbm_view(&mut device.tdma, prow.view_mut());
    let mut pstage = HbmTensor::<i32, Chip, m![T % { PS }]>::new();
    stage
        .view()
        .tile::<m![T / { PS }], 1, m![T / { PS } = 1 # { P }, T % { PS }]>(PI)
        .to_hbm_view(&mut device.tdma, pstage.view_mut());
    let v: Row<bf16, T, HALF, m![Nq / 256 % 8, Nq % 256]> = raw
        .view()
        .tile::<m![Nq / 2048], 1, m![T % { PS }, Nq / 2048 = 1 # 4, Nq % 2048]>(3)
        .to_dm(&mut device.tdma);
    let v: Row<bf16, T, HALF, m![Ns, Ds]> = unsafe { v.reshape() };
    let k: Row<bf16, T, HALF, m![Nq / 256 % 8, Nq % 256]> = raw
        .view()
        .tile::<m![Nq / 2048], 1, m![T % { PS }, Nq / 2048 = 1 # 4, Nq % 2048]>(2)
        .to_dm(&mut device.tdma);
    let k: Row<bf16, T, HALF, m![Ns, Ds]> = unsafe { k.reshape() };
    let mut offsets = HbmTensor::<i32, Chip, m![T % { PS }]>::new();
    rope_offsets
        .view()
        .tile::<m![T / { PS }], 1, m![T / { PS } = 1 # { P }, T % { PS }]>(PI)
        .to_hbm_view(&mut device.tdma, offsets.view_mut());
    let c: Row<bf16, T, HALF, m![Ds]> = cos.gather::<m![E], m![Ds]>().by_byte_offsets(&offsets).to_dm(&mut device.tdma);
    let s: Row<bf16, T, HALF, m![Ds]> = sin.gather::<m![E], m![Ds]>().by_byte_offsets(&offsets).to_dm(&mut device.tdma);
    let q: Row<bf16, T, HALF, m![Nq / 512 % 8, Nq / 256 % 2, Nq % 256]> = raw
        .view()
        .tile::<m![Nq / 4096], 1, m![T % { PS }, Nq / 4096 = 1 # 2, Nq % 4096]>(0)
        .to_dm(&mut device.tdma);
    let q: Row<bf16, T, HALF, m![{ QO }, Ds]> = unsafe { q.reshape() };

    let v = normalize_value::<T, HALF, m![Ns]>(device, &v);
    let v = e4m3_heads::<T, HALF, m![Ns]>(device, &v);
    v_pool.view_mut().scatter::<m![Rs], m![Ns, Ds]>().by_byte_offsets(pstage.view()).from_dm_view(&mut device.tdma, v.view());
    let t = rope_halves::<T, HALF>(device, &c, &s);
    let k = normalize::<T, HALF, m![Ns]>(device, &k, k_gamma);
    let k = rope_pair::<T, HALF, m![Ns]>(device, &k, &t);
    let k = e4m3_heads::<T, HALF, m![Ns]>(device, &k);
    k_pool.view_mut().scatter::<m![Rs], m![Ns, Ds]>().by_byte_offsets(pstage.view()).from_dm_view(&mut device.tdma, k.view());
    let q = normalize::<T, HALF, QO>(device, &q, q_gamma);
    let q = rope_pair::<T, HALF, QO>(device, &q, &t);
    let q: Row<bf16, T, HALF, m![Hh, Hqs]> = unsafe { q.reshape() };
    qin.view_mut().scatter::<m![Rq], m![Hqs]>().by_byte_offsets(prow.view()).from_dm_view(&mut device.tdma, q.view());
}

#[cfg(feature = "compile-carried")]
use crate::device::tokenwise_int8::passes::ThreePasses;
#[cfg(feature = "compile-carried")]
impl HeadsRowsEarlyPasses8 for ThreePasses {
    #[cfg(feature = "compile-carried")]
    fn heads_rows_early8<T: AxisName, const HALF: usize, const PS: usize>(
        device: &mut Device,
        raw: &HbmTensor<bf16, Chip, m![T, Nq]>,
        q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
        k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
        rope_offsets: &HbmTensor<i32, Chip, m![T]>,
        cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
        sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
        rows: &HbmTensor<i32, Chip, m![T, Hh]>,
        stage: &HbmTensor<i32, Chip, m![T]>,
        qin: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
        k_pool: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
        v_pool: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
    ) {
        let k_gamma = table_vrf::<T, HALF>(device, k_rms_weight);
        let q_gamma = table_vrf::<T, HALF>(device, q_rms_weight);
        heads_pass_rows_early8::<T, HALF, PS, 3, 0>(device, raw, &q_gamma, &k_gamma, rope_offsets, cos, sin, rows, stage, qin, k_pool, v_pool);
        let k_gamma = table_vrf::<T, HALF>(device, k_rms_weight);
        let q_gamma = table_vrf::<T, HALF>(device, q_rms_weight);
        heads_pass_rows_early8::<T, HALF, PS, 3, 1>(device, raw, &q_gamma, &k_gamma, rope_offsets, cos, sin, rows, stage, qin, k_pool, v_pool);
        let k_gamma = table_vrf::<T, HALF>(device, k_rms_weight);
        let q_gamma = table_vrf::<T, HALF>(device, q_rms_weight);
        heads_pass_rows_early8::<T, HALF, PS, 3, 2>(device, raw, &q_gamma, &k_gamma, rope_offsets, cos, sin, rows, stage, qin, k_pool, v_pool);
    }
}
#[cfg(feature = "compile-carried")]
heads_passes_impl!(ThreePasses, 3, [0, 1, 2]);
