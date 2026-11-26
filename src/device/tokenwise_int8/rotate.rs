//! Online rotation and per-row INT8 quantization of a 3,840-wide linear input, one token per
//! slice (`T <= 512`).
//!
//! The rotation is `R = (H60 (x) H64) D / sqrt(3840)` with `k = 64 a + b` (`a` the H60 index, `b`
//! the H64 index), `D` the seeded signs; `D` (and any power-of-two smoothing factor) is folded
//! into the producer, so the input `x` here is already `D x` (exact). Evaluation:
//!
//! 1. stage A, H64 over `b` on the exact bf16 input (packet-reduced contraction, f32 sums):
//!    `u[a, b'] = sum_b H64[b', b] x[a, b]`;
//! 2. `u` split into two bf16 terms: `t0 = trunc16(u)` (bit mask) and `t1 = bf16(u - t0)` (the
//!    difference is exact, its narrowing rounds to nearest even), so `t0 + t1` carries 16 of
//!    `u`'s 24 significant bits. The measured error against unquantized output rose from
//!    0.853% to 0.854% when using two terms instead of three;
//! 3. stage B, H60 over `a` on the two terms (time-reduced contraction, the packet keeps 32
//!    consecutive `b'`): `v[a', b'] = sum_j sum_a H60[a', a] t_j[a, b']` in f32;
//! 4. `y = bf16(v * f32(1 / sqrt(3840)))`;
//! 5. `s = max(amax_k |y_k| / 127, 1e-30)`, `q_k = clamp(rne(y_k / s), -127, 127)` with
//!    `rne(z) = i32(z + 1.5 * 2^23) - 1.5 * 2^23` (the f32 add rounds to nearest even).
//!
//! Every value is the reference's (`precision_policy.Rotation.apply`, then the bf16 narrowing and
//! `quantize_int_groups`) up to the f32 summation order of stages 1 and 3 and the place of the
//! `1 / sqrt(n)` multiply. The output order of `k` is `[b' / 32, a', b' % 32]` (a fixed
//! permutation; the prepared weights use the same one, so the INT8 dot is unchanged).
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Dummy2, H};

axes![Ka = 60, Kb = 64, Ob = 64, Oa = 64, Ra = 60, Kh = 3840, Dum16 = 16];

// The two terms of a stage output (`trunc16(u)`, `bf16(u - trunc16(u))`), for every rotation.
axes![J2 = 2];

/// The constant copies (`crate::axes::Rc`, shared with the engine).
pub(crate) use crate::axes::Rc;

/// One token per slice in passes of `2 HALF` rows: row `t` of a pass on cluster `t / HALF % 2`,
/// slice `t % HALF`, so each cluster's rows are contiguous in HBM.
pub(crate) type TCl<T, const HALF: usize> = m![T / { HALF } % 2];
pub(crate) type TSl<T, const HALF: usize> = m![T % { HALF } # 256];
type Row<D, T, const HALF: usize, E> = DmTensor<D, Chip, TCl<T, HALF>, TSl<T, HALF>, E>;

/// The rotated row's element order, `[b' / 32, a', b' % 32]`, with `a'` padded to 64.
pub(crate) type RotPad = m![Ob / 32, Ra # 64, Ob % 32];
/// The same order without the padding (the HBM code layout).
pub(crate) type RotK = m![Ob / 32, Ra, Ob % 32];
/// The HBM order of the H60 constant (`[Oa / 32, Oa % 8, Ka, Oa / 8 % 4]`, the TRF order).
pub(crate) type H60Hbm = m![Oa / 32, Oa % 8, Ka, Oa / 8 % 4];

/// `f32(1 / sqrt(3840))`, the reference's normalization (`Rotation.scale`, f64, cast to f32).
pub(crate) const INV_SQRT_3840: f32 = 0.016137430609197570_f64 as f32;

 // 1.5 * 2^23: (z + MAGIC) - MAGIC rounds |z| < 2^22 to nearest even
const KH_F32: f32 = 3840.0;
const MASK_HI: i32 = -65_536; // 0xffff_0000: the bf16 part of an f32
const SIGN: i32 = i32::MIN; // 0x8000_0000

/// H64 in the TRF: lane `Ob % 8`, rows `Ob / 8` broadcast in time.
pub(crate) type H64Trf<T, const HALF: usize> = TrfTensor<bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![Ob % 8], m![Ob / 8, Kb]>;
/// One half (`Oa / 32`) of H60 in the TRF: lane `Oa % 8`, one entry read per cycle.
pub(crate) type H60Trf<T, const HALF: usize> = TrfTensor<bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![Oa % 8], m![Ka, Oa / 8 % 4]>;

/// The Hadamard factors in every slice's TRF: `h64 [Ob, Kb]` (Sylvester) and
/// `h60` (Paley I, rows 60..63 zero) stored in the TRF order `[Oa / 32, Oa % 8, Ka, Oa / 8 % 4]`
/// (the host arranges it).
///
/// Constants arrive as `Rc = 16` identical copies in HBM; slice `s` reads copy `s % 16`, so the
/// replication to all slices does not hammer one HBM region (one copy read by every slice took
/// about 20k cycles for 8 KiB in the static schedule).
pub(crate) fn consts<T: AxisName, const HALF: usize, PP: super::passes::Passes>(
    device: &mut Device,
    h64: &HbmTensor<bf16, Chip, m![Rc, Ob, Kb]>,
    h60: &HbmTensor<bf16, Chip, m![Rc, { H60Hbm }]>,
) -> (H64Trf<T, HALF>, H60Trf<T, HALF>, H60Trf<T, HALF>) {
    let d64: Row<bf16, T, HALF, m![Ob, Kb]> = PP::table::<T, HALF, m![Ob, Kb]>(device, h64.view());
    let t64 = device
        .sub
        .begin(d64.view())
        .fetch::<m![Ob % 8, Ob / 8, Kb / 16], m![Kb % 16]>()
        .collect::<m![Ob % 8, Ob / 8, Kb / 16], m![Kb % 16]>()
        .to_trf();
    let d60: Row<bf16, T, HALF, H60Hbm> = PP::table::<T, HALF, H60Hbm>(device, h60.view());
    let h0 = h60_half::<T, HALF>(device, &d60, 0);
    let h1 = h60_half::<T, HALF>(device, &d60, 1);
    (t64, h0, h1)
}

fn h60_half<T: AxisName, const HALF: usize>(device: &mut Device, d60: &Row<bf16, T, HALF, H60Hbm>, i: usize) -> H60Trf<T, HALF> {
    device
        .sub
        .begin(d60.view().tile::<m![Oa / 32], 1, m![Oa / 32 = 1 # 2, Oa % 8, Ka, Oa / 8 % 4]>(i))
        .fetch::<m![Oa % 8, Ka / 4], m![Ka % 4, Oa / 8 % 4]>()
        .collect::<m![Oa % 8, Ka / 4], m![Ka % 4, Oa / 8 % 4]>()
        .to_trf()
}

/// Stage A with the `t0 = trunc16(u)` epilogue.
fn stage_a_t0<T: AxisName, const HALF: usize>(
    device: &mut Device,
    x: &Row<bf16, T, HALF, m![Ka, Kb]>,
    c: &(H64Trf<T, HALF>, H60Trf<T, HALF>, H60Trf<T, HALF>),
    terms: &mut Row<bf16, T, HALF, m![J2, Ka, Ob]>,
) {
    device
        .main
        .begin(x.view())
        .fetch::<m![Ka, Kb / 16], m![Kb % 16]>()
        .collect::<m![Ka, Kb / 16], m![Kb % 16]>()
        .contract_outer::<m![Ka, Kb / 32, Ob / 8], m![Kb % 32], _, _, _>(&c.0)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ka, Ob / 8]>()
        .contract_lane::<m![Ka, Ob / 8], m![Ob % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, MASK_HI)
        .vector_reinterpret::<f32>()
        .vector_final()
        .cast::<bf16, m![Ob % 8 # 16]>()
        .commit_trim::<m![Ob % 8]>()
        .commit_view(terms.view_mut().tile::<m![J2], 1, m![J2 = 1 #{!} 2, Ka, Ob]>(0));
}

/// Stage A with the `t1 = bf16(u - trunc16(u))` epilogue (the difference is exact; the narrowing
/// rounds to nearest even).
fn stage_a_t1<T: AxisName, const HALF: usize>(
    device: &mut Device,
    x: &Row<bf16, T, HALF, m![Ka, Kb]>,
    c: &(H64Trf<T, HALF>, H60Trf<T, HALF>, H60Trf<T, HALF>),
    terms: &mut Row<bf16, T, HALF, m![J2, Ka, Ob]>,
) {
    device
        .main
        .begin(x.view())
        .fetch::<m![Ka, Kb / 16], m![Kb % 16]>()
        .collect::<m![Ka, Kb / 16], m![Kb % 16]>()
        .contract_outer::<m![Ka, Kb / 32, Ob / 8], m![Kb % 32], _, _, _>(&c.0)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ka, Ob / 8]>()
        .contract_lane::<m![Ka, Ob / 8], m![Ob % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_stash()
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, MASK_HI)
        .vector_logic(LogicBinaryOpI32::BitXor, SIGN)
        .vector_reinterpret::<f32>()
        .vector_clip(ClipBinaryOpF32::Add, Stash)
        .vector_final()
        .cast::<bf16, m![Ob % 8 # 16]>()
        .commit_trim::<m![Ob % 8]>()
        .commit_view(terms.view_mut().tile::<m![J2], 1, m![J2 = 1 #{!} 2, Ka, Ob]>(1));
}

/// Stage B for output rows `a'` in half `i` (`Oa / 32 = i`), scaled and narrowed into `y`.
fn stage_b<T: AxisName, const HALF: usize>(
    device: &mut Device,
    terms: &Row<bf16, T, HALF, m![J2, Ka, Ob]>,
    h60: &H60Trf<T, HALF>,
    scale: f32,
    i: usize,
    y: &mut Row<bf16, T, HALF, m![Ob / 32, Oa, Ob % 32]>,
) {
    device
        .main
        .begin(terms.view())
        .fetch::<m![Ob / 32, J2, Ka, Ob % 32 / 16], m![Ob % 16]>()
        .collect::<m![Ob / 32, J2, Ka, Ob % 32 / 16], m![Ob % 16]>()
        .contract_outer::<m![Ob / 32, J2, Ka, Oa / 8 % 4], m![Ob % 32], _, _, _>(h60)
        .contract_packet::<m![Ob % 32]>()
        .contract_time::<m![Ob / 32, Oa / 8 % 4]>()
        .contract_lane::<m![Ob / 32, Oa / 8 % 4, Oa % 8, Ob % 32 / 8], m![Ob % 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ob / 32, Oa / 8 % 4, Oa % 8, Ob % 32 / 4], m![Ob % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scale)
        .vector_widen_concat::<m![Ob / 32, Oa / 8 % 4, Oa % 8, Ob % 32 / 8], m![Ob % 8]>()
        .vector_final()
        .cast::<bf16, m![Ob % 8 # 16]>()
        .commit_trim::<m![Ob % 8]>()
        .commit_view(y.view_mut().tile::<m![Oa / 32], 1, m![Ob / 32, Oa / 32 = 1 #{!} 2, Oa % 32, Ob % 32]>(i));
}

/// The rotated row `y` (bf16, order `[b' / 32, a' (64, rows 60..63 zero), b' % 32]`).
pub(crate) fn rotate<T: AxisName, const HALF: usize>(
    device: &mut Device,
    x: &Row<bf16, T, HALF, m![Ka, Kb]>,
    c: &(H64Trf<T, HALF>, H60Trf<T, HALF>, H60Trf<T, HALF>),
    scale: f32,
) -> Row<bf16, T, HALF, RotPad> {
    let mut terms: Row<bf16, T, HALF, m![J2, Ka, Ob]> = DmTensor::new();
    stage_a_t0::<T, HALF>(device, x, c, &mut terms);
    stage_a_t1::<T, HALF>(device, x, c, &mut terms);
    let mut y: Row<bf16, T, HALF, m![Ob / 32, Oa, Ob % 32]> = DmTensor::new();
    stage_b::<T, HALF>(device, &terms, &c.1, scale, 0, &mut y);
    stage_b::<T, HALF>(device, &terms, &c.2, scale, 1, &mut y);
    unsafe { y.reshape() }
}

axes![Kq = 4096, Sl8 = 8, Sgo = 32, Sgi = 8];

/// The per-row scales of a pass, gathered into slice 0 of each cluster through the switch (one
/// flit per source slice, the scale replicated across the packet by the fetch), compacted to eight
/// rows per packet by a one-hot mask and an intra-slice sum (`s * 1 + 0 *...`, exact), and stored as
/// one contiguous run per cluster: `scales[c][r]` is the scale of row `c * HALF + r`.
pub(crate) fn store_scales<T: AxisName, const HALF: usize, SE: M>(
    device: &mut Device,
    s: Row<f32, T, HALF, m![1 # 8]>,
    onehot: &VrfTensor<f32, Chip, TCl<T, HALF>, m![1 # 256], m![Sgi, Sl8]>,
    scales: HbmTensorViewMut<'_, f32, Chip, SE>,
) {
    // Replicate the scale across the packet: a VRF operand read in broadcast feeds every lane
    // (`max(s, s)`, independent of the stream).
    // Padding slices of a small bucket hold the DM's previous contents, possibly NaN or Inf, and
    // the one-hot compaction below multiplies them by 0: clamp every slice's value to a finite
    // magnitude first (bits AND 0x7fffffff, then at most f32::MAX's bits). Live scales are
    // positive and finite, so they pass unchanged.
    let s: Row<f32, T, HALF, m![1 # 8]> = device
        .main
        .begin(s.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, i32::MAX)
        .vector_clip(ClipBinaryOpI32::Min, 0x7f7f_ffff)
        .vector_reinterpret::<f32>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let sv: VrfTensor<f32, Chip, TCl<T, HALF>, TSl<T, HALF>, m![1 # 8]> =
        device.sub.begin(s.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let s: Row<f32, T, HALF, m![Sl8]> = unsafe { s.reshape() };
    let s8: Row<f32, T, HALF, m![Sl8]> = device
        .main
        .begin(s.view())
        .fetch::<m![1], m![Sl8]>()
        .collect::<m![1], m![Sl8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip_with_mode(ClipBinaryOpF32::Max, BinaryArgMode::Mode11, &sv)
        .vector_final()
        .commit_trim::<m![Sl8]>()
        .commit();
    let s8: DmTensor<f32, Chip, TCl<T, HALF>, m![Sgo, Sgi], m![Sl8]> = unsafe { s8.reshape() };
    let g: DmTensor<f32, Chip, TCl<T, HALF>, m![1 # 256], m![Sgo, Sl8]> = device
        .main
        .begin(s8.view())
        .fetch::<m![1], m![Sl8]>()
        .switch::<m![1 # 256], m![Sgo, Sgi]>(SwitchConfig::Broadcast1 { slice1: 256, slice0: 1 })
        .collect::<m![Sgo, Sgi], m![Sl8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Sgo, Sgi, Sl8 / 4], m![Sl8 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), onehot)
        .vector_intra_slice_reduce::<Sgi, m![Sgo, Sl8 / 4], m![Sl8 % 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_concat::<m![Sgo], m![Sl8]>()
        .vector_final()
        .commit_trim::<m![Sl8]>()
        .commit();
    let g: DmTensor<f32, Chip, TCl<T, HALF>, m![1 # 256], m![T % { HALF } # 256]> = unsafe { g.reshape() };
    // Each cluster writes its rows' live scales, one run.
    g.view().to_hbm_view(&mut device.tdma, scales);
}

/// The one-hot `[8, 8]` f32 pattern (`[r % 8][lane] = (r % 8 == lane)`) in the VRF of slice 0 of
/// each cluster, for `store_scales`.
pub(crate) fn onehot<T: AxisName, const HALF: usize>(
    device: &mut Device,
    eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>,
) -> VrfTensor<f32, Chip, TCl<T, HALF>, m![1 # 256], m![Sgi, Sl8]> {
    // Only slice 0 of each cluster holds it (one small read).
    let d: DmTensor<f32, Chip, m![Dummy2], m![1 # 256], m![Sgi, Sl8]> = eye.to_dm(&mut device.tdma);
    let d: DmTensor<f32, Chip, TCl<T, HALF>, m![1 # 256], m![Sgi, Sl8]> = unsafe { d.reshape() };
    device.sub.begin(d.view()).fetch::<m![Sgi], m![Sl8]>().collect::<m![Sgi], m![Sl8]>().to_vrf()
}

/// The signed norm weight `gamma * D` in each slice (from the `Rc` HBM copies).
pub(crate) fn gamma<T: AxisName, const HALF: usize, PP: super::passes::Passes>(device: &mut Device, g: &HbmTensor<bf16, Chip, m![Rc, H]>) -> Row<bf16, T, HALF, m![Kh]> {
    let d: Row<bf16, T, HALF, m![H]> = PP::table::<T, HALF, m![H]>(device, g.view());
    unsafe { d.reshape() }
}

/// A per-slice constant (`[Rc, E]`, 16 identical copies in HBM) in the DM of every slice: slice
/// `s` reads copy `s % 16`, so 256 slices don't hot-spot one HBM region.
pub(crate) fn table_all<T: AxisName, const HALF: usize, E: M>(device: &mut Device, src: HbmTensorView<'_, bf16, Chip, m![Rc, { E }]>) -> Row<bf16, T, HALF, E> {
    let d: DmTensor<bf16, Chip, m![Dummy2], m![Dum16, Rc], E> = src.to_dm(&mut device.tdma);
    unsafe { d.reshape() }
}

/// One half (`Kh / 1920 = i`) of `h = bf16(x / rms * g)`.
fn normalize_half<T: AxisName, const HALF: usize>(
    device: &mut Device,
    x: &Row<bf16, T, HALF, m![Kh]>,
    g: &Row<bf16, T, HALF, m![Kh]>,
    rms: &VrfTensor<f32, Chip, TCl<T, HALF>, TSl<T, HALF>, m![1 # 8]>,
    i: usize,
    h: &mut Row<bf16, T, HALF, m![Kh]>,
) {
        let gv: VrfTensor<f32, Chip, TCl<T, HALF>, TSl<T, HALF>, m![Kh % 1920]> = device
            .sub
            .begin(g.view().tile::<m![Kh / 1920], 1, m![Kh / 1920 = 1 # 2, Kh % 1920]>(i))
            .fetch::<m![Kh % 1920 / 16], m![Kh % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![Kh % 1920 / 8], m![Kh % 8]>()
            .to_vrf();
        device
            .main
            .begin(x.view().tile::<m![Kh / 1920], 1, m![Kh / 1920 = 1 # 2, Kh % 1920]>(i))
            .fetch::<m![Kh % 1920 / 16], m![Kh % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![Kh % 1920 / 8], m![Kh % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Kh % 1920 / 4], m![Kh % 4]>()
            .vector_fp_binary(FpBinaryOp::DivF, rms)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gv)
            .vector_widen_concat::<m![Kh % 1920 / 8], m![Kh % 8]>()
            .vector_final()
            .cast::<bf16, m![Kh % 8 # 16]>()
            .commit_trim::<m![Kh % 8]>()
            .commit_view(h.view_mut().tile::<m![Kh / 1920], 1, m![Kh / 1920 = 1 #{!} 2, Kh % 1920]>(i));
    }

/// `shared::rmsnorm::normalize` of one row in one slice: `ms = sum(x^2) / 3840 + eps` (an f32
/// fold over the row), `rms = sqrt(ms)`, `h = bf16(x / rms * g)` with the signed gamma `g`. The
/// row is processed in halves so the f32 gamma fits the VRF.
pub(crate) fn normalize<T: AxisName, const HALF: usize>(
    device: &mut Device,
    x: &Row<bf16, T, HALF, m![Kh]>,
    g: &Row<bf16, T, HALF, m![Kh]>,
) -> Row<bf16, T, HALF, m![Kh]> {
    let ms: Row<f32, T, HALF, m![1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![Kh / 16], m![Kh % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Kh / 8], m![Kh % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Kh / 4], m![Kh % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Kh, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(KH_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, crate::EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: Row<f32, T, HALF, m![1 # 8]> = device
        .main
        .begin(ms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: VrfTensor<f32, Chip, TCl<T, HALF>, TSl<T, HALF>, m![1 # 8]> =
        device.sub.begin(rms.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let mut h: Row<bf16, T, HALF, m![Kh]> = DmTensor::new();
    normalize_half::<T, HALF>(device, x, g, &rms, 0, &mut h);
    normalize_half::<T, HALF>(device, x, g, &rms, 1, &mut h);
    h
}

/// One pass (`PI`) of the input norm alone (the BF16 carrier): `h = bf16(x / rms * g)` stored
/// token-major.
pub(crate) fn normalize_pass<T: AxisName, const HALF: usize, const PS: usize, const P: usize, const PI: usize>(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![T, H]>,
    g: &Row<bf16, T, HALF, m![Kh]>,
    h: &mut HbmTensor<bf16, Chip, m![T, H]>,
) {
    let xd: Row<bf16, T, HALF, m![H]> = x
        .view()
        .tile::<m![T / { PS }], 1, m![T / { PS } = 1 # { P }, T % { PS }, H]>(PI)
        .to_dm(&mut device.tdma);
    let xd: Row<bf16, T, HALF, m![Kh]> = unsafe { xd.reshape() };
    let y = normalize::<T, HALF>(device, &xd, g);
    let y: Row<bf16, T, HALF, m![H]> = unsafe { y.reshape() };
    y.view().to_hbm_view(
        &mut device.tdma,
        h.view_mut().tile::<m![T / { PS }], 1, m![T / { PS } = 1 #{!} { P }, T % { PS }, H]>(PI),
    );
}
