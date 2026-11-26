//! Shared pieces of the tokenwise segments: token groups, group broadcasts and the
//! hidden-state RMS norm on eight tokens at a time.
//!
//! A segment's token axis `T` (size `G * 8`) is processed in `G` groups of eight; inside a
//! group the token index `T % 8` rides the TRF lanes of every contraction and the packet of
//! every lane-major vector pass. Phases exchange activations through token-major HBM
//! tensors (`[T, D]`), so each phase chooses its own on-chip layout.
use furiosa_opt_std::prelude::*;

use crate::axes::{Dummy8, H};
use crate::{Chip, EPS};

/// The tile offset of token group `g`. A single-group bucket keeps the loop but its group
/// tile is a no-op, which the compiler accepts only at a literal zero offset. The literal comes
/// from the impl the group count selects (`Groups<1>`), not from a branch: Opt 0.9.0 lowers an
/// `if` on a const generic to a run-time select, which a no-op tile refuses and every other
/// bucket would compute on the device.
#[inline(always)]
pub(crate) fn at<const G: usize>(g: usize) -> usize {
    <Groups<G> as GroupAt>::at(g)
}

/// A bucket's group count as a type, for [`at`].
pub(crate) struct Groups<const G: usize>;

pub(crate) trait GroupAt {
    fn at(g: usize) -> usize;
}

impl<const G: usize> GroupAt for Groups<G> {
    #[inline(always)]
    default fn at(g: usize) -> usize {
        g
    }
}

impl GroupAt for Groups<1> {
    #[inline(always)]
    fn at(_g: usize) -> usize {
        0
    }
}

/// Rows `[8g, 8g + 8)` of a token-major HBM tensor.
pub(crate) fn group_view<'a, T: AxisName, D: M, const G: usize>(
    x: &'a HbmTensor<bf16, Chip, m![T, { D }]>,
    g: usize,
) -> HbmTensorView<'a, bf16, Chip, m![T / 8 = 1 # { G }, T % 8, { D }]> {
    x.view().tile::<m![T / 8], 1, m![T / 8 = 1 # { G }, T % 8, { D }]>(at::<G>(g))
}

/// Writable rows `[8g, 8g + 8)` of a token-major HBM tensor.
pub(crate) fn group_view_mut<'a, T: AxisName, D: M, const G: usize>(
    x: &'a mut HbmTensor<bf16, Chip, m![T, { D }]>,
    g: usize,
) -> HbmTensorViewMut<'a, bf16, Chip, m![T / 8 = 1 #{!} { G }, T % 8, { D }]> {
    x.view_mut().tile::<m![T / 8], 1, m![T / 8 = 1 #{!} { G }, T % 8, { D }]>(at::<G>(g))
}

// ------------------------------------------------------------------- hidden-state RMS norm

const H_F32: f32 = H::SIZE as f32;

/// Eight tokens, four per cluster, each spread over eight slices of 480 elements.
pub(crate) type NormCl<T> = m![T / 4 % 2];
pub(crate) type NormSl<T> = m![T % 4 # 32, H / 480];
type NormRed<T> = m![T % 4 # 32, Dummy8];
pub(crate) type NormRows<T> = DmTensor<bf16, Chip, NormCl<T>, NormSl<T>, m![H % 480]>;
pub(crate) type NormWeight<T> = VrfTensor<f32, Chip, NormCl<T>, NormSl<T>, m![H % 480]>;

/// A gamma `[H]` in every token's eight slices (the DMA replicates it over the token slots).
pub(crate) fn norm_weight<T: AxisName>(device: &mut Device, w: &HbmTensor<bf16, Chip, m![H]>) -> NormWeight<T> {
    let w: NormRows<T> = w.to_dm(&mut device.tdma);
    device
        .sub
        .begin(w.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf()
}

/// `shared::rmsnorm::normalize` for eight tokens: the same command chain per slice, the same
/// partial sums of 480 squares divided by H, the same eight-slice reduction, epsilon and sqrt,
/// then `x / rms * gamma` narrowed to bf16.
pub(crate) fn normalize<T: AxisName>(device: &mut Device, x: &NormRows<T>, w: &NormWeight<T>) -> NormRows<T> {
    let partial: DmTensor<f32, Chip, NormCl<T>, NormSl<T>, m![1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let sum: DmTensor<f32, Chip, NormCl<T>, NormRed<T>, m![1 # 8]> = device
        .main
        .begin(partial.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<NormRed<T>, m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, NormCl<T>, NormRed<T>, m![1 # 8]> = device
        .main
        .begin(sum.view())
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
    let rms: DmTensor<f32, Chip, NormCl<T>, NormSl<T>, m![1 # 8]> = unsafe { rms.reshape() };
    let rms: VrfTensor<f32, Chip, NormCl<T>, NormSl<T>, m![1 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    device
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), w)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}
