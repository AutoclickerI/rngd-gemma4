//! Per-token rotation and quantization. Load the Hadamard constants into DM once
//! per stage and fill the TRF from those copies for each token pass.

use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Dummy2, H};
use crate::device::tokenwise_int8::passes::{OnePass, TwoPasses};
use crate::device::tokenwise_int8::rotate::{self, Dum16, H60Hbm, H60Trf, H64Trf, Ka, Kb, Kh, Kq, Ob, Oa, Ra, Rc, RotK, RotPad, Sgi, Sl8, TCl, TSl};

type Row<D, T, const HALF: usize, E> = DmTensor<D, Chip, TCl<T, HALF>, TSl<T, HALF>, E>;
type Onehot<T, const HALF: usize> = VrfTensor<f32, Chip, TCl<T, HALF>, m![1 # 256], m![Sgi, Sl8]>;

const MAGIC: f32 = 12_582_912.0;
const MAGIC_NEG_I32: i32 = -12_582_912;

/// The H64 and H60 constants from their `Rc` HBM copies into every slice's DM.
pub(crate) fn consts_dm<T: AxisName, const HALF: usize>(
    device: &mut Device,
    h64: &HbmTensor<bf16, Chip, m![Rc, Ob, Kb]>,
    h60: &HbmTensor<bf16, Chip, m![Rc, { H60Hbm }]>,
) -> (Row<bf16, T, HALF, m![Ob, Kb]>, Row<bf16, T, HALF, H60Hbm>) {
    let d64: DmTensor<bf16, Chip, m![Dummy2], m![Dum16, Rc], m![Ob, Kb]> = h64.to_dm(&mut device.tdma);
    let d60: DmTensor<bf16, Chip, m![Dummy2], m![Dum16, Rc], H60Hbm> = h60.to_dm(&mut device.tdma);
    (unsafe { d64.reshape() }, unsafe { d60.reshape() })
}

/// H64 from its DM copy into the TRF.
pub(crate) fn h64_trf<T: AxisName, const HALF: usize>(device: &mut Device, d64: &Row<bf16, T, HALF, m![Ob, Kb]>) -> H64Trf<T, HALF> {
    device
        .sub
        .begin(d64.view())
        .fetch::<m![Ob % 8, Ob / 8, Kb / 16], m![Kb % 16]>()
        .collect::<m![Ob % 8, Ob / 8, Kb / 16], m![Kb % 16]>()
        .to_trf()
}

pub(crate) fn h60_half<T: AxisName, const HALF: usize>(device: &mut Device, d60: &Row<bf16, T, HALF, H60Hbm>, i: usize) -> H60Trf<T, HALF> {
    device
        .sub
        .begin(d60.view().tile::<m![Oa / 32], 1, m![Oa / 32 = 1 # 2, Oa % 8, Ka, Oa / 8 % 4]>(i))
        .fetch::<m![Oa % 8, Ka / 4], m![Ka % 4, Oa / 8 % 4]>()
        .collect::<m![Oa % 8, Ka / 4], m![Ka % 4, Oa / 8 % 4]>()
        .to_trf()
}

/// `s = max(amax / 127, 1e-30)` of one row (f32, in the eight-lane scalar layout).
fn row_scale<T: AxisName, const HALF: usize>(device: &mut Device, y: &Row<bf16, T, HALF, RotPad>) -> Row<f32, T, HALF, m![1 # 8]> {
    let y: DmTensorView<'_, bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![Kq]> = unsafe { y.view().reshape() };
    device
        .main
        .begin(y)
        .fetch::<m![Kq / 16], m![Kq % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Kq / 8], m![Kq % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, i32::MAX)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![Kq / 4], m![Kq % 4]>()
        .vector_intra_slice_reduce::<Kq, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_fp_div(127f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1e-30f32)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

/// `q = clamp(rne(y / s), -127, 127)` as i8, over the live rows only (a dense code row).
fn quantize<T: AxisName, const HALF: usize>(
    device: &mut Device,
    y: &Row<bf16, T, HALF, RotPad>,
    s: &Row<f32, T, HALF, m![1 # 8]>,
) -> Row<i8, T, HALF, RotK> {
    let s: VrfTensor<f32, Chip, TCl<T, HALF>, TSl<T, HALF>, m![1 # 8]> =
        device.sub.begin(s.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    device
        .main
        .begin(y.view())
        .fetch::<m![Ob / 32, Ra, Ob % 32 / 16], m![Ob % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ob / 32, Ra, Ob % 32 / 8], m![Ob % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ob / 32, Ra, Ob % 32 / 4], m![Ob % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &s)
        .vector_fp_binary(FpBinaryOp::AddF, MAGIC)
        .vector_widen_concat::<m![Ob / 32, Ra, Ob % 32 / 8], m![Ob % 8]>()
        .vector_fp_to_fxp(31)
        .vector_clip(ClipBinaryOpI32::AddFxp, MAGIC_NEG_I32)
        .vector_clip(ClipBinaryOpI32::Min, 127)
        .vector_clip(ClipBinaryOpI32::Max, -127)
        .vector_final()
        .cast::<i8, m![Ob % 8 # 32]>()
        .commit_trim::<m![Ob % 8]>()
        .commit()
}

/// Normalize, rotate and quantize one pass using the TRF constants.
#[allow(clippy::too_many_arguments)]
fn norm_rotate_quantize_pass<T: AxisName, const HALF: usize, const PS: usize, const P: usize, const PI: usize>(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![T, H]>,
    g: &Row<bf16, T, HALF, m![Kh]>,
    d64: &Row<bf16, T, HALF, m![Ob, Kb]>,
    d60: &Row<bf16, T, HALF, H60Hbm>,
    scale: f32,
    onehot: &Onehot<T, HALF>,
    codes: &mut HbmTensor<i8, Chip, m![T, Ob / 32, Ra, Ob % 32]>,
    scales: &mut HbmTensor<f32, Chip, m![T]>,
) {
    let c = (h64_trf::<T, HALF>(device, d64), h60_half::<T, HALF>(device, d60, 0), h60_half::<T, HALF>(device, d60, 1));
    let xd: Row<bf16, T, HALF, m![H]> = x
        .view()
        .tile::<m![T / { PS }], 1, m![T / { PS } = 1 # { P }, T % { PS }, H]>(PI)
        .to_dm(&mut device.tdma);
    let xd: Row<bf16, T, HALF, m![Kh]> = unsafe { xd.reshape() };
    let h = rotate::normalize::<T, HALF>(device, &xd, g);
    let h: Row<bf16, T, HALF, m![Ka, Kb]> = unsafe { h.reshape() };
    let y = rotate::rotate::<T, HALF>(device, &h, &c, scale);
    let s = row_scale::<T, HALF>(device, &y);
    let q = quantize::<T, HALF>(device, &y, &s);
    q.view().to_hbm_view(
        &mut device.tdma,
        codes
            .view_mut()
            .tile::<m![T / { PS }], 1, m![T / { PS } = 1 #{!} { P }, T % { PS }, Ob / 32, Ra, Ob % 32]>(PI),
    );
    rotate::store_scales::<T, HALF, m![T / { PS } = 1 #{!} { P }, T % { PS }]>(
        device,
        s,
        onehot,
        scales.view_mut().tile::<m![T / { PS }], 1, m![T / { PS } = 1 #{!} { P }, T % { PS }]>(PI),
    );
}

/// The per-token phases of the QKV and O stages over all passes, constants loaded once.
#[allow(clippy::too_many_arguments)]
pub(crate) trait RotPasses {
    fn norm_rotate_quantize<T: AxisName, const HALF: usize, const PS: usize>(
        device: &mut Device,
        x: &HbmTensor<bf16, Chip, m![T, H]>,
        g: &Row<bf16, T, HALF, m![Kh]>,
        h64: &HbmTensor<bf16, Chip, m![Rc, Ob, Kb]>,
        h60: &HbmTensor<bf16, Chip, m![Rc, { H60Hbm }]>,
        scale: f32,
        onehot: &Onehot<T, HALF>,
        codes: &mut HbmTensor<i8, Chip, m![T, Ob / 32, Ra, Ob % 32]>,
        scales: &mut HbmTensor<f32, Chip, m![T]>,
    );
}

macro_rules! rot_passes_impl {
    ($ty:ty, $p:literal, [$($i:literal),*]) => {
        impl RotPasses for $ty {
            fn norm_rotate_quantize<T: AxisName, const HALF: usize, const PS: usize>(
                device: &mut Device,
                x: &HbmTensor<bf16, Chip, m![T, H]>,
                g: &Row<bf16, T, HALF, m![Kh]>,
                h64: &HbmTensor<bf16, Chip, m![Rc, Ob, Kb]>,
                h60: &HbmTensor<bf16, Chip, m![Rc, { H60Hbm }]>,
                scale: f32,
                onehot: &Onehot<T, HALF>,
                codes: &mut HbmTensor<i8, Chip, m![T, Ob / 32, Ra, Ob % 32]>,
                scales: &mut HbmTensor<f32, Chip, m![T]>,
            ) {
                let (d64, d60) = consts_dm::<T, HALF>(device, h64, h60);
                $(norm_rotate_quantize_pass::<T, HALF, PS, $p, $i>(device, x, g, &d64, &d60, scale, onehot, codes, scales);)*
            }
        }
    };
}

rot_passes_impl!(OnePass, 1, [0]);
rot_passes_impl!(TwoPasses, 2, [0, 1]);

#[cfg(feature = "compile-carried")]
use crate::device::tokenwise_int8::passes::ThreePasses;

#[cfg(feature = "compile-carried")]
rot_passes_impl!(ThreePasses, 3, [0, 1, 2]);
