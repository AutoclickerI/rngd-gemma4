//! The per-token phases run in passes of `PS = 2 HALF <= 512` rows (one row per slice). A rolled
//! loop over the passes costs runtime command overhead and hides cross-pass overlap, and a guarded
//! second call does not compile when it writes a parameter, so the pass count is a type: the stage
//! is monomorphized with `OnePass` or `TwoPasses`.
use furiosa_opt_std::prelude::*;

use super::post;
use super::rotate::{self, H60Hbm, Kb, Kh, Ob, Ra, Rc, Sgi, Sl8, TCl, TSl};
use crate::Chip;
use crate::device::proj_fast::rotq::RotPasses;
use crate::axes::H;

type Row<D, T, const HALF: usize, E> = DmTensor<D, Chip, TCl<T, HALF>, TSl<T, HALF>, E>;
type Onehot<T, const HALF: usize> = VrfTensor<f32, Chip, TCl<T, HALF>, m![1 # 256], m![Sgi, Sl8]>;

#[allow(clippy::too_many_arguments)]
pub(crate) trait Passes {
    /// A per-slice constant table (`[Rc, E]`, 16 copies) in the per-token layout's slices.
    fn table<T: AxisName, const HALF: usize, E: M>(device: &mut Device, src: HbmTensorView<'_, bf16, Chip, m![Rc, { E }]>) -> Row<bf16, T, HALF, E>;
    fn norm_quantize_natural<T: AxisName, const HALF: usize, const PS: usize>(
        device: &mut Device, x: &HbmTensor<bf16, Chip, m![T, H]>,
        g: &Row<bf16, T, HALF, m![Kh]>, onehot: &Onehot<T, HALF>,
        codes: &mut HbmTensor<i8, Chip, m![T, H]>, scales: &mut HbmTensor<f32, Chip, m![T]>,
    );
    fn normalize_rows<T: AxisName, const HALF: usize, const PS: usize>(
        device: &mut Device,
        x: &HbmTensor<bf16, Chip, m![T, H]>,
        g: &Row<bf16, T, HALF, m![Kh]>,
        h: &mut HbmTensor<bf16, Chip, m![T, H]>,
    );
    fn post_attention<T: AxisName, const HALF: usize, const PS: usize>(
        device: &mut Device,
        y: &HbmTensor<bf16, Chip, m![T, H]>,
        gamma: &Row<bf16, T, HALF, m![Kh]>,
        x: &mut HbmTensor<bf16, Chip, m![T, H]>,
    );
    fn post_ffn<T: AxisName, const HALF: usize, const PS: usize>(
        device: &mut Device,
        y: &HbmTensor<bf16, Chip, m![T, H]>,
        gamma: &Row<bf16, T, HALF, m![Kh]>,
        scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
        x: &mut HbmTensor<bf16, Chip, m![T, H]>,
    );
}

/// `T <= 512`.
pub(crate) struct OnePass;
pub(crate) struct TwoPasses;
#[cfg(feature = "compile-carried")]
/// Three 512-row passes for Tok1536.
pub(crate) struct ThreePasses;

macro_rules! passes_impl {
    ($ty:ident, $p:literal, [$($i:literal),*], $table:ident) => {
        impl Passes for $ty {
            fn table<T: AxisName, const HALF: usize, E: M>(device: &mut Device, src: HbmTensorView<'_, bf16, Chip, m![Rc, { E }]>) -> Row<bf16, T, HALF, E> {
                rotate::$table::<T, HALF, E>(device, src)
            }
            fn norm_quantize_natural<T: AxisName, const HALF: usize, const PS: usize>(
                device: &mut Device, x: &HbmTensor<bf16, Chip, m![T, H]>,
                g: &Row<bf16, T, HALF, m![Kh]>, onehot: &Onehot<T, HALF>,
                codes: &mut HbmTensor<i8, Chip, m![T, H]>, scales: &mut HbmTensor<f32, Chip, m![T]>,
            ) {
                $(
                    let xd: Row<bf16, T, HALF, m![H]> = x.view()
                        .tile::<m![T / { PS }], 1, m![T / { PS } = 1 # $p, T % { PS }, H]>($i).to_dm(&mut device.tdma);
                    let xd: Row<bf16, T, HALF, m![Kh]> = unsafe { xd.reshape() };
                    super::rotate2::norm_quantize_pass_dm_i8::<T, HALF, PS, $p, $i>(device, &xd, g, onehot, codes, scales);
                )*
            }
            fn normalize_rows<T: AxisName, const HALF: usize, const PS: usize>(
                device: &mut Device,
                x: &HbmTensor<bf16, Chip, m![T, H]>,
                g: &Row<bf16, T, HALF, m![Kh]>,
                h: &mut HbmTensor<bf16, Chip, m![T, H]>,
            ) {
                $(rotate::normalize_pass::<T, HALF, PS, $p, $i>(device, x, g, h);)*
            }
            fn post_attention<T: AxisName, const HALF: usize, const PS: usize>(
                device: &mut Device,
                y: &HbmTensor<bf16, Chip, m![T, H]>,
                gamma: &Row<bf16, T, HALF, m![Kh]>,
                x: &mut HbmTensor<bf16, Chip, m![T, H]>,
            ) {
                $(post::post_attention_pass::<T, HALF, PS, $p, $i>(device, y, gamma, x);)*
            }
            fn post_ffn<T: AxisName, const HALF: usize, const PS: usize>(
                device: &mut Device,
                y: &HbmTensor<bf16, Chip, m![T, H]>,
                gamma: &Row<bf16, T, HALF, m![Kh]>,
                scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
                x: &mut HbmTensor<bf16, Chip, m![T, H]>,
            ) {
                let s = post::scalar_vrf::<T, HALF>(device, scalar);
                $(post::post_ffn_pass::<T, HALF, PS, $p, $i>(device, y, gamma, &s, x);)*
            }
        }
    };
}

passes_impl!(OnePass, 1, [0], table_all);
passes_impl!(TwoPasses, 2, [0, 1], table_all);
#[cfg(feature = "compile-carried")]
passes_impl!(ThreePasses, 3, [0, 1, 2], table_all);

// ------------------------------------------- int8s: the QKV GEMM input codes per token axis

/// Select natural or rotated INT8 QKV by token axis. Opt 0.9.0 lowers an ordinary
/// size guard into both device branches and cannot translate parameter writes inside it.
/// Preserve each layout's allocation order for reproducible carried images.
pub(crate) trait Int8sBucket: AxisName {
    /// `core8_full::full_qkv_gemm_i8`'s codes `[T, H]` and per-token scales.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn sliding_qkv_codes<T: AxisName, const HALF: usize, const PS2: usize, PP: RotPasses + Passes>(
        device: &mut Device,
        x: &HbmTensor<bf16, Chip, m![T, H]>,
        input_rms_weight_signed: &HbmTensor<bf16, Chip, m![Rc, H]>,
        h64: &HbmTensor<bf16, Chip, m![Rc, Ob, Kb]>,
        h60: &HbmTensor<bf16, Chip, m![Rc, { H60Hbm }]>,
        eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>,
    ) -> (HbmTensor<i8, Chip, m![T, H]>, HbmTensor<f32, Chip, m![T]>);
}

/// The rotated Sliding QKV codes, in the pre-int8s order of `project_v1`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn sliding_qkv_codes_rotated<T: AxisName, const HALF: usize, const PS2: usize, PP: RotPasses + Passes>(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![T, H]>,
    input_rms_weight_signed: &HbmTensor<bf16, Chip, m![Rc, H]>,
    h64: &HbmTensor<bf16, Chip, m![Rc, Ob, Kb]>,
    h60: &HbmTensor<bf16, Chip, m![Rc, { H60Hbm }]>,
    eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>,
) -> (HbmTensor<i8, Chip, m![T, H]>, HbmTensor<f32, Chip, m![T]>) {
    let mut codes = HbmTensor::<i8, Chip, m![T, Ob / 32, Ra, Ob % 32]>::new();
    let mut scales = HbmTensor::<f32, Chip, m![T]>::new();
    let onehot = rotate::onehot::<T, HALF>(device, eye);
    let g = rotate::gamma::<T, HALF, PP>(device, input_rms_weight_signed);
    <PP as RotPasses>::norm_rotate_quantize::<T, HALF, PS2>(device, x, &g, h64, h60, rotate::INV_SQRT_3840, &onehot, &mut codes, &mut scales);
    let codes: HbmTensor<i8, Chip, m![T, H]> = unsafe { codes.reshape() };
    (codes, scales)
}

/// The natural codes of both QKV GEMMs (`norm_quantize_natural` is a `Passes` method; the rotation tables are unused).
#[allow(clippy::type_complexity)]
fn qkv_codes_natural<T: AxisName, const HALF: usize, const PS2: usize, PP: Passes>(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![T, H]>,
    input_rms_weight_signed: &HbmTensor<bf16, Chip, m![Rc, H]>,
    eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>,
) -> (HbmTensor<i8, Chip, m![T, H]>, HbmTensor<f32, Chip, m![T]>) {
    let mut scales = HbmTensor::<f32, Chip, m![T]>::new();
    let onehot = rotate::onehot::<T, HALF>(device, eye);
    let mut codes = HbmTensor::<i8, Chip, m![T, H]>::new();
    let g = super::rotate2::gamma_nat_ag2::<T, HALF>(device, input_rms_weight_signed);
    PP::norm_quantize_natural::<T, HALF, PS2>(device, x, &g, &onehot, &mut codes, &mut scales);
    (codes, scales)
}

/// One impl per token axis: `rotated` below 512 rows, `natural` from 512 (`int8s` default form).
macro_rules! int8s_bucket {
    (rotated: $($t:ident),*) => {$(
        impl Int8sBucket for crate::axes::$t {
            fn sliding_qkv_codes<T: AxisName, const HALF: usize, const PS2: usize, PP: RotPasses + Passes>(
                device: &mut Device, x: &HbmTensor<bf16, Chip, m![T, H]>, input_rms_weight_signed: &HbmTensor<bf16, Chip, m![Rc, H]>,
                h64: &HbmTensor<bf16, Chip, m![Rc, Ob, Kb]>, h60: &HbmTensor<bf16, Chip, m![Rc, { H60Hbm }]>,
                eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>,
            ) -> (HbmTensor<i8, Chip, m![T, H]>, HbmTensor<f32, Chip, m![T]>) {
                sliding_qkv_codes_rotated::<T, HALF, PS2, PP>(device, x, input_rms_weight_signed, h64, h60, eye)
            }
        }
    )*};
    (natural: $($t:ident),*) => {$(
        impl Int8sBucket for crate::axes::$t {
            fn sliding_qkv_codes<T: AxisName, const HALF: usize, const PS2: usize, PP: RotPasses + Passes>(
                device: &mut Device, x: &HbmTensor<bf16, Chip, m![T, H]>, input_rms_weight_signed: &HbmTensor<bf16, Chip, m![Rc, H]>,
                _h64: &HbmTensor<bf16, Chip, m![Rc, Ob, Kb]>, _h60: &HbmTensor<bf16, Chip, m![Rc, { H60Hbm }]>,
                eye: &HbmTensor<f32, Chip, m![Sgi, Sl8]>,
            ) -> (HbmTensor<i8, Chip, m![T, H]>, HbmTensor<f32, Chip, m![T]>) {
                qkv_codes_natural::<T, HALF, PS2, PP>(device, x, input_rms_weight_signed, eye)
            }
        }
    )*};
}

int8s_bucket!(rotated: Tok8, Tok32, Tok128, Tok256, Tok384);
int8s_bucket!(natural: Tok512, Tok640, Tok768, Tok896, Tok1024);
#[cfg(feature = "compile-carried")]
int8s_bucket!(natural: Tok1536);
