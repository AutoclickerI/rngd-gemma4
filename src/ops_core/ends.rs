//! Gather the emitting rows from layer 47, run Full O, the FFN and the final norm,
//! then stream the tied head weights to produce softcapped BF16 logits.

use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;
use crate::device::layout::Cluster;
use crate::device::tokenwise::head;
use crate::device::tokenwise_int8::passes::{OnePass, Passes};
use crate::device::tokenwise_int8::{core16, core16_full};
use crate::ops_core::{FfnBf16, FullOBf16};

mod head16;
pub mod head4;

/// Layer 47's O and FFN on the core for the `T` emitting rows, then the final norm into `h`
/// (`G = T / 8` groups of eight). `HS` is the gather's slice layout
/// (`[T, Hh, 1 # k]`, `2 T k = 256`); the core's per-token
/// phases take `HALF = T / 2`, `PS2 = T`, one pass; its GEMM rings `TS` token slots of `[TT, TR]`
/// tokens over the O/Down rows' slice digits (`ORING`, `OXS`, `OYS` over `H / 16`) and the
/// Up/Gate rows' (`URING`, `UXS`, `UYS` over `L`).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn prefix<
    T: AxisName,
    HS: M,
    const G: usize,
    const HALF: usize,
    const PS2: usize,
    PP: Passes,
    ORING: M,
    OXS: M,
    OYS: M,
    URING: M,
    UXS: M,
    UYS: M,
    TT: M,
    TR: M,
    const TS: usize,
>(
    device: &mut Device,
    out_f: &HbmTensor<bf16, Chip, m![Rq, Hqf]>,
    a_rows: &HbmTensor<i32, Chip, m![T, Hh]>,
    x: &HbmTensor<bf16, Chip, m![Tok1024, H]>,
    rows: &HbmTensor<i32, Chip, m![T]>,
    o: &FullOBf16<'_>,
    ffn: &FfnBf16<'_>,
    layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
    final_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    h: &mut HbmTensor<bf16, Chip, m![T, H]>,
) {
    // Layer 47's output halves, gathered as `ops_kv_publish::full_finish_*` gathers them.
    let halves: DmTensor<bf16, Chip, Cluster, HS, m![Hqf]> = out_f.gather::<m![Rq], m![Hqf]>().by_byte_offsets(a_rows).to_dm(&mut device.tdma);
    let mut a3 = HbmTensor::<bf16, Chip, m![T, Hh, Hqf]>::new();
    halves.view().to_hbm_view(&mut device.tdma, a3.view_mut());
    // The same bytes as `[T, Qf]` (`Hh x Hqf = Qf`); the O stage tiles the pre-existing `T` axis.
    let ar: HbmTensor<bf16, Chip, m![T, Qf]> = unsafe { a3.reshape() };
    let mut xr = HbmTensor::<bf16, Chip, m![T, H]>::new();
    head::gather_rows::<Tok1024, m![H], T>(device, x, rows, &mut xr);

    core16_full::full_o_stage::<T, HALF, PS2, PP, ORING, OXS, OYS, TT, TR, TS>(
        device, &ar, &mut xr, o.o_weight, o.o_weight_scale, o.post_attn_rms_weight,
    );
    core16::ffn_stage::<T, HALF, PS2, PP, URING, UXS, UYS, ORING, OXS, OYS, TT, TR, TS>(
        device,
        &mut xr,
        ffn.pre_ff_rms_weight,
        ffn.up_weight_packed,
        ffn.gate_weight_packed,
        ffn.down_weight_packed,
        ffn.up_weight_scale,
        ffn.gate_weight_scale,
        ffn.down_weight_scale,
        ffn.up_global_scale,
        ffn.gate_global_scale,
        ffn.down_global_scale,
        ffn.post_ff_rms_weight,
        layer_scalar,
    );
    head16::final_norm::<T, G>(device, &xr, final_rms_weight, h);
}

/// The head's token tiling: `TS = 8` ring slots, tokens `[T / 8, T % 8]` (one layout for every
/// output bucket; the activation stream is `T x 120` cycles per slice per K half).
type HRing = m![W / 512 % 8];
type HXs = m![W / 64 % 8];
type HYs = m![W / 16 % 4];

use crate::device::tokenwise_int8::{ffn_fast::ug4::Hb,rotate::{Ob,Kb,H60Hbm,Sgi,Sl8}};
macro_rules! w4_heads {
    ($name:ident,$out:ident,$hslice:ty,$groups:literal,$half:literal,$ps2:literal,
     [$oring:ty,$oxs:ty,$oys:ty],[$uring:ty,$uxs:ty,$uys:ty],$ctt:ty,$ctr:ty,$cts:literal) => {
        #[device(chip=1)]
        pub fn $name(
            device:&mut Device,out_f:&HbmTensor<bf16,Chip,m![Rq,Hqf]>,a_rows:&HbmTensor<i32,Chip,m![$out,Hh]>,
            x:&HbmTensor<bf16,Chip,m![Tok1024,H]>,rows:&HbmTensor<i32,Chip,m![$out]>,o:FullOBf16<'_>,ffn:FfnBf16<'_>,
            layer_scalar:&HbmTensor<bf16,Chip,m![1 # 8]>,signed_final_rms:&HbmTensor<bf16,Chip,m![H]>,
            codes:&HbmTensor<i8,Chip,m![W,Hb]>,scales:&HbmTensor<bf16,Chip,m![W]>,
            h64:&HbmTensor<bf16,Chip,m![Rc,Ob,Kb]>,h60:&HbmTensor<bf16,Chip,m![Rc,{H60Hbm}]>,
            eye:&HbmTensor<f32,Chip,m![Sgi,Sl8]>,logits:&mut HbmTensor<bf16,Chip,m![$out,W]>,
        ) {
            let first=head4::load_chunk::<Hb>(device,codes,0);
            let ws=head16::load_scales8::<HRing,HXs,HYs>(device,scales);
            let mut h=HbmTensor::<bf16,Chip,m![$out,H]>::new();
            prefix::<$out,$hslice,$groups,$half,$ps2,OnePass,$oring,$oxs,$oys,$uring,$uxs,$uys,$ctt,$ctr,$cts>(device,out_f,a_rows,x,rows,&o,&ffn,layer_scalar,signed_final_rms,&mut h);
            head4::logits::<$out,$half,$ps2>(device,&h,codes,&ws,first,h64,h60,eye,logits);
        }
    }
}
w4_heads!(last_bf16_out128,Out128,m![Out128,Hh],16,64,128,
 [m![1],m![H/64 # 64],m![H/16%4]],[m![1],m![L/512 # 32,L/64%2],m![L/16%4]],m![Out128/64],m![Out128%64],64);
