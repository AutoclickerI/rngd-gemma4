//! Q4 Full → Sliding segments fuse the next layer's Sliding attention.
//! The decode groups and prompt rounds use the same layout as the RSS programs.

use furiosa_opt_std::prelude::*;

use super::resident::{
    Pool, Prompt941, Prompt999, PROMPT_BLOCK, load_rs_band, load_rs_mid, load_rs_w1,
    Rr, store_round, wtop,
};
use super::{FullOInt8, Int8Consts, Int8SlidingQkvWeights};
use crate::Chip;
#[cfg(feature = "compile-carried")]
use crate::axes::{Ds, Ns};
use crate::axes::{H, Hh, Hqf, Hqs, Rq, Tok1024};
use crate::ops_dsl_attention::paired::pair384 as dgpair;
use crate::ops_dsl_attention::prompt_blocks::{self as pb, D, G2, N, Q};

/// Full-to-Sliding segments with Q4 feed-forward kernels and paired Sliding attention.
// The server loads these Q4 entries by image path.
pub mod q4 {
    use super::*;
    use crate::ops_core::q4serve::mid_fs_q4_t1024_nat_body;

macro_rules! paired_rfs_hoist_q4_entries {
    ($tok:ident, $body:ident, $name:ident, $gs:ident, [$($g:ident: $w:ident),+], $pt:ident, [$(($off:ident, $tm:ident, $cap:ident, $load:ident, $round:ident, $r:literal)),+]) => {
        /// Layer `l`'s INT8 FP8-KV `mid_fs` and layer `l + 1`'s whole Sliding attention (module doc; hoisted gathers).
        #[device(chip = 1)]
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            device: &mut Device,
            groups: $gs<'_>,
            out_s: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
            out_f: &HbmTensor<bf16, Chip, m![Rq, Hqf]>,
            rows_f: &HbmTensor<i32, Chip, m![$tok, Hh]>,
            rope_s: &HbmTensor<i32, Chip, m![$tok]>,
            rows_s: &HbmTensor<i32, Chip, m![$tok, Hh]>,
            stage_s: &HbmTensor<i32, Chip, m![$tok]>,
            x: &mut HbmTensor<bf16, Chip, m![$tok, H]>,
            consts: Int8Consts<'_>,
            o: FullOInt8<'_>,
            ffn: crate::ops_core::q4serve::Q4FfnWeights<'_>,
            layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
            next: Int8SlidingQkvWeights<'_>,
            qin_s: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
            k_n: &mut Pool,
            v_n: &mut Pool,
            prompt: $pt<'_>,
            out_q: &mut HbmTensor<bf16, Chip, m![Rr, Q, N, G2, D]>,
        ) {
            $(let mut $g = ($w::gather_group(device, groups.$g.kv_off, groups.$g.cap, k_n, v_n), groups.$g);)+
            $body(device, out_f, rows_f, rope_s, rows_s, stage_s, x, consts, o, ffn, layer_scalar, next, qin_s, k_n, v_n);
            $($w::refresh_newest(device, $g.1.kv_off, k_n, v_n, &mut $g.0);)+
            $($w::attend_gathered(device, qin_s, $g.1.rows, &$g.0, out_s);)+
            let masks = pb::Masks { tm_rows: prompt.tm_rows, cap_rows: prompt.cap_rows, cap_hi: prompt.cap_hi };
            $(let $off = $load(device, qin_s, PROMPT_BLOCK + $r, prompt.$off, &pb::Keys { tm: prompt.$tm, cap: prompt.$cap }, &masks, k_n, v_n);)+
            let cap_hi: pb::Cap = prompt.cap_hi.to_dm(&mut device.tdma);
            $(
                let o = pb::$round(device, $off, &cap_hi);
                store_round(device, &o, out_q, PROMPT_BLOCK + $r);
            )+
        }
    };
}

#[cfg(feature = "compile-carried")]
use crate::ops_core::q4serve::mid_fs_q4_t1536_body;

#[cfg(feature = "compile-carried")]
include!("resident_fs_g12_q4.rs");
#[cfg(feature = "compile-carried")]
include!("resident_fs_t1024_g12_q4.rs");
}

// These entry definitions are rebuilt explicitly; the server loads their carried images.
#[cfg(feature = "compile-carried")]
include!("resident_fs_t128.rs");

#[cfg(feature = "compile-carried")]
use crate::axes::Tok1536;
#[cfg(feature = "compile-carried")]
use super::resident::PairGroups12;
#[cfg(feature = "compile-carried")]
use crate::ops_dsl_attention::paired::pair384 as dgpair_pv;
