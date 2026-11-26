//! Sliding-to-Full transitions over E4M3 KV pools.
//! They publish the Sliding layer's KV and prepare the next Full layer's Q/K/V.

use furiosa_opt_std::prelude::*;

use super::Int8Consts;
use crate::Chip;
use crate::axes::*;

// Full QKV and the corresponding row moves share the head phase.
macro_rules! mid_sf_i8_v9_entries {
    ($tok:ident, $name:ident, $body:ident) => {
        /// Sliding -> Full with the same projection and FFN body as the resident RSF program.
        /// Resident group selection depends on the other requests in the step. It must not
        /// select projection numerics: the standalone and resident paths call one body.
        #[device(chip = 1)]
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            device: &mut Device,
            out_s: &HbmTensor<bf16, Chip, m![Rq, Hqs]>,
            rows_s: &HbmTensor<i32, Chip, m![$tok, Hh]>,
            stage_s: &HbmTensor<i32, Chip, m![$tok]>,
            publish_s: &HbmTensor<i32, Chip, m![$tok]>,
            rope_f: &HbmTensor<i32, Chip, m![$tok]>,
            rows_f: &HbmTensor<i32, Chip, m![$tok, Hh]>,
            publish_f: &HbmTensor<i32, Chip, m![$tok]>,
            x: &mut HbmTensor<bf16, Chip, m![$tok, H]>,
            consts: Int8Consts<'_>,
            o: crate::ops_tokenwise_int8::SlidingOStaticF8<'_>,
            ffn: crate::ops_tokenwise_int8::StaticF8FfnWeights<'_>,
            layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
            next: crate::ops_core::full_resident::FullQkvStaticF8<'_>,
            k_l: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
            v_l: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
            qin_f: &mut HbmTensor<bf16, Chip, m![Rq, Hqf]>,
            k_n: &mut HbmTensor<f8e4m3, Chip, m![Rf, Df]>,
            v_n: &mut HbmTensor<f8e4m3, Chip, m![Rf, Df]>,
        ) {
            super::full_resident::$body(
                device, out_s, rows_s, stage_s, publish_s, rope_f, rows_f, publish_f, x, &consts, &o, &ffn, layer_scalar, &next,
                k_l, v_l, qin_f, k_n, v_n,
            );
        }
    };
}

mid_sf_i8_v9_entries!(Tok1024, mid_sf_int8_fp8kv_t1024, body_t1024);

// Same body as the wide resident paths, preserving fallback arithmetic.
#[cfg(feature = "compile-carried")]
mid_sf_i8_v9_entries!(Tok1536, mid_sf_int8_fp8kv_t1536, body_t1536);
