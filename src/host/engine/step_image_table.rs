//! Generated metadata for the step families carried in
//! `src/host/prebuilt/step.frcz` (one registry row each) and their members (`step_image` module doc).

/// One member of the step families: its definition path, the launch the server binds for it (the bound inputs
/// fill its first `inputs` slots and the bound outputs fill its `outputs` slots).
pub(super) struct Member {
    pub path: &'static str,
    pub inputs: usize,
    pub outputs: &'static [usize],
}

/// One step family: its registry row, its argument slots (the last one the control
/// word) and the individually routed members (indices in `MEMBERS`, in opcode order, with holes marked by `NO_MEMBER`).
pub(super) struct Family {
    pub bucket: usize,
    pub name: &'static str,
    pub row: &'static str,
    pub slots: usize,
    pub ops: &'static [usize],
}

/// Bytes of the server's padding buffer; no family slot is larger.
pub(super) const PADDING_BYTES: usize = 33554432;

pub(super) const NO_MEMBER: usize = usize::MAX;

pub(super) const MEMBERS: &[Member] = &[
    Member { path: "furiosa_opt_gemma4::ops_core::full_resident::rsf_int8_fp8kv_t1024_s8", inputs: 66, outputs: &[36, 37, 38, 39, 40, 41] },
    Member { path: "furiosa_opt_gemma4::ops_dsl_attention::decode::w384_fp8::swa_decode_fp8_w384", inputs: 5, outputs: &[5] },
    Member { path: "furiosa_opt_gemma4::ops_dsl_attention::prefill_full::w512_fp8::full_prefill_fp8_w512_c512", inputs: 6, outputs: &[6] },
    Member { path: "furiosa_opt_gemma4::ops_dsl_attention::prompt_blocks::swa_prompt_fp8_4_4_1", inputs: 20, outputs: &[1, 6, 11] },
    Member { path: "furiosa_opt_gemma4::ops_dsl_attention::decode_full::f2048_fp8::full_decode_fp8_w2048", inputs: 5, outputs: &[5] },
    Member { path: "furiosa_opt_gemma4::ops_dsl_attention::prefill_full::w2048_fp8::full_prefill_fp8_w2048_c512", inputs: 6, outputs: &[6] },
    Member { path: "furiosa_opt_gemma4::ops_dsl_attention::prefill_full::w768_fp8::full_prefill_fp8_w768_c512", inputs: 6, outputs: &[6] },
    Member { path: "furiosa_opt_gemma4::ops_dsl_attention::prompt_blocks::swa_prompt_fp8_4_4_4", inputs: 20, outputs: &[1, 6, 11] },
    Member { path: "furiosa_opt_gemma4::ops_dsl_attention::prefill_full::w256_fp8::full_prefill_fp8_w256_c512", inputs: 6, outputs: &[6] },
    Member { path: "furiosa_opt_gemma4::ops_dsl_attention::prefill_full::w1024c1024_fp8::full_prefill_fp8_w1024_c1024", inputs: 8, outputs: &[8] },
    Member { path: "furiosa_opt_gemma4::ops_dsl_attention::prefill_full::w1024_fp8::full_prefill_fp8_w1024_c512", inputs: 6, outputs: &[6] },
    Member { path: "furiosa_opt_gemma4::ops_dsl_attention::prefill_full::w1280_fp8::full_prefill_fp8_w1280_c512", inputs: 6, outputs: &[6] },
    Member { path: "furiosa_opt_gemma4::ops_dsl_attention::prefill_full::w1536_fp8::full_prefill_fp8_w1536_c512", inputs: 6, outputs: &[6] },
    Member { path: "furiosa_opt_gemma4::ops_dsl_attention::decode_full::f1024_fp8::full_decode_fp8_w1024", inputs: 5, outputs: &[5] },
    Member { path: "furiosa_opt_gemma4::ops_dsl_attention::prefill_full::w1792_fp8::full_prefill_fp8_w1792_c512", inputs: 6, outputs: &[6] },
    Member { path: "furiosa_opt_gemma4::ops_dsl_attention::prefill_full::w2048c1024_fp8::full_prefill_fp8_w2048_c1024", inputs: 8, outputs: &[8] },
    Member { path: "furiosa_opt_gemma4::ops_dsl_attention::decode_full::f3072_fp8::full_decode_fp8_w3072", inputs: 5, outputs: &[5] },
    Member { path: "furiosa_opt_gemma4::ops_dsl_attention::decode_full::f512_fp8::full_decode_fp8_w512", inputs: 5, outputs: &[5] },
    Member { path: "furiosa_opt_gemma4::ops_dsl_attention::prefill::w128_fp8::swa_prefill_fp8_w128_c512", inputs: 7, outputs: &[7] },
    Member { path: "furiosa_opt_gemma4::ops_dsl_attention::prefill::band512_fp8::swa_prefill_fp8_band_c512", inputs: 7, outputs: &[7] },
    Member { path: "furiosa_opt_gemma4::ops_dsl_attention::prefill::w384_fp8::swa_prefill_fp8_w384_c512", inputs: 7, outputs: &[7] },
    Member { path: "furiosa_opt_gemma4::ops_dsl_attention::decode::w256_fp8::swa_decode_fp8_w256", inputs: 5, outputs: &[5] },
    Member { path: "furiosa_opt_gemma4::ops_dsl_attention::prompt_blocks::swa_prompt_fp8_4_3_1", inputs: 20, outputs: &[1, 6, 11] },
    Member { path: "furiosa_opt_gemma4::ops_core::full_resident::rsf_int8_fp8kv_t1536_s8", inputs: 66, outputs: &[36, 37, 38, 39, 40, 41] },
    Member { path: "furiosa_opt_gemma4::ops_core::resident::q4::rss_int8_fp8kv_t1536_g12_p999_pair", inputs: 72, outputs: &[18, 54, 55, 56, 57, 58] },
    Member { path: "furiosa_opt_gemma4::ops_core::resident_fs::q4::rfs_int8_fp8kv_t1536_g12_p999_pair", inputs: 69, outputs: &[18, 53, 54, 55] },
    Member { path: "furiosa_opt_gemma4::ops_core::resident::q4::rss_int8_fp8kv_t1024_g12_p941_pair", inputs: 72, outputs: &[18, 54, 55, 56, 57, 58] },
    Member { path: "furiosa_opt_gemma4::ops_core::resident_fs::q4::rfs_int8_fp8kv_t1024_g12_p941_pair", inputs: 69, outputs: &[18, 53, 54, 55] },
];

pub(super) const FAMILIES: &[Family] = &[
    Family { bucket: 1536, name: "t1536_g12_p999_pair", row: "furiosa_opt_gemma4::host::step_image::t1536_g12_p999_pair", slots: 115, ops: &[24, 23, 25, 1, 2, NO_MEMBER, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] },
    Family { bucket: 1024, name: "t1024_g12_p941_pair", row: "furiosa_opt_gemma4::host::step_image::t1024_g12_p941_pair", slots: 107, ops: &[26, 0, 27, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22] },
];
