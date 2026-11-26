//! Cohort-stable standalone Q4 segments for every served Q4 layer.
pub mod q4 {
use super::super::*;
use super::super::q4serve::Q4FfnWeights;
#[device(chip = 1)]
pub fn mid_ss_q4_t1024(
    device: &mut Device,
    out_s: &HbmTensor<bf16, Chip, m![Rq, Hqs]>,
    rows_s: &HbmTensor<i32, Chip, m![Tok1024, Hh]>,
    stage_s: &HbmTensor<i32, Chip, m![Tok1024]>,
    publish_s: &HbmTensor<i32, Chip, m![Tok1024]>,
    rope_s: &HbmTensor<i32, Chip, m![Tok1024]>,
    x: &mut HbmTensor<bf16, Chip, m![Tok1024, H]>,
    consts: Int8Consts<'_>,
    o: crate::ops_tokenwise_int8::SlidingOStaticF8<'_>,
    ffn: Q4FfnWeights<'_>,
    layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
    next: crate::ops_tokenwise_int8::StaticF8SlidingQkvWeights<'_>,
    k_l: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
    v_l: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
    qin_s: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
    k_n: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
    v_n: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
) {
    super::super::q4serve::mid_ss_q4_port_t1024_body::<ffn_fast::GlOf<Tok1024>>(device, out_s, rows_s, stage_s, publish_s, rope_s, x, consts, o, ffn, layer_scalar, next, k_l, v_l, qin_s, k_n, v_n);
}

#[device(chip = 1)]
pub fn mid_fs_q4_t1024(
    device: &mut Device,
    out_f: &HbmTensor<bf16, Chip, m![Rq, Hqf]>,
    rows_f: &HbmTensor<i32, Chip, m![Tok1024, Hh]>,
    rope_s: &HbmTensor<i32, Chip, m![Tok1024]>,
    rows_s: &HbmTensor<i32, Chip, m![Tok1024, Hh]>,
    stage_s: &HbmTensor<i32, Chip, m![Tok1024]>,
    x: &mut HbmTensor<bf16, Chip, m![Tok1024, H]>,
    consts: Int8Consts<'_>,
    o: FullOInt8<'_>,
    ffn: Q4FfnWeights<'_>,
    layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
    next: Int8SlidingQkvWeights<'_>,
    qin_s: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
    k_n: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
    v_n: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
) {
    super::super::q4serve::mid_fs_q4_t1024_body(device, out_f, rows_f, rope_s, rows_s, stage_s, x, consts, o, ffn, layer_scalar, next, qin_s, k_n, v_n);
}
}
