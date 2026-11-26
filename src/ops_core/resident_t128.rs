use crate::axes::Tok128;
// Tok128 Q4 decode-only resident programs. Past KV gathers precede the exact class512 body;
// refresh the newest entry after QKV before attending. Routing requires zero prompt rows.
macro_rules! drain128_ss_entries {
    ($name:ident, $gs:ident, $($g:ident),+) => {
        #[derive(DeviceSend)]
        pub struct $gs<'a> { $(pub $g: wtop::Group<'a>,)+ }
        #[device(chip = 1)]
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
    device: &mut Device,
            groups: $gs<'_>,
    out_s: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
    rows_s: &HbmTensor<i32, Chip, m![Tok128, Hh]>,
    stage_s: &HbmTensor<i32, Chip, m![Tok128]>,
    publish_s: &HbmTensor<i32, Chip, m![Tok128]>,
    rope_s: &HbmTensor<i32, Chip, m![Tok128]>,
    x: &mut HbmTensor<bf16, Chip, m![Tok128, H]>,
    consts: Int8Consts<'_>,
    o: crate::ops_tokenwise_int8::SlidingOStaticF8<'_>,
    ffn: super::q4serve::Q4FfnWeights<'_>,
    layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
    next: crate::ops_tokenwise_int8::StaticF8SlidingQkvWeights<'_>,
    k_l: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
    v_l: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
    qin_s: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
    k_n: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
    v_n: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,

        ) {
            let $gs { $($g),+ } = groups;
            $(let mut $g = (wtop::gather_group(device, $g.kv_off, $g.cap, k_n, v_n), $g);)+
            super::smallbatch::mid_ss_q4_class512_t128_body(device, &*out_s, rows_s, stage_s, publish_s, rope_s, x, consts, o, ffn, layer_scalar, next, k_l, v_l, qin_s, k_n, v_n);
            $(
                wtop::refresh_newest(device, $g.1.kv_off, k_n, v_n, &mut $g.0);
                wtop::attend_gathered(device, qin_s, $g.1.rows, &$g.0, out_s);
            )+
        }
    };
}
drain128_ss_entries!(rss_q4_class512_t128_g1, Drain128SsG1, g0);
drain128_ss_entries!(rss_q4_class512_t128_g2, Drain128SsG2, g0, g1);
drain128_ss_entries!(rss_q4_class512_t128_g4, Drain128SsG4, g0, g1, g2, g3);
drain128_ss_entries!(rss_q4_class512_t128_g8, Drain128SsG8, g0, g1, g2, g3, g4, g5, g6, g7);
drain128_ss_entries!(rss_q4_class512_t128_g16, Drain128SsG16, g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15);
