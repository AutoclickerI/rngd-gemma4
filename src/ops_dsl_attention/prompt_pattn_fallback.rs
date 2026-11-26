#[macro_export]
macro_rules! pattn_fallback_mid {
    ($wn:ident, $t:literal, none, $device:ident, $s:ident, $k:ident, $q:ident, $scale:ident, $tm:ident) => {};
    ($wn:ident, $t:literal, ($ms:literal, $ml:literal), $device:ident, $s:ident, $k:ident, $q:ident, $scale:ident, $tm:ident) => {
        $crate::pattn_prompt_mid_range_native!($wn, $t, $ms, $ml, $device, $s, $k, $q, $scale, $tm);
    };
    ($wn:ident, 3, all, $device:ident, $s:ident, $k:ident, $q:ident, $scale:ident, $tm:ident) => {
        $crate::pattn_prompt_mid_range_native!($wn, 3, 0, 2, $device, $s, $k, $q, $scale, $tm);
    };
}
#[macro_export]
macro_rules! pattn_prefill_round {
    ($band:tt, $wn:ident, $nr:literal, $t:tt, $mid:tt, $device:ident, $r:literal, $q_dm:ident,
     $kg:ident, $vg:ident, $tm:ident, $cap_lo:ident, $cap_hi:ident, $out:ident) => {
        let kg: DmTensor<f8e4m3, Chip, C, SK<$wn>, m![$wn / 128, $wn % 8, D]> = unsafe { $kg.reshape() };
        let vg: DmTensor<f8e4m3, Chip, C, SK<$wn>, m![$wn / 128, $wn % 8, D]> = unsafe { $vg.reshape() };
        $crate::prompt_query_codes!(
            $device, $q_dm.view().tile::<m![Q / 512], 1, m![1 # $nr, Q % 8, G2, D]>($r) => q8, scale
        );
        let qs_vrf: VrfTensor<f32, Chip, C, S, m![G2, Q % 8]> = $device.sub.begin(scale.view()).fetch::<m![G2], L>().collect::<m![G2], L>().to_vrf();
        // The queries: lane = query, element = [head, d].
        let q_trf: TrfTensor<f8e4m3, Chip, C, S, L, m![G2, D]> = $device
            .sub
            .begin(q8.view())
            .fetch::<m![Q % 8, G2, D / 32], m![D % 32]>()
            .collect::<m![Q % 8, G2, D / 32], m![D % 32]>()
            .to_trf();
        // K/V stay in their stored E4M3 format throughout the contractions.
        let k = &kg;
        let v = &vg;

        // 1. Scores [position, head] x [query] in f32, clipped.
        let mut s: DmTensor<f32, Chip, C, S, m![$wn, G2, Q % 8]> = DmTensor::new();
        let cap_hi: DmTensorView<'_, f32, Chip, C, S, m![$wn % 128, Q % 8]> = unsafe { $cap_hi.view().reshape() };
        $crate::pattn_fallback_mid!($wn, $t, $mid, $device, s, k, q_trf, qs_vrf, $tm);
        $crate::pattn_prompt_edge_native!($wn, $t, $device, s, k, q_trf, qs_vrf, cap_hi, $t - 1);
        $crate::dsl_band_only!($band, {
            let cap_lo: DmTensorView<'_, f32, Chip, C, S, m![$wn % 128, Q % 8]> = unsafe { $cap_lo.view().reshape() };
            $crate::pattn_prompt_edge_native!($wn, $t, $device, s, k, q_trf, qs_vrf, cap_lo, 0);
        });

        let o_bf: DmTensor<bf16, Chip, C, S, m![G2, Q % 8, D]> =
            $crate::prompt_weighted_values!($device, $wn, s, v);
        o_bf.view().to_hbm_view(&mut $device.tdma, $out.view_mut().tile::<m![Q / 512], 1, m![1 #{!} $nr, Q % 512, N, G2, D]>($r));
    };
}
