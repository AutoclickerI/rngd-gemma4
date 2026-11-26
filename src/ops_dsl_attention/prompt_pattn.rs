//! Native E4M3 prompt attention contractions.
//! Q scale = max(amax(row, head)/256, 1e-30); K/V use stored codes.
//! P codes = E4M3(256 * exp(score - max)); denominator sums unrounded f32 exponentials.
//! Attention's 1/sqrt(D) is already folded into the resident query producer.
use furiosa_opt_std::prelude::*;
use super::prompt_blocks::*;

/// Quantize one query group from a full DM view or a round tile.
#[doc(hidden)]
#[macro_export]
macro_rules! prompt_query_codes {
    ($device:ident, $query:expr => $codes:ident, $scale:ident) => {
        let $scale: DmTensor<f32, Chip, C, S, m![G2, Q % 8 / 2, Q % 2]> = $device.main.begin($query)
            .fetch::<m![G2, Q % 8, D / 16], m![D % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![G2, Q % 8, D / 8], m![D % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd, i32::MAX).vector_reinterpret::<f32>()
            .vector_narrow_split::<m![G2, Q % 8, D / 4], m![D % 4]>()
            .vector_intra_slice_reduce::<D, m![G2, Q % 8], m![1 # 4]>(IntraSliceReduceOpF32::Max)
            .vector_fp_div(256f32)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_clip(ClipBinaryOpF32::Max, 1e-30f32).vector_final().transpose::<m![G2, Q % 8 / 2], m![Q % 2 # 8]>().commit_trim::<m![Q % 2]>().commit();
        let $scale: DmTensor<f32, Chip, C, S, m![G2, Q % 8]> = unsafe { $scale.reshape() };
        let sv: VrfTensor<f32, Chip, C, S, m![Q % 8, G2, 1 # 8]> = $device.sub.begin($scale.view())
            .fetch::<m![Q % 8, G2], m![1 # 8]>().collect::<m![Q % 8, G2], m![1 # 8]>().to_vrf();
        let $codes: DmTensor<f8e4m3, Chip, C, S, m![Q % 8, G2, D]> = $device.main.begin($query)
            .fetch::<m![Q % 8, G2, D / 16], m![D % 16]>().fetch_cast::<f32>()
            .collect::<m![Q % 8, G2, D / 8], m![D % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Q % 8, G2, D / 4], m![D % 4]>()
            .vector_fp_div(&sv)
            .vector_widen_concat::<m![Q % 8, G2, D / 8], m![D % 8]>().vector_final()
            .cast::<f8e4m3, m![D % 8 # 32]>().commit_trim::<m![D % 8]>().commit();
    };
}

type Query = DmTensor<bf16, Chip, C, S, m![Q % 8, G2, D]>;
type Codes = DmTensor<f8e4m3, Chip, C, S, m![Q % 8, G2, D]>;
type Scale = DmTensor<f32, Chip, C, S, m![G2, Q % 8]>;

fn query_codes(device: &mut Device, q: &Query) -> (Codes, Scale) {
    crate::prompt_query_codes!(device, q.view() => codes, scale);
    (codes, scale)
}

#[doc(hidden)]
#[macro_export]
macro_rules! pattn_prompt_mid_range_native {
    ($wn:ident, $t:literal, $ms:literal, $ml:literal, $device:ident, $s:ident, $k:ident, $q_trf:ident, $scale:ident, $tm:ident) => {
        let tm_storage: DmTensorView<'_, f32, Chip, C, S, m![Tp, Q % 8]> = unsafe { $tm.view().reshape() };
        // Materialize before relabel: Opt 0.9 cannot reshape a tiled view.
        let tm_range: DmTensor<f32, Chip, C, S, m![Tp = $ml, Q % 8]> = $device.sub
            .begin(tm_storage.tile::<m![Tp], $ml, m![Tp = $ml # 32, Q % 8]>($ms))
            .fetch::<m![Tp = $ml], L>().collect::<m![Tp = $ml], L>()
            .commit_trim::<L>().commit();
        let tm_range: DmTensor<f32, Chip, C, S, m![$wn / 128 = $ml, Q % 8]> = unsafe { tm_range.reshape() };
        let tm_vrf: VrfTensor<f32, Chip, C, S, m![$wn / 128 = $ml, Q % 8]> =
            $device.sub.begin(tm_range.view()).fetch::<m![$wn / 128 = $ml], L>().collect::<m![$wn / 128 = $ml], L>().to_vrf();
        $device
            .main
            .begin($k.view().tile::<m![$wn / 128], $ml, m![$wn / 128 = $ml # $t, $wn % 8, D]>($ms))
            .fetch::<m![$wn / 128 = $ml, $wn % 8, D / 32], m![D % 32]>()

            .switch::<S, m![$wn / 128 = $ml, $wn % 8, $wn / 8 % 16, D / 32]>(SwitchConfig::Broadcast01 { slice1: 16, slice0: 1, time0: 8 })
            .collect::<m![$wn / 128 = $ml, $wn % 8, $wn / 8 % 16, D / 32], m![D % 32]>()
            .contract_outer::<m![$wn / 128 = $ml, $wn % 8, $wn / 8 % 16, D / 64, G2], m![D % 64], _, _, _>(&$q_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![$wn / 128 = $ml, $wn % 8, $wn / 8 % 16, G2]>()
            .contract_lane::<m![$wn / 128 = $ml, $wn % 8, $wn / 8 % 16, G2], L>(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![$wn / 128 = $ml, $wn % 8, $wn / 8 % 16, G2, Q % 8 / 4], m![Q % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &$scale)
            .vector_widen_concat::<m![$wn / 128 = $ml, $wn % 8, $wn / 8 % 16, G2], L>()
            .vector_clip(ClipBinaryOpF32::Min, &tm_vrf)
            .vector_final()
            .commit_trim::<L>()
            .commit_view($s.view_mut().tile::<m![$wn / 128], $ml, m![$wn / 128 = $ml #{!} $t, $wn % 128, G2, Q % 8]>($ms));
    };
}

/// Edge contractions over stored E4M3 keys, with Q rescaling before the mask.
#[doc(hidden)]
#[macro_export]
macro_rules! pattn_prompt_edge_native {
    ($wn:ident, $t:literal, $device:ident, $s:ident, $k:ident, $q_trf:ident, $scale:ident, $cap:ident, $tile:expr) => {
        let cap_vrf: VrfTensor<f32, Chip, C, S, m![$wn % 128, Q % 8]> =
            $device.sub.begin($cap).fetch::<m![$wn % 128], L>().collect::<m![$wn % 128], L>().to_vrf();
        $device
            .main
            .begin($k.view().tile::<m![$wn / 128], 1, m![1 # $t, $wn % 128 % 8, D]>($tile))
            .fetch::<m![$wn % 8, D / 32], m![D % 32]>()

            .switch::<S, m![$wn % 8, $wn / 8 % 16, D / 32]>(SwitchConfig::Broadcast01 { slice1: 16, slice0: 1, time0: 8 })
            .collect::<m![$wn % 8, $wn / 8 % 16, D / 32], m![D % 32]>()
            .contract_outer::<m![$wn % 8, $wn / 8 % 16, D / 64, G2], m![D % 64], _, _, _>(&$q_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![$wn % 8, $wn / 8 % 16, G2]>()
            .contract_lane::<m![$wn % 8, $wn / 8 % 16, G2], L>(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![$wn % 8, $wn / 8 % 16, G2, Q % 8 / 4], m![Q % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &$scale)
            .vector_widen_concat::<m![$wn % 8, $wn / 8 % 16, G2], L>()
            .vector_clip(ClipBinaryOpF32::Min, &cap_vrf)
            .vector_final()
            .commit_trim::<L>()
            .commit_view($s.view_mut().tile::<m![$wn / 128], 1, m![1 #{!} $t, $wn % 128, G2, Q % 8]>($tile));
    };
}

// Apply caps in the order specified by the resident round.
macro_rules! prompt_scores_native {
    (routed_trim, $wn:ident, $t:literal, $device:ident, $s:ident, $k:ident, $q_trf:ident, $scale:ident, $tm:ident, $cap_hi:ident, $cap_lo:ident, $cap_mid:ident) => {
        crate::pattn_prompt_mid_range_native!($wn, $t, 2, 1, $device, $s, $k, $q_trf, $scale, $tm);
        crate::pattn_prompt_edge_native!($wn, $t, $device, $s, $k, $q_trf, $scale, $cap_lo, 0);
        crate::pattn_prompt_edge_native!($wn, $t, $device, $s, $k, $q_trf, $scale, $cap_mid, 1);
        crate::pattn_prompt_edge_native!($wn, $t, $device, $s, $k, $q_trf, $scale, $cap_hi, 3);
    };
    (causal2, $wn:ident, $t:literal, $device:ident, $s:ident, $k:ident, $q_trf:ident, $scale:ident, $tm:ident, $cap_hi:ident, $cap_lo:ident, $cap_mid:ident) => {
        crate::pattn_prompt_mid_range_native!($wn, $t, 0, 1, $device, $s, $k, $q_trf, $scale, $tm);
        crate::pattn_prompt_edge_native!($wn, $t, $device, $s, $k, $q_trf, $scale, $cap_hi, $t - 1);
    };
    (trim3, $wn:ident, $t:literal, $device:ident, $s:ident, $k:ident, $q_trf:ident, $scale:ident, $tm:ident, $cap_hi:ident, $cap_lo:ident, $cap_mid:ident) => {
        crate::pattn_prompt_mid_range_native!($wn, $t, 1, 1, $device, $s, $k, $q_trf, $scale, $tm);
        crate::pattn_prompt_edge_native!($wn, $t, $device, $s, $k, $q_trf, $scale, $cap_hi, $t - 1);
        crate::pattn_prompt_edge_native!($wn, $t, $device, $s, $k, $q_trf, $scale, $cap_lo, 0);
    };
    (single, $wn:ident, $t:literal, $device:ident, $s:ident, $k:ident, $q_trf:ident, $scale:ident, $tm:ident, $cap_hi:ident, $cap_lo:ident, $cap_mid:ident) => {
        crate::pattn_prompt_edge_native!($wn, $t, $device, $s, $k, $q_trf, $scale, $cap_hi, $t - 1);
    };
}

/// Normalize the clipped scores and contract the stored E4M3 values.
#[doc(hidden)]
#[macro_export]
macro_rules! prompt_weighted_values {
    ($device:ident, $wn:ident, $s:ident, $v:ident) => {{
        // 2. Max per (head, query) over the window.
        let max: DmTensor<f32, Chip, C, S, m![G2, Q % 8]> = $device
            .main
            .begin($s.view())
            .fetch::<m![G2, $wn], L>()
            .collect::<m![G2, $wn], L>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![G2, $wn, Q % 8 / 4], m![Q % 4]>()
            .vector_intra_slice_reduce::<$wn, m![G2, Q % 8 / 4], m![Q % 4]>(IntraSliceReduceOpF32::Max)
            .vector_widen_concat::<m![G2], L>()
            .vector_final()
            .commit_trim::<L>()
            .commit();
        let max_vrf: VrfTensor<f32, Chip, C, S, m![G2, Q % 8]> = $device.sub.begin(max.view()).fetch::<m![G2], L>().collect::<m![G2], L>().to_vrf();

        // 3. P = E4M3(256 exp(score - max)), transposed 8 positions at a time for the TRF.
        let p: DmTensor<f8e4m3, Chip, C, S, m![Q % 8, G2, $wn]> = $device
            .main
            .begin($s.view())
            .fetch::<m![G2, $wn], L>()
            .collect::<m![G2, $wn], L>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![G2, $wn, Q % 8 / 4], m![Q % 4]>()
            .vector_fp_binary(FpBinaryOp::SubF, &max_vrf)
            .vector_fp_unary(FpUnaryOp::Exp)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 256f32)
            .vector_widen_concat::<m![G2, $wn], L>()
            .vector_final()
            .cast::<f8e4m3, m![Q % 8 # 32]>()
            .transpose::<m![G2, $wn / 8, Q % 8], m![$wn % 8 # 32]>()
            .commit_trim::<m![$wn % 8]>()
            .commit();

        // Sum unrounded exponentials in position order on Sub while Main runs PV.
        let sum: DmTensor<f32, Chip, C, S, m![G2, Q % 8]> = $device
            .sub
            .begin($s.view())
            .fetch::<m![G2, $wn], L>()
            .collect::<m![G2, $wn], L>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![G2, $wn, Q % 8 / 4], m![Q % 4]>()
            .vector_fp_binary(FpBinaryOp::SubF, &max_vrf)
            .vector_fp_unary(FpUnaryOp::Exp)
            .vector_intra_slice_reduce::<$wn, m![G2, Q % 8 / 4], m![Q % 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_concat::<m![G2], L>()
            .vector_final()
            .commit_trim::<L>()
            .commit();

        // 4. PV: P stationary (lane = query, element = [head, position]); the values all-gathered as the keys were.
        let p_trf: TrfTensor<f8e4m3, Chip, C, S, L, m![G2, $wn]> =
            $device.sub.begin(p.view()).fetch::<m![Q % 8, G2, $wn / 32], m![$wn % 32]>().collect::<m![Q % 8, G2, $wn / 32], m![$wn % 32]>().to_trf();
        let o: DmTensor<f32, Chip, C, S, m![D / 32, G2, Q % 8, D % 32]> = $device.main.begin($v.view()).fetch::<m![D / 32, $wn / 128, $wn % 8], m![D % 32]>()
            .switch::<S, m![D / 32, $wn / 128, $wn / 8 % 16, $wn % 8]>(SwitchConfig::Broadcast01 { slice1: 16, slice0: 1, time0: 8 })
            .collect::<m![D / 32, $wn / 128, $wn / 8 % 16, $wn % 8], m![D % 32]>()
            .contract_outer::<m![D / 32, $wn, G2], m![D % 32], _, _, _>(&p_trf)
            .contract_packet::<m![D % 32]>()
            .contract_time::<m![D / 32, G2]>()
            .contract_lane::<m![D / 32, G2, Q % 8, D % 32 / 8], m![D % 8]>(LaneMode::Sequential)
            .commit_trim::<m![D % 8]>()
            .commit();

        // 5. Normalize, round.
        let sum_vrf: VrfTensor<f32, Chip, C, S, m![G2, Q % 8]> = $device.sub.begin(sum.view()).fetch::<m![G2], L>().collect::<m![G2], L>().to_vrf();
        $device
            .main
            .begin(o.view())
            .fetch::<m![G2, Q % 8, D / 8], m![D % 8]>()
            .collect::<m![G2, Q % 8, D / 8], m![D % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![G2, Q % 8, D / 4], m![D % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1f32 / 256f32)
            .vector_fp_div(&sum_vrf)
            .vector_widen_concat::<m![G2, Q % 8, D / 8], m![D % 8]>()
            .vector_final()
            .cast::<bf16, m![D % 8 # 16]>()
            .commit_trim::<m![D % 8]>()
            .commit()
    }};
}

macro_rules! native_round {
    ($fname:ident, $wn:ident, $t:literal, $form:ident) => {
        native_round!(masked, $fname, $wn, $t, $form);
    };
    ($masks:ident, $fname:ident, $wn:ident, $t:literal, $form:ident) => {
        pub(crate) fn $fname(
            device: &mut Device,
            loaded: native_round!(@loaded $masks, $wn),
            cap_hi: &Cap,
        ) -> Out {
            native_round!(@bind $masks, loaded => q_dm, kg, vg, _tm, _cap_lo);
            let (q8, qs) = query_codes(device, &q_dm);
            let qs_vrf: VrfTensor<f32, Chip, C, S, m![G2, Q % 8]> = device.sub.begin(qs.view()).fetch::<m![G2], L>().collect::<m![G2], L>().to_vrf();
            // The queries: lane = query, element = [head, d].
            let q_trf: TrfTensor<f8e4m3, Chip, C, S, L, m![G2, D]> = device
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
            let cap_hi: DmTensorView<'_, f32, Chip, C, S, m![$wn % 128, Q % 8]> = unsafe { cap_hi.view().reshape() };
            native_round!(@lower $masks, $wn, _cap_lo => _cap_lo, _cap_mid);
            prompt_scores_native!($form, $wn, $t, device, s, k, q_trf, qs_vrf, _tm, cap_hi, _cap_lo, _cap_mid);

            crate::prompt_weighted_values!(device, $wn, s, v)
        }
    };
    (@loaded masked, $wn:ident) => { (
        DmTensor<bf16, Chip, C, S, m![Q % 8, G2, D]>,
        DmTensor<f8e4m3, Chip, C, SK<$wn>, m![$wn / 128, $wn % 8, D]>,
        DmTensor<f8e4m3, Chip, C, SK<$wn>, m![$wn / 128, $wn % 8, D]>,
        DmTensor<f32, Chip, C, S, m![$wn / 128 # 32, Q % 8]>,
        LowerCaps,
    ) };
    (@loaded single, $wn:ident) => { (
        DmTensor<bf16, Chip, C, S, m![Q % 8, G2, D]>,
        DmTensor<f8e4m3, Chip, C, SK<$wn>, m![$wn / 128, $wn % 8, D]>,
        DmTensor<f8e4m3, Chip, C, SK<$wn>, m![$wn / 128, $wn % 8, D]>,
    ) };
    (@bind masked, $loaded:ident => $q:ident, $k:ident, $v:ident, $tm:ident, $lo:ident) => {
        let ($q,$k,$v,$tm,$lo)=$loaded;
    };
    (@bind single, $loaded:ident => $q:ident, $k:ident, $v:ident, $tm:ident, $lo:ident) => {
        let ($q,$k,$v)=$loaded;
    };
    (@lower masked, $wn:ident, $input:ident => $lo:ident, $mid:ident) => {
        let lower: DmTensorView<'_,f32,Chip,C,S,m![Edge,$wn % 128,Q % 8]> = unsafe { $input.view().reshape() };
        let $mid=lower.tile::<m![Edge],1,m![1 # 2,$wn % 128,Q % 8]>(1);
        let $lo=lower.tile::<m![Edge],1,m![1 # 2,$wn % 128,Q % 8]>(0);
    };
    (@lower single, $wn:ident, $input:ident => $lo:ident, $mid:ident) => {};

}

native_round!(round_w1, W1, 1, single);
#[cfg(feature = "compile-carried")]
native_round!(single, round_w1_nomasks, W1, 1, single);
native_round!(round_w2, W2, 2, causal2);
native_round!(round_w3b, W3, 3, trim3);
native_round!(round_w4r, W4, 4, routed_trim);
