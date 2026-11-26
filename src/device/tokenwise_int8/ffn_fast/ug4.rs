//! INT4 Gate/Up packing and contraction with FP8 Down.

    use super::*;
    axes![Hb = 1920, G240 = 240, J8 = 8, P2 = 2];
    use super::super::rotate::{Ka, Kq, RotK, RotPad};

    // `wug_scale`'s rows: 0 the Up rows' scales, 1 the column term `C'`.
    axes![Xc2 = 2];

    /// Activation clip `f32(0.8)`, inverse grid maximum `f32(1 / 7.5)` and half the scale floor.
    const MID4_ACT_CLIP: f32 = 0.8f64 as f32;
    const MID4_INV_7_5: f32 = 1.0 / 7.5;
    const MID4_HALF_FLOOR: f32 = 1e-30 * 0.5;
    /// The nibble rounding: `z + 1.5 * 2^23` lies in `[2^23, 2^24)` for `|z| < 2^22`, where the f32 grid is the
    /// integers, so the add rounds `z` to the nearest integer (ties to even).
    const NIB_MAGIC: f32 = 12_582_912.0;
    const NIB_MAGIC_NEG_I32: i32 = -12_582_912;
    /// `R2`'s offset: `2 sum_k c + R2_OFF >= 0` for `c >= -8` over 3,840 codes (16 per code).
    pub(crate) const R2_OFF: i32 = 61_440;
    /// `R2_OFF`'s share per code (16.0) as a const item: CTFE folds it. Written inline in a device body, the signed
    /// `/` carried rustc's MIR overflow check `(rhs == -1) & (lhs == MIN)` (`rustc_mir_build` `build_binary_op`), a
    /// scalar `BitAnd` unsupported by the Opt 0.9 MIR translator.
    const R2_OFF_PER_CODE: f32 = (R2_OFF / 3840) as f32;

    /// The Up rows' scales or `C'` of all four passes with zeros in the padded ring pair.
    type UpRows<URING, UXS, UYS> = DmTensor<f32, Chip, UCl, USl<URING, UXS, UYS>, m![Lu / 16 % 4, Lu / 8 % 2, Lu % 8]>;
    /// A per-token row value (`sx` or `R2`) in every Up/Gate slice, in token groups of `QS`.
    type TokAll<D, T, URING, UXS, UYS, const QS: usize> = DmTensor<D, Chip, UCl, USl<URING, UXS, UYS>, m![T / { QS }, T % { QS }]>;
    /// `gemm::XRing` over all tokens with the token groups explicit: slice `(ring, T % 64, K / CH % 4)` holds that
    /// chunk of the tokens at ring position `T % 64`, group-major.
    type XR4<T, URING, K, const CH: usize, const QS: usize> =
        DmTensor<i8, Chip, UCl, m![{ URING }, T % 64, K / { CH } % 4], m![T / { QS }, T % { QS } / 64, K % { CH }]>;
    /// One pass's Up/Gate outputs, written one token group at a time.
    type YQ<T, URING, UXS, UYS, const QS: usize> =
        DmTensor<bf16, Chip, UCl, USl<URING, UXS, UYS>, m![T / { QS }, T % { QS } / 64, T % 64, Lu / 8 % 2, Lu % 8]>;

    /// 2b's half step `s = max(max_k f32(f32(0.8 |y_k|) * f32(1 / 7.5)) / 2, 1e-30 / 2)` (`rotate::Mid4`).
    fn row_scale_mid4<T: AxisName, const HALF: usize>(device: &mut Device, y: &Row<bf16, T, HALF, RotPad>) -> Row<f32, T, HALF, m![1 # 8]> {
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
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), MID4_ACT_CLIP)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), MID4_INV_7_5)
            .vector_intra_slice_reduce::<Kq, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
            .vector_fp_div(2f32)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_clip(ClipBinaryOpF32::Max, MID4_HALF_FLOOR)
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit()
    }

    /// `v = f32(f32(y / s) - 1)` over the live rows (2b's first code pass).
    fn mid4_offset<T: AxisName, const HALF: usize>(device: &mut Device, y: &Row<bf16, T, HALF, RotPad>, s: &Row<f32, T, HALF, m![1 # 8]>) -> Row<f32, T, HALF, RotK> {
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
            .vector_fp_binary(FpBinaryOp::AddF, -1f32)
            .vector_widen_concat::<m![Ob / 32, Ra, Ob % 32 / 8], m![Ob % 8]>()
            .vector_final()
            .commit_trim::<m![Ob % 8]>()
            .commit()
    }

    /// The nibbles `clamp(rne(v / 2), -8, 7)`, i.e. `(q - 1) / 2` of 2b's odd code `q` (same pass shape as
    /// `mid4_codes`: one multiply, one add, the conversion and the clip).
    fn nib_codes<T: AxisName, const HALF: usize>(device: &mut Device, v: &Row<f32, T, HALF, RotK>) -> Row<i8, T, HALF, RotK> {
        device
            .main
            .begin(v.view())
            .fetch::<m![Ob / 32, Ra, Ob % 32 / 8], m![Ob % 8]>()
            .collect::<m![Ob / 32, Ra, Ob % 32 / 8], m![Ob % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Ob / 32, Ra, Ob % 32 / 4], m![Ob % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.5f32)
            .vector_fp_binary(FpBinaryOp::AddF, NIB_MAGIC)
            .vector_widen_concat::<m![Ob / 32, Ra, Ob % 32 / 8], m![Ob % 8]>()
            .vector_fp_to_fxp(31)
            .vector_clip(ClipBinaryOpI32::AddFxp, NIB_MAGIC_NEG_I32)
            .vector_clip(ClipBinaryOpI32::Min, 7)
            .vector_clip(ClipBinaryOpI32::Max, -8)
            .vector_final()
            .cast::<i8, m![Ob % 8 # 32]>()
            .commit_trim::<m![Ob % 8]>()
            .commit()
    }

    /// `R2 = 2 sum_k c_k + R2_OFF` of one nibble row as f32 in `[1 # 8]` (a row scale's shape): each element
    /// `2 c + 16` lies in [0, 30], so every partial sum is an integer below 2^17 and the f32 sum is exact.
    fn row_sum2<T: AxisName, const HALF: usize>(device: &mut Device, c: &Row<i8, T, HALF, RotK>) -> Row<f32, T, HALF, m![1 # 8]> {
        let c: DmTensorView<'_, i8, Chip, TCl<T, HALF>, TSl<T, HALF>, m![Kh]> = unsafe { c.view().reshape() };
        device
            .main
            .begin(c)
            .fetch::<m![Kh / 16], m![Kh % 16]>()
            .fetch_cast::<i32>()
            .collect::<m![Kh / 8], m![Kh % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp_to_fp(31)
            .vector_narrow_split::<m![Kh / 4], m![Kh % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 2f32)
            .vector_fp_binary(FpBinaryOp::AddF, R2_OFF_PER_CODE)
            .vector_intra_slice_reduce::<Kh, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit()
    }

    /// A nibble row as `[G240, P2, J8]` (16-code groups, `P2 = 0` the low half of each byte, `P2 = 1` the high half):
    /// Reshape the owned tensor so the two nibble halves can be tiled on the `P2` axis.
    type NibGroups<T, const HALF: usize> = Row<i8, T, HALF, m![G240, P2, J8]>;

    /// Compute `c & 15` as `c + min(max(-16 c, 0), 16)` with the Clip ALUs.
    fn low_nibbles<T: AxisName, const HALF: usize>(device: &mut Device, c: &NibGroups<T, HALF>) -> Row<i8, T, HALF, m![G240, J8]> {
        device
            .main
            .begin(c.view().tile::<m![P2], 1, m![G240, P2 = 1 # 2, J8]>(0))
            .fetch::<m![G240], m![J8]>()
            .fetch_cast::<i32>()
            .collect::<m![G240], m![J8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_stash()
            .vector_fxp_to_fp(31)
            .vector_narrow_split::<m![G240, J8 / 4], m![J8 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), -16f32)
            .vector_widen_concat::<m![G240], m![J8]>()
            .vector_fp_to_fxp(31)
            .vector_clip(ClipBinaryOpI32::Max, 0)
            .vector_clip(ClipBinaryOpI32::Min, 16)
            .vector_clip(ClipBinaryOpI32::AddFxp, Stash)
            .vector_final()
            .cast::<i8, m![J8 # 32]>()
            .commit_trim::<m![J8]>()
            .commit()
    }

    /// `16 c` (bytes in [-128, 112]) of every group's last eight codes (`[G240, P2 = 1, J8]`), in the Fp cluster
    /// (`16 c` and the rounding add are exact, so `nib_codes`' magic-number conversion returns the integer).
    fn high_x16<T: AxisName, const HALF: usize>(device: &mut Device, c: &NibGroups<T, HALF>) -> Row<i8, T, HALF, m![G240, J8]> {
        device
            .main
            .begin(c.view().tile::<m![P2], 1, m![G240, P2 = 1 # 2, J8]>(1))
            .fetch::<m![G240], m![J8]>()
            .fetch_cast::<i32>()
            .collect::<m![G240], m![J8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp_to_fp(31)
            .vector_narrow_split::<m![G240, J8 / 4], m![J8 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 16f32)
            .vector_fp_binary(FpBinaryOp::AddF, NIB_MAGIC)
            .vector_widen_concat::<m![G240], m![J8]>()
            .vector_fp_to_fxp(31)
            .vector_clip(ClipBinaryOpI32::AddFxp, NIB_MAGIC_NEG_I32)
            .vector_final()
            .cast::<i8, m![J8 # 32]>()
            .commit_trim::<m![J8]>()
            .commit()
    }

    /// `ffn4::pack` of one nibble row into pass `PI`'s tile of the packed pair tensor: byte `(g, j) = (c[g, 0, j] &
    /// 15) | (c[g, 1, j] << 4)`, formed as `lo + hi` (`low_nibbles` in [0, 15] plus `high_x16`, whose low four bits
    /// are zero, so the add equals the OR and lies in [-128, 127]) by a Clip-cluster i32 add zip of the two halves
    /// (`ffn::add_partials` uses whole tensors). `ug4_tok128_matches_mid4` checks
    /// bit equality with `model4::pack16`.
    pub(crate) fn pack_into<T: AxisName, const HALF: usize, const PS: usize, const P: usize, const PI: usize>(
        device: &mut Device,
        c: Row<i8, T, HALF, RotK>,
        pk: &mut Row<i8, T, HALF, m![T / { PS }, G240, J8]>,
    ) {
        let c: NibGroups<T, HALF> = unsafe { c.reshape() };
        let lo = low_nibbles::<T, HALF>(device, &c);
        let hi = high_x16::<T, HALF>(device, &c);
        device
            .main
            .begin_interleaved::<Pi, _, _, _, _, _>(lo.view(), hi.view())
            .fetch::<m![G240 / 16, Pi, G240 % 16], m![J8]>()
            .fetch_cast::<i32>()
            .collect::<m![G240 / 16, Pi, G240 % 16], m![J8]>()
            .vector_init()
            .vector_intra_slice_unzip::<Pi, m![G240]>()
            .vector_clip_zip(ClipBinaryOpI32::AddFxp)
            .vector_final()
            .cast::<i8, m![J8 # 32]>()
            .commit_trim::<m![J8]>()
            .commit_view(pk.view_mut().tile::<m![T / { PS }], 1, m![T / { PS } = 1 #{!} { P }, G240, J8]>(PI));
    }

    /// `norm_rotate_quantize_pass_dm_j1`'s norm and one-term rotation, unchanged.
    fn rot_rows<T: AxisName, const HALF: usize>(
        device: &mut Device,
        xd: &Row<bf16, T, HALF, m![Kh]>,
        g: &Row<bf16, T, HALF, m![Kh]>,
        dc: &DmConsts<T, HALF>,
    ) -> Row<bf16, T, HALF, RotPad> {
        let c = rotate2::trf_consts::<T, HALF>(device, dc);
        let h = rotate2::normalize_pair::<T, HALF>(device, xd, g);
        let h: Row<bf16, T, HALF, m![Ka, Kb]> = unsafe { h.reshape() };
        rotate2::rotate_j1::<T, HALF>(device, &h, &c, rotate::INV_SQRT_3840)
    }

    /// The rotated row's 2b half step and nibbles.
    fn nib_rows<T: AxisName, const HALF: usize>(
        device: &mut Device,
        xd: &Row<bf16, T, HALF, m![Kh]>,
        g: &Row<bf16, T, HALF, m![Kh]>,
        dc: &DmConsts<T, HALF>,
    ) -> (Row<f32, T, HALF, m![1 # 8]>, Row<i8, T, HALF, RotK>) {
        let y = rot_rows::<T, HALF>(device, xd, g, dc);
        let s = row_scale_mid4::<T, HALF>(device, &y);
        let v = mid4_offset::<T, HALF>(device, &y, &s);
        (s, nib_codes::<T, HALF>(device, &v))
    }

    /// Pass `PI`'s row values (`[1 # 8]`, non-negative) into `dst [T]` (`rotate::store_scales`).
    fn store_rows<T: AxisName, const HALF: usize, const PS: usize, const P: usize, const PI: usize>(
        device: &mut Device,
        s: Row<f32, T, HALF, m![1 # 8]>,
        onehot: &Onehot<T, HALF>,
        dst: &mut HbmTensor<f32, Chip, m![T]>,
    ) {
        rotate::store_scales::<T, HALF, m![T / { PS } = 1 #{!} { P }, T % { PS }]>(
            device,
            s,
            onehot,
            dst.view_mut().tile::<m![T / { PS }], 1, m![T / { PS } = 1 #{!} { P }, T % { PS }]>(PI),
        );
    }

    /// One pre-FFN pass, two-flit form: norm and rotation, 2b's half step, the nibbles, `R2`, the packed row into
    /// pass `PI`'s tile of `pk` (stored after the last pass), the scale and `R2` stores.
    #[allow(clippy::too_many_arguments)]
    fn pre_pass_pk<T: AxisName, const HALF: usize, const PS: usize, const P: usize, const PI: usize>(
        device: &mut Device,
        xd: &Row<bf16, T, HALF, m![Kh]>,
        g: &Row<bf16, T, HALF, m![Kh]>,
        dc: &DmConsts<T, HALF>,
        onehot: &Onehot<T, HALF>,
        pk: &mut Row<i8, T, HALF, m![T / { PS }, G240, J8]>,
        scales: &mut HbmTensor<f32, Chip, m![T]>,
        r2s: &mut HbmTensor<f32, Chip, m![T]>,
    ) {
        let (s, c) = nib_rows::<T, HALF>(device, xd, g, dc);
        let r2 = row_sum2::<T, HALF>(device, &c);
        pack_into::<T, HALF, PS, P, PI>(device, c, pk);
        store_rows::<T, HALF, PS, P, PI>(device, s, onehot, scales);
        store_rows::<T, HALF, PS, P, PI>(device, r2, onehot, r2s);
    }

    /// `row_scales_ag`'s all-gather of a per-token f32 value `[T]` into every Up/Gate slice's DM, kept in DM in
    /// token groups (`[T / QS, T % QS]`; each Up/Gate sub-pass fetches its group into the VRF).
    fn rows_ag_f32<T: AxisName, URING: M, UXS: M, UYS: M, const QS: usize>(
        device: &mut Device,
        sx: &HbmTensor<f32, Chip, m![T]>,
    ) -> TokAll<f32, T, URING, UXS, UYS, QS> {
        let p: DmTensor<f32, Chip, m![Dummy2], m![Rc, T / 64 % 2, T % 64 / 8], m![T / 128, T % 8]> = sx.to_dm(&mut device.tdma);
        let p: DmTensor<f32, Chip, UCl, m![Rc, T / 64 % 2, T % 64 / 8], m![T / 128, T % 8]> = unsafe { p.reshape() };
        let g: DmTensor<f32, Chip, UCl, m![Rc, X16s], m![T / 128, T / 64 % 2, T % 64 / 8, T % 8]> = device
            .main
            .begin(p.view())
            .fetch::<m![T / 128], m![T % 8]>()
            .switch::<m![Rc, X16s], m![T / 128, T / 64 % 2, T % 64 / 8]>(SwitchConfig::Broadcast1 { slice1: 16, slice0: 1 })
            .collect::<m![T / 128, T / 64 % 2, T % 64 / 8], m![T % 8]>()
            .commit_trim::<m![T % 8]>()
            .commit();
        unsafe { g.reshape() }
    }

    /// `rows_ag_f32` of `R2` (integers below 2^17, exact in f32), converted to i32 in the gather pass (the
    /// Up/Gate epilogue adds it in the fixed-point stage).
    fn rows_ag_i32<T: AxisName, URING: M, UXS: M, UYS: M, const QS: usize>(
        device: &mut Device,
        r2: &HbmTensor<f32, Chip, m![T]>,
    ) -> TokAll<i32, T, URING, UXS, UYS, QS> {
        let p: DmTensor<f32, Chip, m![Dummy2], m![Rc, T / 64 % 2, T % 64 / 8], m![T / 128, T % 8]> = r2.to_dm(&mut device.tdma);
        let p: DmTensor<f32, Chip, UCl, m![Rc, T / 64 % 2, T % 64 / 8], m![T / 128, T % 8]> = unsafe { p.reshape() };
        let g: DmTensor<i32, Chip, UCl, m![Rc, X16s], m![T / 128, T / 64 % 2, T % 64 / 8, T % 8]> = device
            .main
            .begin(p.view())
            .fetch::<m![T / 128], m![T % 8]>()
            .switch::<m![Rc, X16s], m![T / 128, T / 64 % 2, T % 64 / 8]>(SwitchConfig::Broadcast1 { slice1: 16, slice0: 1 })
            .collect::<m![T / 128, T / 64 % 2, T % 64 / 8], m![T % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fp_to_fxp(31)
            .vector_final()
            .commit_trim::<m![T % 8]>()
            .commit();
        unsafe { g.reshape() }
    }

    /// Token group `q` of a per-token f32 value in the VRF.
    fn vrf_q_f32<T: AxisName, URING: M, UXS: M, UYS: M, const QS: usize, const NQ: usize>(
        device: &mut Device,
        d: &TokAll<f32, T, URING, UXS, UYS, QS>,
        q: usize,
    ) -> VrfTensor<f32, Chip, UCl, USl<URING, UXS, UYS>, m![T % { QS }]> {
        device
            .sub
            .begin(d.view().tile::<m![T / { QS }], 1, m![T / { QS } = 1 # { NQ }, T % { QS }]>(crate::device::tokenwise::common::at::<NQ>(q)))
            .fetch::<m![T % { QS } / 8], m![T % 8]>()
            .collect::<m![T % { QS } / 8], m![T % 8]>()
            .to_vrf()
    }

    /// Token group `q` of `R2` (i32) in the VRF.
    fn vrf_q_i32<T: AxisName, URING: M, UXS: M, UYS: M, const QS: usize, const NQ: usize>(
        device: &mut Device,
        d: &TokAll<i32, T, URING, UXS, UYS, QS>,
        q: usize,
    ) -> VrfTensor<i32, Chip, UCl, USl<URING, UXS, UYS>, m![T % { QS }]> {
        device
            .sub
            .begin(d.view().tile::<m![T / { QS }], 1, m![T / { QS } = 1 # { NQ }, T % { QS }]>(crate::device::tokenwise::common::at::<NQ>(q)))
            .fetch::<m![T % { QS } / 8], m![T % 8]>()
            .collect::<m![T % { QS } / 8], m![T % 8]>()
            .to_vrf()
    }

    /// Row `i` of `wug_scale [Xc2, Lu]` (0: the Up rows' scales, 1: `C'`) for all four passes with zeros in the
    /// padded ring pair (`gl_up_scales`: the padded slices' TRF holds DM leftovers, and `* 0` makes their
    /// outputs 0; the integer accumulator cannot be NaN).
    fn up_rows<URING: M, UXS: M, UYS: M>(device: &mut Device, wug_scale: &HbmTensor<f32, Chip, m![Xc2, Lu]>, i: usize) -> UpRows<URING, UXS, UYS> {
        let junk: DmTensor<f32, Chip, UCl, GSl, m![Lu / 16 % 4, Lu / 8 % 2, Lu % 8]> = DmTensor::new();
        let z: DmTensor<f32, Chip, UCl, GSl, m![Lu / 16 % 4, Lu / 8 % 2, Lu % 8]> = device
            .main
            .begin(junk.view())
            .fetch::<m![Lu / 16 % 4, Lu / 8 % 2], m![Lu % 8]>()
            .collect::<m![Lu / 16 % 4, Lu / 8 % 2], m![Lu % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_reinterpret::<i32>()
            .vector_logic(LogicBinaryOpI32::BitAnd, 0)
            .vector_reinterpret::<f32>()
            .vector_final()
            .commit_trim::<m![Lu % 8]>()
            .commit();
        let mut z: UpRows<URING, UXS, UYS> = unsafe { z.reshape() };
        wug_scale.view().tile::<m![Xc2], 1, m![Xc2 = 1 # 2, Lu]>(i).to_dm_view(&mut device.tdma, z.view_mut());
        z
    }

    /// Pass `p`'s 16 values per slice of an `up_rows` tensor in the VRF.
    fn col_rows<URING: M, UXS: M, UYS: M>(device: &mut Device, d: &UpRows<URING, UXS, UYS>, p: usize) -> VrfTensor<f32, Chip, UCl, USl<URING, UXS, UYS>, m![Lu / 8 % 2, Lu % 8]> {
        device
            .sub
            .begin(d.view().tile::<m![Lu / 16 % 4], 1, m![Lu / 16 % 4 = 1 # 4, Lu / 8 % 2, Lu % 8]>(p))
            .fetch::<m![Lu / 8 % 2], m![Lu % 8]>()
            .collect::<m![Lu / 8 % 2], m![Lu % 8]>()
            .to_vrf()
    }

    /// One Up/Gate contraction over token group `q` (`gemm::contract` with the token axes `[T % QS / 64, T % 64]`)
    /// and the two-flit epilogue `bf16(((f32((acc << 2) + R2) + C') * sx) * ws)`, written into group `q` of `y`.
    /// Shift and add `R2` in the fixed-point cluster. With `|acc| <= 3840 * 64`, the sum
    /// has magnitude less than `2^21` and converts to f32 exactly.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn contract4q<T: AxisName, URING: M, UXS: M, UYS: M, K: AxisName, const F: usize, const CH: usize, const CH2: usize, const QS: usize, const NQ: usize>(
        device: &mut Device,
        x: &XR4<T, URING, K, CH, QS>,
        w: &gemm::WTrf<UCl, USl<URING, UXS, UYS>, Lu, K, F, CH, CH2>,
        r2: &VrfTensor<i32, Chip, UCl, USl<URING, UXS, UYS>, m![T % { QS }]>,
        cw: &VrfTensor<f32, Chip, UCl, USl<URING, UXS, UYS>, m![Lu / 8 % 2, Lu % 8]>,
        sx: &VrfTensor<f32, Chip, UCl, USl<URING, UXS, UYS>, m![T % { QS }]>,
        ws: &VrfTensor<f32, Chip, UCl, USl<URING, UXS, UYS>, m![Lu / 8 % 2, Lu % 8]>,
        q: usize,
        y: &mut YQ<T, URING, UXS, UYS, QS>,
    ) {
        device
            .main
            .begin(x.view().tile::<m![T / { QS }], 1, m![T / { QS } = 1 # { NQ }, T % { QS } / 64, K % { CH }]>(crate::device::tokenwise::common::at::<NQ>(q)))
            .fetch::<m![T % { QS } / 64, K / 32 % { F }], m![K % 32]>()
            .switch::<USl<URING, UXS, UYS>, m![T % { QS } / 64, T % 64, K / 32 % { F }, K / { CH } % 4]>(SwitchConfig::Broadcast01 {
                slice1: 64,
                slice0: 4,
                time0: F,
            })
            .collect::<m![T % { QS } / 64, T % 64, K / 32 % { F }, K / { CH } % 4], m![K % 32]>()
            .contract_outer::<m![T % { QS } / 64, T % 64, K / 32 % { F }, K / { CH2 } % 2, Lu / 8 % 2], m![K / { CH } % 2, K % 32], _, _, _>(w)
            .contract_packet::<m![1]>()
            .contract_time::<m![T % { QS } / 64, T % 64, Lu / 8 % 2]>()
            .contract_lane::<m![T % { QS } / 64, T % 64, Lu / 8 % 2], m![Lu % 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::LeftShift, 2)
            .vector_fxp(FxpBinaryOp::AddFxp, r2)
            .vector_fxp_to_fp(31)
            .vector_narrow_split::<m![T % { QS } / 64, T % 64, Lu / 8 % 2, Lu / 4 % 2], m![Lu % 4]>()
            .vector_fp_binary(FpBinaryOp::AddF, cw)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), sx)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), ws)
            .vector_widen_concat::<m![T % { QS } / 64, T % 64, Lu / 8 % 2], m![Lu % 8]>()
            .vector_final()
            .cast::<bf16, m![Lu % 8 # 16]>()
            .commit_trim::<m![Lu % 8]>()
            .commit_view(y.view_mut().tile::<m![T / { QS }], 1, m![T / { QS } = 1 #{!} { NQ }, T % { QS } / 64, T % 64, Lu / 8 % 2, Lu % 8]>(crate::device::tokenwise::common::at::<NQ>(q)));
    }

pub(crate) mod q4f8;

pub(crate) mod serve;

mod xg;
