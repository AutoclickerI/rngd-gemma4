//! INT8 Full O with token-partitioned clusters: eight output groups x 32 token groups use all 256 slices.
//! Exact i32 K8192 sum; preserve rotation, scaling, bf16 and post-attention arithmetic.
use super::*;
use super::rotate::{TCl, TSl, Kq};

type Sl<T, const HALF: usize> = m![H / 480, T % { HALF } / 8];
pub(crate) type Weight<T, const HALF: usize> =
    DmTensor<i8, Chip, TCl<T, HALF>, m![H / 480, H % 32], m![H / 32 % 15, Kq]>;
type X<T, const HALF: usize, const PS: usize> =
    DmTensor<i8, Chip, TCl<T, HALF>, Sl<T, HALF>, m![T / { PS }, T % 8, Kq]>;
type Parts<T, const HALF: usize, const PS: usize> =
    DmTensor<i32, Chip, TCl<T, HALF>, Sl<T, HALF>, m![Pi, H % 480, T / { PS }, T % 8]>;

pub(crate) fn load_weight<T: AxisName, const HALF: usize>(
    device: &mut Device, wo: &HbmTensor<i8, Chip, m![H, Qf]>, half: usize,
) -> Weight<T, HALF> {
    let w = wo.view().tile::<m![Qf / 4096], 1, m![H, Qf / 4096 = 1 # 2, Qf % 4096]>(half);
    let w: DmTensor<i8, Chip, TCl<T, HALF>, m![H / 480, H % 32], m![H / 32 % 15, Qf % 4096]> = w.to_dm(&mut device.tdma);
    unsafe { w.reshape() }
}

fn half<T: AxisName, const HALF: usize, const PS: usize>(
    device: &mut Device, x: &X<T, HALF, PS>, w: &Weight<T, HALF>, h: usize,
    pp: &mut Parts<T, HALF, PS>,
) {
    let xt: TrfTensor<i8, Chip, TCl<T, HALF>, Sl<T, HALF>, m![T % 8],
        m![T / { PS }, Kq / 32 % 16, Kq / 1024 % 4, Kq / 512 % 2, Kq % 32]> = device.main.begin(x.view())
        .fetch::<m![T % 8, T / { PS }, Kq / 32 % 16, Kq / 1024 % 4, Kq / 512 % 2], m![Kq % 32]>()
        .collect::<m![T % 8, T / { PS }, Kq / 32 % 16, Kq / 1024 % 4, Kq / 512 % 2], m![Kq % 32]>()
        .to_trf();
    device.main.begin(w.view())
        .fetch::<m![H / 32 % 15, Kq / 32 % 16, Kq / 512 % 8], m![Kq % 32]>()
        .switch::<Sl<T, HALF>, m![H / 32 % 15, H % 32, Kq / 32 % 16, Kq / 512 % 8]>(
            SwitchConfig::Broadcast01 { slice1: 32, slice0: 1, time0: 128 })
        .collect::<m![H / 32 % 15, H % 32, Kq / 32 % 16, Kq / 512 % 8], m![Kq % 32]>()
        .contract_outer::<m![H / 32 % 15, H % 32, Kq / 32 % 16, Kq / 1024 % 4, T / { PS }],
            m![Kq / 512 % 2, Kq % 32], _, _, _>(&xt)
        .contract_packet::<m![1]>()
        .contract_time::<m![H / 32 % 15, H % 32, T / { PS }]>()
        .contract_lane::<m![H / 32 % 15, H % 32, T / { PS }], m![T % 8]>(LaneMode::Interleaved)
        .commit_trim::<m![T % 8]>()
        .commit_view(pp.view_mut().tile::<m![Pi], 1,
            m![Pi = 1 #{!} 2, H % 480, T / { PS }, T % 8]>(h));
}

pub(crate) fn gemm<T: AxisName, const HALF: usize, const PS: usize>(
    device: &mut Device, codes: &HbmTensor<i8, Chip, m![T, Qf]>,
    scales: &HbmTensor<f32, Chip, m![T]>,
    wd0: Weight<T, HALF>, wo: &HbmTensor<i8, Chip, m![H, Qf]>,
    ws: &HbmTensor<f32, Chip, m![H]>,
) -> (DmTensor<bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![Kh]>, DmTensor<bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![Kh]>) {
    let mut pp = Parts::<T, HALF, PS>::new();
    let cd: HbmTensorView<'_, i8, Chip, m![T, Qf / 4096, Kq]> = unsafe { codes.view().reshape() };
    let x0: X<T, HALF, PS> = cd.tile::<m![Qf / 4096], 1, m![T, Qf / 4096 = 1 # 2, Kq]>(0).to_dm(&mut device.tdma);
    half::<T, HALF, PS>(device, &x0, &wd0, 0, &mut pp);
    let x1: X<T, HALF, PS> = cd.tile::<m![Qf / 4096], 1, m![T, Qf / 4096 = 1 # 2, Kq]>(1).to_dm(&mut device.tdma);
    let wd1 = load_weight::<T, HALF>(device, wo, 1);
    half::<T, HALF, PS>(device, &x1, &wd1, 1, &mut pp);
    // Scales follow each token into the same geometry as its integer sum.
    let sd: DmTensor<f32, Chip, TCl<T, HALF>, Sl<T, HALF>, m![T / { PS }, T % 8]> = scales.to_dm(&mut device.tdma);
    let sx: VrfTensor<f32, Chip, TCl<T, HALF>, Sl<T, HALF>, m![T / { PS }, T % 8]> = device.sub.begin(sd.view()).fetch::<m![T / { PS }], m![T % 8]>().collect::<m![T / { PS }], m![T % 8]>().to_vrf();
    let wd: DmTensor<f32, Chip, TCl<T, HALF>, Sl<T, HALF>, m![H % 480]> = ws.to_dm(&mut device.tdma);
    let wv: VrfTensor<f32, Chip, TCl<T, HALF>, Sl<T, HALF>, m![H % 480]> = device.sub.begin(wd.view())
        .fetch::<m![H % 480 / 8], m![H % 8]>().collect::<m![H % 480 / 8], m![H % 8]>().to_vrf();
    let y: DmTensor<bf16, Chip, TCl<T, HALF>, Sl<T, HALF>, m![T / { PS }, T % 8, H % 480]> = device.main.begin(pp.view())
        .fetch::<m![T / { PS }, H % 480 / 16, Pi, H % 16], m![T % 8]>()
        .collect::<m![T / { PS }, H % 480 / 16, Pi, H % 16], m![T % 8]>()
        .vector_init().vector_intra_slice_unzip::<Pi, m![T / { PS }, H % 480]>()
        .vector_fxp_zip(FxpBinaryOp::AddFxp)
        .vector_fxp_to_fp(31)
        .vector_narrow_split::<m![T / { PS }, H % 480, T / 4 % 2], m![T % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &sx)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &wv)
        .vector_widen_concat::<m![T / { PS }, H % 480], m![T % 8]>()
        .vector_final().cast::<bf16, m![T % 8 # 16]>()
        .transpose::<m![T / { PS }, H % 480 / 4, T % 8], m![H % 4 # 16]>()
        .commit_trim::<m![H % 4]>().commit();
    let mut r0: DmTensor<bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![H]> = DmTensor::new();
    let mut r1: DmTensor<bf16, Chip, TCl<T, HALF>, TSl<T, HALF>, m![H]> = DmTensor::new();
    y.view().tile::<m![T / { PS }], 1, m![T / { PS } = 1 # 2, T % 8, H % 480]>(0).to_dm_view(&mut device.tdma, r0.view_mut());
    y.view().tile::<m![T / { PS }], 1, m![T / { PS } = 1 # 2, T % 8, H % 480]>(1).to_dm_view(&mut device.tdma, r1.view_mut());
    (unsafe { r0.reshape() }, unsafe { r1.reshape() })
}
