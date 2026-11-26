//! FP32 projection scales and head norms, with only the final BF16 table narrowing.
use crate::device::tokenwise_int8::qkv::Nq;
use crate::device::proj_fast::heads::table_vrf;
use crate::device::tokenwise_int8::rotate::{TCl, TSl};
use crate::{Chip, EPS, axes::*};
use furiosa_opt_std::prelude::*;

type Row<D, T, const HALF: usize, EL> = DmTensor<D, Chip, TCl<T, HALF>, TSl<T, HALF>, EL>;
type Vrf<T, const HALF: usize, EL> = VrfTensor<f32, Chip, TCl<T, HALF>, TSl<T, HALF>, EL>;
type QO = m![Ns, Gs];

fn rms<T: AxisName, const HALF: usize, O: M>(
    device: &mut Device,
    x: &Row<f32, T, HALF, m![{ O }, Ds]>,
) -> Vrf<T, HALF, m![{O},1 #8]> {
    let ms: Row<f32, T, HALF, m![{O},1 #8]> = device
        .main
        .begin(x.view())
        .fetch::<m![{ O }, Ds / 8], m![Ds % 8]>()
        .collect::<m![{ O }, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{ O }, Ds / 4], m![Ds % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::AddF, -0f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<Ds, m![{ O }], m![1 #4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(256f32)
        .vector_widen_pad::<m![1 #8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 #8]>()
        .commit();
    let root: Row<f32, T, HALF, m![{O},1 #8]> = device
        .main
        .begin(ms.view())
        .fetch::<m![{ O }], m![1 #8]>()
        .collect::<m![{ O }], m![1 #8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 #4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 #8]>()
        .vector_final()
        .commit_trim::<m![1 #8]>()
        .commit();
    device
        .sub
        .begin(root.view())
        .fetch::<m![{ O }], m![1 #8]>()
        .collect::<m![{ O }], m![1 #8]>()
        .to_vrf()
}

fn normalized<T: AxisName, const HALF: usize, O: M>(
    device: &mut Device,
    x: &Row<f32, T, HALF, m![{ O }, Ds]>,
    g: &Vrf<T, HALF, m![Ds]>,
) -> Row<bf16, T, HALF, m![{ O }, Ds]> {
    let r = rms::<T, HALF, O>(device, x);
    device
        .main
        .begin(x.view())
        .fetch::<m![{ O }, Ds / 8], m![Ds % 8]>()
        .collect::<m![{ O }, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{ O }, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &r)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), g)
        .vector_widen_concat::<m![{ O }, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds%8 #16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

fn normalized_v<T: AxisName, const HALF: usize>(
    device: &mut Device,
    x: &Row<f32, T, HALF, m![Ns, Ds]>,
) -> Row<f8e4m3, T, HALF, m![Ns, Ds]> {
    let r = rms::<T, HALF, m![Ns]>(device, x);
    device.main.begin(x.view())
        .fetch::<m![Ns, Ds / 8], m![Ds % 8]>()
        .collect::<m![Ns, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &r)
        .vector_widen_concat::<m![Ns, Ds / 8], m![Ds % 8]>()
        .vector_clip(ClipBinaryOpF32::Min, 448f32)
        .vector_clip(ClipBinaryOpF32::Max, -448f32)
        .vector_final()
        .cast::<f8e4m3, m![Ds % 8 # 32]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

fn pass<const PI: usize>(
    device: &mut Device,
    raw: &HbmTensor<f32, Chip, m![Tok1024, Nq]>,
    qg: &Vrf<Tok1024, 256, m![Ds]>,
    kg: &Vrf<Tok1024, 256, m![Ds]>,
    q: &mut HbmTensor<bf16, Chip, m![Tok1024, Qs]>,
    k: &mut HbmTensor<bf16, Chip, m![Tok1024, Ps]>,
    v: &mut HbmTensor<f8e4m3, Chip, m![Tok1024, Ps]>,
) {
    let qr: Row<f32, Tok1024, 256, m![Nq % 4096]> = raw
        .view()
        .tile::<m![Tok1024 / 512], 1, m![Tok1024/512=1 #2,Tok1024%512,Nq]>(PI)
        .tile::<m![Nq / 4096], 1, m![Tok1024/512=1 #2,Tok1024%512,Nq/4096=1 #2,Nq%4096]>(0)
        .to_dm(&mut device.tdma);
    let qr: Row<f32, Tok1024, 256, m![{ QO }, Ds]> = unsafe { qr.reshape() };
    let qn = normalized::<Tok1024, 256, QO>(device, &qr, qg);
    let qn: Row<bf16, Tok1024, 256, m![Qs]> = unsafe { qn.reshape() };
    qn.view().to_hbm_view(
        &mut device.tdma,
        q.view_mut()
            .tile::<m![Tok1024 / 512], 1, m![Tok1024/512=1 #{!}2,Tok1024%512,Qs]>(PI),
    );
    part_kv::<PI, 2>(device, raw, kg, k, v);
    part_kv::<PI, 3>(device, raw, kg, k, v);
}

fn part_kv<const PI: usize, const WHICH: usize>(
    device: &mut Device,
    raw: &HbmTensor<f32, Chip, m![Tok1024, Nq]>,
    kg: &Vrf<Tok1024, 256, m![Ds]>,
    k: &mut HbmTensor<bf16, Chip, m![Tok1024, Ps]>,
    v: &mut HbmTensor<f8e4m3, Chip, m![Tok1024, Ps]>,
) {
    let r: Row<f32, Tok1024, 256, m![Nq % 2048]> = raw
        .view()
        .tile::<m![Tok1024 / 512], 1, m![Tok1024/512=1 #2,Tok1024%512,Nq]>(PI)
        .tile::<m![Nq / 2048], 1, m![Tok1024/512=1 #2,Tok1024%512,Nq/2048=1 #4,Nq%2048]>(WHICH)
        .to_dm(&mut device.tdma);
    let r: Row<f32, Tok1024, 256, m![Ns, Ds]> = unsafe { r.reshape() };
    if WHICH == 2 {
        let n = normalized::<Tok1024, 256, m![Ns]>(device, &r, kg);
        let n: Row<bf16, Tok1024, 256, m![Ps]> = unsafe { n.reshape() };
        n.view().to_hbm_view(
            &mut device.tdma,
            k.view_mut()
                .tile::<m![Tok1024 / 512], 1, m![Tok1024/512=1 #{!}2,Tok1024%512,Ps]>(PI),
        );
    } else {
        let n = normalized_v::<Tok1024, 256>(device, &r);
        let n: Row<f8e4m3, Tok1024, 256, m![Ps]> = unsafe { n.reshape() };
        n.view().to_hbm_view(
            &mut device.tdma,
            v.view_mut()
                .tile::<m![Tok1024 / 512], 1, m![Tok1024/512=1 #{!}2,Tok1024%512,Ps]>(PI),
        );
    }
}

#[device(chip = 1)]
pub fn prepare_norm_t1024(
    device: &mut Device,
    raw: &HbmTensor<f32, Chip, m![Tok1024, Nq]>,
    qg: &HbmTensor<bf16, Chip, m![Ds]>,
    kg: &HbmTensor<bf16, Chip, m![Ds]>,
    q: &mut HbmTensor<bf16, Chip, m![Tok1024, Qs]>,
    k: &mut HbmTensor<bf16, Chip, m![Tok1024, Ps]>,
    v: &mut HbmTensor<f8e4m3, Chip, m![Tok1024, Ps]>,
) {
    let qg = table_vrf::<Tok1024, 256>(device, qg);
    let kg = table_vrf::<Tok1024, 256>(device, kg);
    pass::<0>(device, raw, &qg, &kg, q, k, v);
    pass::<1>(device, raw, &qg, &kg, q, k, v);
}
