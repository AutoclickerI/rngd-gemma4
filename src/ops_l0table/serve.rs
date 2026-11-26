//! Layer-0 lookup, RoPE and attention staging.
use crate::device::layout::{Cluster, Slice};
use crate::device::proj_fast::heads;
use crate::device::tokenwise_int8::rotate::{TCl, TSl};
use crate::ops_core::moves;
use crate::{Chip, axes::*};
use furiosa_opt_std::prelude::*;

#[derive(DeviceSend)]
pub struct Tables<'a> {
    pub q: &'a HbmTensor<bf16, Chip, m![W, Qs]>,
    pub k: &'a HbmTensor<bf16, Chip, m![W, Ps]>,
    pub v: &'a HbmTensor<f8e4m3, Chip, m![W, Ps]>,
}

type Row<D, T, const HALF: usize, EL> = DmTensor<D, Chip, TCl<T, HALF>, TSl<T, HALF>, EL>;
type QHeads = m![Ns, Gs];

/// The embedding offset is id * 7680. After shifting out 512, Q31 multiplication
/// by round(2^31 / 15) recovers the ID exactly with rounding over the whole vocabulary.
fn table_offsets<R: M, const SHIFT: i32>(
    device: &mut Device, index: &DmTensor<i32, Chip, Cluster, Slice, R>,
) -> HbmTensor<i32, Chip, R> {
    let offsets: DmTensor<i32, Chip, Cluster, Slice, R> = device.main.begin(index.view())
        .fetch::<m![{R} / 8], m![{R} % 8]>()
        .collect::<m![{R} / 8], m![{R} % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::LogicRightShift, 9)
        .vector_fxp(FxpBinaryOp::MulFxp, 143_165_577)
        .vector_fxp(FxpBinaryOp::LeftShift, SHIFT)
        .vector_final().commit_trim::<m![{R} % 8]>().commit();
    let mut out = HbmTensor::new();
    offsets.view().to_hbm_view(&mut device.tdma, out.view_mut());
    out
}

fn lookup<T: AxisName, R: M, const HALF: usize>(
    device: &mut Device, tables: &Tables<'_>, index: &HbmTensor<i32, Chip, R>,
    rope: &HbmTensor<i32, Chip, R>, cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
) -> (Row<bf16, T, HALF, m![Qs]>, Row<bf16, T, HALF, m![Ps]>, Row<f8e4m3, T, HALF, m![Ps]>) {
    let ix: DmTensor<i32, Chip, Cluster, Slice, R> = index.to_dm(&mut device.tdma);
    let qi = table_offsets::<R, 13>(device, &ix);
    let ki = table_offsets::<R, 12>(device, &ix);
    let vi = table_offsets::<R, 11>(device, &ix);
    let c: Row<bf16, T, HALF, m![Ds]> = cos.gather::<m![E], m![Ds]>().by_byte_offsets(rope).to_dm(&mut device.tdma);
    let s: Row<bf16, T, HALF, m![Ds]> = sin.gather::<m![E], m![Ds]>().by_byte_offsets(rope).to_dm(&mut device.tdma);
    let coeff = heads::rope_halves::<T, HALF>(device, &c, &s);
    let q: Row<bf16, T, HALF, m![Qs]> = tables.q.gather::<m![W], m![Qs]>().by_byte_offsets(&qi).to_dm(&mut device.tdma);
    let q = heads::rope_pair::<T, HALF, QHeads>(device, &unsafe { q.reshape() }, &coeff);
    let k: Row<bf16, T, HALF, m![Ps]> = tables.k.gather::<m![W], m![Ps]>().by_byte_offsets(&ki).to_dm(&mut device.tdma);
    let k = heads::rope_pair::<T, HALF, m![Ns]>(device, &unsafe { k.reshape() }, &coeff);
    let v = tables.v.gather::<m![W], m![Ps]>().by_byte_offsets(&vi).to_dm(&mut device.tdma);
    (unsafe { q.reshape() }, unsafe { k.reshape() }, v)
}

fn lookup_half<const P: usize>(
    device: &mut Device, tables: &Tables<'_>,
    index: &HbmTensor<i32, Chip, m![Tok1024]>, rope: &HbmTensor<i32, Chip, m![Tok1024]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>, sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
    q: &mut HbmTensor<bf16, Chip, m![Tok1024, Qs]>,
    k: &mut HbmTensor<bf16, Chip, m![Tok1024, Ps]>,
    v: &mut HbmTensor<f8e4m3, Chip, m![Tok1024, Ps]>,
) {
    let mut ix = HbmTensor::<i32, Chip, m![Tok1024 % 512]>::new();
    let mut pos = HbmTensor::<i32, Chip, m![Tok1024 % 512]>::new();
    index.view().tile::<m![Tok1024 / 512], 1, m![Tok1024 / 512 = 1 # 2, Tok1024 % 512]>(P)
        .to_hbm_view(&mut device.tdma, ix.view_mut());
    rope.view().tile::<m![Tok1024 / 512], 1, m![Tok1024 / 512 = 1 # 2, Tok1024 % 512]>(P)
        .to_hbm_view(&mut device.tdma, pos.view_mut());
    let (qr, kr, vr) = lookup::<Tok1024, m![Tok1024 % 512], 256>(device, tables, &ix, &pos, cos, sin);
    qr.view().to_hbm_view(&mut device.tdma, q.view_mut().tile::<m![Tok1024 / 512], 1, m![Tok1024 / 512 = 1 #{!} 2, Tok1024 % 512, Qs]>(P));
    kr.view().to_hbm_view(&mut device.tdma, k.view_mut().tile::<m![Tok1024 / 512], 1, m![Tok1024 / 512 = 1 #{!} 2, Tok1024 % 512, Ps]>(P));
    vr.view().to_hbm_view(&mut device.tdma, v.view_mut().tile::<m![Tok1024 / 512], 1, m![Tok1024 / 512 = 1 #{!} 2, Tok1024 % 512, Ps]>(P));
}

fn lookup_wide_pass<const P: usize>(
    device: &mut Device, tables: &Tables<'_>,
    index: &HbmTensor<i32, Chip, m![Tok1536]>, rope: &HbmTensor<i32, Chip, m![Tok1536]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>, sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
    q: &mut HbmTensor<bf16, Chip, m![Tok1536, Qs]>,
    k: &mut HbmTensor<bf16, Chip, m![Tok1536, Ps]>,
    v: &mut HbmTensor<f8e4m3, Chip, m![Tok1536, Ps]>,
) {
    let mut ix = HbmTensor::<i32, Chip, m![Tok1536 % 512]>::new();
    let mut pos = HbmTensor::<i32, Chip, m![Tok1536 % 512]>::new();
    index.view().tile::<m![Tok1536 / 512], 1, m![Tok1536 / 512 = 1 # 3, Tok1536 % 512]>(P)
        .to_hbm_view(&mut device.tdma, ix.view_mut());
    rope.view().tile::<m![Tok1536 / 512], 1, m![Tok1536 / 512 = 1 # 3, Tok1536 % 512]>(P)
        .to_hbm_view(&mut device.tdma, pos.view_mut());
    let (qr, kr, vr) = lookup::<Tok1536, m![Tok1536 % 512], 256>(device, tables, &ix, &pos, cos, sin);
    qr.view().to_hbm_view(&mut device.tdma, q.view_mut().tile::<m![Tok1536 / 512], 1, m![Tok1536 / 512 = 1 #{!} 3, Tok1536 % 512, Qs]>(P));
    kr.view().to_hbm_view(&mut device.tdma, k.view_mut().tile::<m![Tok1536 / 512], 1, m![Tok1536 / 512 = 1 #{!} 3, Tok1536 % 512, Ps]>(P));
    vr.view().to_hbm_view(&mut device.tdma, v.view_mut().tile::<m![Tok1536 / 512], 1, m![Tok1536 / 512 = 1 #{!} 3, Tok1536 % 512, Ps]>(P));
}

macro_rules! pre {
    ($name:ident, $t:ident, $half:literal, $hs:ty, $he:ty, $ks:ty, $ke:ty, $ft:ty, $ct:ty) => {
        #[device(chip = 1)]
        pub fn $name(
            device: &mut Device, index: &HbmTensor<i32, Chip, m![$t]>,
            rope: &HbmTensor<i32, Chip, m![$t]>,
            rows: &HbmTensor<i32, Chip, m![$t, Hh]>, stage: &HbmTensor<i32, Chip, m![$t]>,
            tables: Tables<'_>, cos: &HbmTensor<bf16, Chip, m![E, Ds]>, sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
            qin: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
            k0: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
            v0: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
        ) {
            let mut q = HbmTensor::<bf16, Chip, m![$t, Qs]>::new();
            let mut k = HbmTensor::<bf16, Chip, m![$t, Ps]>::new();
            let mut v = HbmTensor::<f8e4m3, Chip, m![$t, Ps]>::new();
            pre!(@lookup $t, $half, device, tables, index, rope, cos, sin, q, k, v);
            moves::scatter_q::<$t, $hs, $he>(device, q, rows, qin);
            moves::scatter_kv_e4m3::<$t, $ks, $ke, $ft, $ct>(device, k, stage, k0);
            let v: HbmTensor<f8e4m3, Chip, m![$t, Ns, Ds]> = unsafe { v.reshape() };
            let v: DmTensor<f8e4m3, Chip, Cluster, $ks, $ke> = v.to_dm(&mut device.tdma);
            v0.view_mut().scatter::<m![Rs], m![Ns, Ds]>().by_byte_offsets(stage.view()).from_dm_view(&mut device.tdma, v.view());
        }
    };
    (@lookup Tok1024, $half:literal, $d:ident, $t:ident, $i:ident, $r:ident, $c:ident, $s:ident, $q:ident, $k:ident, $v:ident) => {
        lookup_half::<0>($d, &$t, $i, $r, $c, $s, &mut $q, &mut $k, &mut $v);
        lookup_half::<1>($d, &$t, $i, $r, $c, $s, &mut $q, &mut $k, &mut $v);
    };
    (@lookup Tok1536, $half:literal, $d:ident, $t:ident, $i:ident, $r:ident, $c:ident, $s:ident, $q:ident, $k:ident, $v:ident) => {
        lookup_wide_pass::<0>($d, &$t, $i, $r, $c, $s, &mut $q, &mut $k, &mut $v);
        lookup_wide_pass::<1>($d, &$t, $i, $r, $c, $s, &mut $q, &mut $k, &mut $v);
        lookup_wide_pass::<2>($d, &$t, $i, $r, $c, $s, &mut $q, &mut $k, &mut $v);
    };
    (@lookup $tok:ident, $half:literal, $d:ident, $t:ident, $i:ident, $r:ident, $c:ident, $s:ident, $q:ident, $k:ident, $v:ident) => {
        let (qr, kr, vr) = lookup::<$tok, m![$tok], $half>($d, &$t, $i, $r, $c, $s);
        qr.view().to_hbm_view(&mut $d.tdma, $q.view_mut());
        kr.view().to_hbm_view(&mut $d.tdma, $k.view_mut());
        vr.view().to_hbm_view(&mut $d.tdma, $v.view_mut());
    };
}

pre!(lookup_t128, Tok128, 64, m![Tok128, Hh], m![Hqs], m![Tok128, 1 # 2], m![Ns, Ds], m![Ns, Ds / 16], m![Ns, Ds / 8]);
pre!(lookup_t1024, Tok1024, 256, m![Tok1024 % 128, Hh], m![Tok1024 / 128, Hqs], m![Tok1024 % 128, 1 # 2], m![Tok1024 / 128, Ns, Ds], m![Tok1024 / 128, Ns, Ds / 16], m![Tok1024 / 128, Ns, Ds / 8]);
pre!(lookup_t1536, Tok1536, 256, m![Tok1536 % 128, Hh], m![Tok1536 / 128, Hqs], m![Tok1536 % 128, 1 # 2], m![Tok1536 / 128, Ns, Ds], m![Tok1536 / 128, Ns, Ds / 16], m![Tok1536 / 128, Ns, Ds / 8]);

macro_rules! pack {
    ($name:ident, $t:ident, $s:ty, $r:ty) => {
        #[device(chip = 1)]
        pub fn $name<
            X: AxisName,
        >(
            device: &mut Device, x: &HbmTensor<bf16, Chip, m![X, H]>,
            src: &HbmTensor<i32, Chip, m![$t]>, out: &mut HbmTensor<bf16, Chip, m![$t, H]>,
        ) {
            let rows: DmTensor<bf16, Chip, Cluster, $s, $r> = x.gather::<m![X], m![H]>().by_byte_offsets(src).to_dm(&mut device.tdma);
            rows.view().to_hbm_view(&mut device.tdma, out.view_mut());
        }
    };
}
pack!(pack_t128, Tok128, m![Tok128, 1 # 2], m![H]);
pack!(pack_t512, Tok512, m![Tok512 % 256], m![Tok512 / 256, H]);
pack!(pack_t1024, Tok1024, m![Tok1024 % 128, 1 # 2], m![Tok1024 / 128, H]);
pack!(pack_t1536, Tok1536, m![Tok1536 % 128, 1 # 2], m![Tok1536 / 128, H]);

#[device(chip = 1)]
pub fn publish_t1024(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Tok1024, Qs]>,
    k: &HbmTensor<bf16, Chip, m![Tok1024, Ps]>,
    v: &HbmTensor<f8e4m3, Chip, m![Tok1024, Ps]>,
    qi: &HbmTensor<i32, Chip, m![Tok1024]>,
    ki: &HbmTensor<i32, Chip, m![Tok1024]>,
    vi: &HbmTensor<i32, Chip, m![Tok1024]>,
    qt: &mut HbmTensor<bf16, Chip, m![W, Qs]>,
    kt: &mut HbmTensor<bf16, Chip, m![W, Ps]>,
    vt: &mut HbmTensor<f8e4m3, Chip, m![W, Ps]>,
) {
    let q: DmTensor<bf16, Chip, m![Tok1024 / 512], m![Tok1024 % 256], m![Tok1024 / 256 % 2, Qs]> =
        q.to_dm(&mut device.tdma);
    let k: DmTensor<bf16, Chip, m![Tok1024 / 512], m![Tok1024 % 256], m![Tok1024 / 256 % 2, Ps]> =
        k.to_dm(&mut device.tdma);
    let v: DmTensor<f8e4m3, Chip, m![Tok1024 / 512], m![Tok1024 % 256], m![Tok1024 / 256 % 2, Ps]> =
        v.to_dm(&mut device.tdma);
    qt.view_mut()
        .scatter::<m![W], m![Qs]>()
        .by_byte_offsets(qi.view())
        .from_dm_view(&mut device.tdma, q.view());
    kt.view_mut()
        .scatter::<m![W], m![Ps]>()
        .by_byte_offsets(ki.view())
        .from_dm_view(&mut device.tdma, k.view());
    vt.view_mut()
        .scatter::<m![W], m![Ps]>()
        .by_byte_offsets(vi.view())
        .from_dm_view(&mut device.tdma, v.view());
}

/// Allocates the three table tensors on the device and zeroes every row from one 1024-row zero tile per
/// table, so only those 14 MiB cross PCIe instead of the whole 3.76 GB (`ops_glue::allocate_sliding_pool8`'s
/// pattern). Startup only. `publish_t1024` overwrites every row before the table serves; the zeroes keep
/// the contents defined in between. The DM layout is `publish_t1024`'s, expressed over the table's rows.
#[device(chip = 1)]
pub fn allocate_t1024(
    device: &mut Device,
    zq: &HbmTensor<bf16, Chip, m![Tok1024, Qs]>,
    zk: &HbmTensor<bf16, Chip, m![Tok1024, Ps]>,
    zv: &HbmTensor<f8e4m3, Chip, m![Tok1024, Ps]>,
) -> (
    HbmTensor<bf16, Chip, m![W, Qs]>,
    HbmTensor<bf16, Chip, m![W, Ps]>,
    HbmTensor<f8e4m3, Chip, m![W, Ps]>,
) {
    let zq: DmTensor<bf16, Chip, m![Tok1024 / 512], m![Tok1024 % 256], m![Tok1024 / 256 % 2, Qs]> =
        zq.to_dm(&mut device.tdma);
    let zk: DmTensor<bf16, Chip, m![Tok1024 / 512], m![Tok1024 % 256], m![Tok1024 / 256 % 2, Ps]> =
        zk.to_dm(&mut device.tdma);
    let zv: DmTensor<f8e4m3, Chip, m![Tok1024 / 512], m![Tok1024 % 256], m![Tok1024 / 256 % 2, Ps]> =
        zv.to_dm(&mut device.tdma);
    let zq: DmTensor<bf16, Chip, m![W % 1024 / 512], m![W % 256], m![W % 512 / 256, Qs]> =
        unsafe { zq.reshape() };
    let zk: DmTensor<bf16, Chip, m![W % 1024 / 512], m![W % 256], m![W % 512 / 256, Ps]> =
        unsafe { zk.reshape() };
    let zv: DmTensor<f8e4m3, Chip, m![W % 1024 / 512], m![W % 256], m![W % 512 / 256, Ps]> =
        unsafe { zv.reshape() };
    let mut qt: HbmTensor<bf16, Chip, m![W, Qs]> = HbmTensor::new();
    let mut kt: HbmTensor<bf16, Chip, m![W, Ps]> = HbmTensor::new();
    let mut vt: HbmTensor<f8e4m3, Chip, m![W, Ps]> = HbmTensor::new();
    for tile in 0..W::SIZE / Tok1024::SIZE {
        let at = tile * Tok1024::SIZE;
        zq.view().to_hbm_view(&mut device.tdma, qt.view_mut().tile::<m![W], 1024, m![W = 1024 #{!} 262144, Qs]>(at));
        zk.view().to_hbm_view(&mut device.tdma, kt.view_mut().tile::<m![W], 1024, m![W = 1024 #{!} 262144, Ps]>(at));
        zv.view().to_hbm_view(&mut device.tdma, vt.view_mut().tile::<m![W], 1024, m![W = 1024 #{!} 262144, Ps]>(at));
    }
    (qt, kt, vt)
}
