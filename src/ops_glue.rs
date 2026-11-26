//! Device allocators for each layer’s Sliding and Full FP8 KV pools.
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;
use crate::device::layout::Cluster;

// Each pool must start finite, including masked rows that attention may gather.
macro_rules! kv_pool {
    ($name:ident, $page:ident, $pool:ident, $rows:literal, $total:literal, $count:ident, [$($width:tt)+]) => {
        #[device(chip = 1)]
        pub fn $name(
            device: &mut Device,
            zeros: &HbmTensor<f8e4m3, Chip, m![$page, $($width)+]>,
        ) -> (HbmTensor<f8e4m3, Chip, m![$pool, $($width)+]>, HbmTensor<f8e4m3, Chip, m![$pool, $($width)+]>) {
            let zeros: DmTensor<f8e4m3, Chip, Cluster, m![$page % 256], m![$page / 256, $($width)+]> =
                zeros.to_dm(&mut device.tdma);
            let zeros: DmTensor<f8e4m3, Chip, Cluster, m![$pool % 256], m![$pool % $rows / 256, $($width)+]> =
                unsafe { zeros.reshape() };
            let mut k: HbmTensor<f8e4m3, Chip, m![$pool, $($width)+]> = HbmTensor::new();
            let mut v: HbmTensor<f8e4m3, Chip, m![$pool, $($width)+]> = HbmTensor::new();
            for page in 0..$count::SIZE {
                zeros.view().to_hbm_view(&mut device.tdma, k.view_mut()
                    .tile::<m![$pool], $rows, m![$pool = $rows #{!} $total, $($width)+]>(page * $page::SIZE));
                zeros.view().to_hbm_view(&mut device.tdma, v.view_mut()
                    .tile::<m![$pool], $rows, m![$pool = $rows #{!} $total, $($width)+]>(page * $page::SIZE));
            }
            (k, v)
        }

        const _: () = assert!($page::SIZE == $rows && $pool::SIZE == $total);
    };
}

kv_pool!(allocate_sliding_pool8, Ts, Rs, 1024, 104448, Ao, [Ns, Ds]);
kv_pool!(allocate_full_pool, Tf, Rf, 512, 263168, Kp, [Df]);
