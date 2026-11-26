//! Shared FP8 Down weight views and pairwise f32 reduction.
use super::*;

pub(super) fn weight_quarter<'a>(
    wd: &'a HbmTensor<f8e4m3, Chip, m![Kq4, H, Kd]>,
    pass: usize,
    group: usize,
) -> HbmTensorView<'a, f8e4m3, Chip, m![Kq4 / 2 = 1 # 2, Kq4 % 2 = 1 # 2, H, Kd]> {
    wd.view()
        .tile::<m![Kq4 % 2], 1, m![Kq4 / 2, Kq4 % 2 = 1 # 2, H, Kd]>(pass)
        .tile::<m![Kq4 / 2], 1, m![Kq4 / 2 = 1 # 2, Kq4 % 2 = 1 # 2, H, Kd]>(group)
}

// Keep the (q0 + q1) + (q2 + q3) tree on both slice layouts.
macro_rules! down_pair_sum {
    ($pipe:expr, $a:expr, $b:expr, ring [$($rows:tt)*]) => {
        down_pair_sum!($pipe, $a, $b,
            m![$($rows)* T % 64 / 8, Pi, T % 8, H / 8 % 2],
            m![$($rows)* T % 64, H / 8 % 2],
            m![$($rows)* T % 64, H / 8 % 2, H / 4 % 2],
            m![H % 8], m![H % 4])
    };
    ($pipe:expr, $a:expr, $b:expr, slices $ps:ident) => {
        down_pair_sum!($pipe, $a, $b,
            m![T / { $ps }, H % 480 / 16, Pi, H % 16],
            m![T / { $ps }, H % 480],
            m![T / { $ps }, H % 480, T / 4 % 2],
            m![T % 8], m![T % 4])
    };
    ($pipe:expr, $a:expr, $b:expr, $fetch:ty, $unzip:ty, $narrow:ty, $lane:ty, $half:ty) => {
        ($pipe).begin_interleaved::<Pi, _, _, _, _, _>($a, $b)
            .fetch::<$fetch, $lane>()
            .collect::<$fetch, $lane>()
            .vector_init().vector_intra_slice_unzip::<Pi, $unzip>()
            .vector_narrow_split::<$narrow, $half>()
            .vector_fp_zip(FpBinaryOp::AddF)
    };
}
pub(super) use down_pair_sum;
