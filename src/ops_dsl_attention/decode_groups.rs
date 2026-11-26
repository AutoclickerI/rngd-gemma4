//! Axis layout for an eight-row Sliding decode group with a 384-position window.

pub mod w384_fp8 {
    use furiosa_opt_std::prelude::*;

    axes![Rw = 8, Po = 64, Pi = 6, Px = 64];

    pub type Chip = m![1];
    pub type C = m![Rw / 4];
    pub type S = m![Rw % 4, Po];
    pub type S1 = m![Rw % 4, 1 # 64];
    pub type SX = m![Rw % 4, Px];
    pub type Off = HbmTensor<i32, Chip, m![Rw, Pi, Po % 32, Po / 32]>;
    pub type Cap = HbmTensor<f32, Chip, m![Rw, Po, Pi]>;
}

const _: () = assert!(64 * <w384_fp8::Pi as furiosa_opt_std::prelude::AxisName>::SIZE == super::window::GROUP_WINDOW, "384-position group matches Sliding window");
