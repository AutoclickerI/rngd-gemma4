use furiosa_opt_std::prelude::*;

use crate::axes::E;
use crate::Chip;

pub(crate) async fn upload_rope<D: AxisName>(
    device: &mut Device,
    dim: usize,
    theta: f32,
    rotated_dim: usize,
) -> Result<(HbmTensor<bf16, Chip, m![E, D]>, HbmTensor<bf16, Chip, m![E, D]>), Error> {
    let (cos, sin) = rope_data(dim, theta, rotated_dim);
    Ok((
        HostTensor::<bf16, m![E, D]>::from_vec(cos)
            .to_hbm(&mut device.pdma)
            .await?,
        HostTensor::<bf16, m![E, D]>::from_vec(sin)
            .to_hbm(&mut device.pdma)
            .await?,
    ))
}

fn rope_data(dim: usize, theta: f32, rotated_dim: usize) -> (Vec<bf16>, Vec<bf16>) {
    let half = dim / 2;
    let rotated_half = rotated_dim / 2;
    let mut cos = Vec::with_capacity(E::SIZE * dim);
    let mut sin = Vec::with_capacity(E::SIZE * dim);
    for pos in 0..E::SIZE {
        for i in 0..half {
            if i < rotated_half {
                let freq = (pos as f32) / theta.powf((2 * i) as f32 / dim as f32);
                cos.push(bf16::from_f32(freq.cos()));
                sin.push(bf16::from_f32(-freq.sin()));
            } else {
                cos.push(bf16::from_f32(1.0));
                sin.push(bf16::from_f32(0.0));
            }
        }
        for i in 0..half {
            if i < rotated_half {
                let freq = (pos as f32) / theta.powf((2 * i) as f32 / dim as f32);
                cos.push(bf16::from_f32(freq.cos()));
                sin.push(bf16::from_f32(freq.sin()));
            } else {
                cos.push(bf16::from_f32(1.0));
                sin.push(bf16::from_f32(0.0));
            }
        }
    }
    (cos, sin)
}
