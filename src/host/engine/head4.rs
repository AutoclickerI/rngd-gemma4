//! W4 head supplied-weight preparation for the core and in-family heads.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::*};
use crate::device::tokenwise_int8::{ffn_fast::ug4::Hb,rotate::{Ob,Kb,H60Hbm,Sgi,Sl8}};
use super::int8::{copies, eye, fwht64, h60_trf_order, paley, signs_from_seed, sylvester};
use crate::host::load::{embedding, upload};
const K: usize = H::SIZE;
struct Rot {
    h60: Vec<f32>,
    d: Vec<f32>,
    balance: Vec<f32>,
}
impl Rot {
    fn new() -> Self {
        let d = signs_from_seed(0x7461_696c_6335, K);
        let balance = BALANCE_EXP.iter().map(|e| 2.0f32.powi((*e as i8) as i32)).collect();
        Self { h60: paley(60), d, balance }
    }
    fn forward(&self, u: &mut [f32; K], y: &mut [f32; K]) {
        for a in 0..60 { fwht64(&mut u[a * 64..(a + 1) * 64]); }
        y.iter_mut().for_each(|v| *v = 0.0);
        for ap in 0..60 {
            for a in 0..60 {
                let h = self.h60[ap * 60 + a];
                for b in 0..64 { y[ap * 64 + b] += h * u[a * 64 + b]; }
            }
        }
        let n = 1.0 / (K as f32).sqrt();
        y.iter_mut().for_each(|v| *v *= n);
    }

}
fn odd4(z: f32) -> f32 {
    let h = z / 2.0;
    if h >= 7.0 { return 15.0; }
    if h < -7.0 { return -15.0; }
    if h.is_nan() { return (2.0 * h.floor() + 1.0).clamp(-15.0, 15.0); }
    // Here h lies in [-7,7), so truncation is exact in i32. Subtract one
    // exactly when a negative fractional part was truncated toward zero.
    let n = h as i32;
    (2 * (n - i32::from(h < n as f32)) + 1) as f32
}

// The balance diagonal D satisfies D^4 = a/b for the calibration moments and
// weight columns. Frozen exponents and clipping indices reconstruct W*D at
// startup. The signed final norm folds in the inverse D.
const BALANCE_EXP: &[u8; K] = include_bytes!("head4_balance_exponents.i8");
const CLIP_INDICES: &[u8; 262144] = include_bytes!("head4_clip_indices.u8");

/// Reconstruct the fixed transformed-weight choice; code/scale geometry is unchanged.
fn quantize_row_index(
    rot: &Rot, row: &[u8], out: &mut [u8], index: u8, u: &mut [f32; K], y: &mut [f32; K],
) -> bf16 {
    assert_eq!(row.len(), 2 * K);
    assert_eq!(out.len(), K / 2);
    assert!(index < 36);
    for (i, bytes) in row.chunks_exact(2).enumerate() {
        let value = bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32();
        u[i] = (value * rot.balance[i]) * rot.d[i];
    }
    rot.forward(u, y);
    let amax = y.iter().fold(0f32, |m,v| m.max(v.abs()));
    if amax == 0.0 { out.fill(0); return bf16::from_f32(0.0); }
    let ratio = (30 + 2 * index as usize) as f32 / 100.0;
    let scale = bf16::from_f32(ratio * amax / 15.0);
    let sf = scale.to_f32();
    assert!(sf > 0.0 && sf.is_finite());
    out.fill(0);
    for a in 0..60 { for b in 0..64 {
        let c = (odd4(y[64*a+b] / sf) as i32 - 1) / 2;
        assert!((-8..=7).contains(&c));
        let i = (b/32*60+a)*32+b%32;
        let byte = (i/16)*8+i%8;
        let shift = if i%16<8 { 0 } else { 4 };
        out[byte] |= ((c&15) as u8) << shift;
    }}
    scale
}

fn balanced_signed_gamma(rot: &Rot, gamma: &[bf16]) -> Vec<bf16> {
    assert_eq!(gamma.len(), K);
    (0..K).map(|i| {
        let value = gamma[i].to_f32() / rot.balance[i];
        let scaled = bf16::from_f32(value);
        assert!(value.is_finite());
        assert_eq!(scaled.to_f32(), value);
        if rot.d[i] < 0.0 { bf16::from_bits(scaled.to_bits() ^ 0x8000) } else { scaled }
    }).collect()
}

pub(crate) type HostHead=(Vec<u8>,Vec<bf16>);

fn prepare_host(checkpoint: &[u8]) -> Result<HostHead, String> {
    let tensors=safetensors::SafeTensors::deserialize(checkpoint).map_err(|e|e.to_string())?;
    let view = embedding(&tensors).map_err(|e| e.to_string())?;
    let bytes=view.data();
    let mut codes=vec![0u8;W::SIZE*Hb::SIZE];let mut scales=vec![bf16::from_f32(0.0);W::SIZE];
    let threads=std::thread::available_parallelism().map_or(2,|n|n.get()).saturating_sub(1).clamp(1,7);
    quantize_rows(bytes,&mut codes,&mut scales,threads);
    Ok((codes,scales))
}

fn quantize_rows(bytes: &[u8], codes: &mut [u8], scales: &mut [bf16], threads: usize) {
    assert_eq!(codes.len(), scales.len() * K / 2);
    assert_eq!(bytes.len(), scales.len() * 2 * K);
    let per = scales.len().div_ceil(threads);
    let rot = Rot::new();
    std::thread::scope(|scope| {
        for (part, (codes, scales)) in codes.chunks_mut(per * K / 2).zip(scales.chunks_mut(per)).enumerate() {
            let rot = &rot;
            scope.spawn(move || {
                let (mut u, mut y) = ([0.; K], [0.; K]);
                for (r, (codes, scale)) in codes.chunks_exact_mut(K / 2).zip(scales).enumerate() {
                    let row = part * per + r;
                    *scale = quantize_row_index(
                        rot, &bytes[row * 2 * K..(row + 1) * 2 * K], codes, CLIP_INDICES[row], &mut u, &mut y,
                    );
                }
            });
        }
    });
}

pub(crate) struct Head {
    pub codes:HbmTensor<i8,Chip,m![W,Hb]>,pub scales:HbmTensor<bf16,Chip,m![W]>,
    pub signed_final_rms:HbmTensor<bf16,Chip,m![H]>,
    pub h64:HbmTensor<bf16,Chip,m![Rc,Ob,Kb]>,pub h60:HbmTensor<bf16,Chip,m![Rc,{H60Hbm}]>,
    pub eye:HbmTensor<f32,Chip,m![Sgi,Sl8]>,
}
impl Head {
    pub(crate) async fn upload(device:&mut Device,(codes,scales):HostHead,gamma:&HbmTensor<bf16,Chip,m![H]>)->Result<Self,Error> {
        let rot=Rot::new();let gamma=gamma.to_host::<m![H]>(&mut device.pdma).await?.into_vec();
        let signed = balanced_signed_gamma(&rot, &gamma);
        let h64: Vec<_> = sylvester(64).into_iter().map(bf16::from_f32).collect();
        let h60 = h60_trf_order(&rot.h60);
        Ok(Self {
            codes:HostTensor::<i8,m![W,Hb]>::from_buf(codes).to_hbm(&mut device.pdma).await?,
            scales:upload(device, scales).await?,
            signed_final_rms:upload(device, signed).await?,
            h64:upload(device, copies(&h64)).await?,
            h60:upload(device, copies(&h60)).await?,
            eye: upload(device, eye()).await?,
        })
    }
}

/// Join on every path; startup errors never leave an unowned weight worker.
pub(crate) struct Preparation(Option<std::thread::JoinHandle<Result<HostHead,String>>>);
impl Preparation {
    pub(crate) fn start(checkpoint: std::sync::Arc<memmap2::Mmap>)->Result<Self,std::io::Error> {
        Ok(Self(Some(std::thread::Builder::new().name("w4-head-prep".into()).spawn(move||prepare_host(&checkpoint))?)))
    }
    pub(crate) fn finish(mut self)->Result<HostHead,String> {
        self.0.take().unwrap().join().map_err(|_|"W4 head preparation panicked".to_string())?
    }
}
impl Drop for Preparation {fn drop(&mut self){if let Some(handle)=self.0.take(){let _=handle.join();}}}
