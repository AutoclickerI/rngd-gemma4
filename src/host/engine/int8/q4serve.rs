//! Startup preparation for 4-bit Gate/Up on the 39 layers whose next attention is Sliding.
//! Odd-grid weights and scales are packed in the natural Down basis with smoothed,
//! signed pre-FFN gamma.
//! The host prepares the Q4 weights on a separate thread during device weight preparation.
use super::*;
use crate::device::tokenwise_int8::ffn_fast::ug4::{Hb, Xc2};
use crate::ops_core::q4serve::Q4FfnWeights;
use crate::ops_tokenwise_int8::StaticF8FfnWeights;

/// Layers whose Sliding → Sliding or Full → Sliding segment uses Q4 Gate/Up.
pub(crate) const LAYERS: &[usize] = &[0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45];

/// A Q4 layer's packed rows on the host.
pub(super) struct Host {
    codes: Vec<i8>,
    scales: Vec<f32>,
}

mod packed;

/// A Q4 layer's device tensors.
pub(crate) struct Q4 {
    pre: HbmTensor<bf16, Chip, m![Rc, H]>,
    codes: HbmTensor<i8, Chip, m![Lu, Hb]>,
    scales: HbmTensor<f32, Chip, m![Xc2, Lu]>,
}

/// Uploads layer `l`'s packed rows and its folded gamma (`q4cal::fold_gamma`, the Mlp signs, `Rc` copies).
pub(super) async fn upload(device: &mut Device, l: usize, h: Host, pre: &HbmTensor<bf16, Chip, m![H]>) -> Result<Q4, BoxError> {
    let gamma: HostTensor<bf16, m![H]> = pre.to_host(&mut device.pdma).await?;
    let folded = q4cal::fold_gamma(l, &gamma.into_vec())?;
    let pre = up(device, copies(&signed(&folded, &signs(l, Site::Mlp, H::SIZE)))).await?;
    let codes = up(device, h.codes).await?;
    let scales = up(device, h.scales).await?;
    Ok(Q4 { pre, codes, scales })
}

impl Q4 {
    /// The static-FP8 FFN arguments followed by the three Q4 tensors.
    pub(crate) fn args<'a>(&'a self, f: &StaticF8FfnWeights<'a>) -> Q4FfnWeights<'a> {
        Q4FfnWeights {
            sigma_down: f.sigma_down,
            sigma: f.sigma,
            pre_ff_rms_weight: f.pre_ff_rms_weight,
            wug: f.wug,
            wug_scale: f.wug_scale,
            wd: f.wd,
            wd_scale: f.wd_scale,
            post_ff_rms_weight: f.post_ff_rms_weight,
            pre_signed: &self.pre,
            codes: &self.codes,
            scales: &self.scales,
        }
    }

}

/// Pack Q4 rows while the checkpoint tensors upload.
pub(crate) struct Prefetch(Option<std::thread::JoinHandle<Result<Vec<Host>, String>>>);

impl Prefetch {
    pub(crate) fn start(checkpoint: std::sync::Arc<memmap2::Mmap>) -> Self {
        Self(Some(std::thread::spawn(move || {
            let tensors = safetensors::SafeTensors::deserialize(&checkpoint).map_err(|e| e.to_string())?;
            let h60 = paley(60);
            let lut = crate::host::load::e4m3_lut();
            LAYERS.iter().map(|&layer| packed::direct(&tensors, layer, &h60, &lut)).collect()
        })))
    }

    pub(super) fn finish(mut self, jobs: &[usize]) -> Result<Vec<Host>, String> {
        if jobs != LAYERS {
            return Err("Q4 preparation requires the 39 Sliding-bound Gemma layers".into());
        }
        self.0.take().unwrap().join().map_err(|_| "Q4 preparation thread panicked".to_string())?
    }
}

impl Drop for Prefetch {
    fn drop(&mut self) {
        if let Some(handle) = self.0.take() {
            let _ = handle.join();
        }
    }
}
