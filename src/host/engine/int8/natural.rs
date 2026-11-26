//! Prepare weights using the embedded calibration exponents and static FP8 scales.
use super::*;
use crate::ops_core::prep::natural as kernels;
use kernels::ScalePad;

/// LZ4-compressed signed exponents in layer-major order.
#[allow(long_running_const_eval)] pub(crate) const EXP: &[u8] = &crate::host::envelope_lookup::unpack::<1335296>(include_bytes!("int8s_exp.lz4"));

/// Each layer record stores QKV, MLP, Down, and O exponents in that order.
pub(super) struct Factors<'a> {
    pub(super) qkv: &'a [u8],
    pub(super) o: &'a [u8],
}

impl Factors<'static> {
    pub(super) fn load(layers: &[crate::host::load::Layer]) -> Result<Vec<Self>, BoxError> {
        let mut rest = EXP;
        let mut factors = Vec::with_capacity(layers.len());
        for layer in layers {
            let o_width = match layer {
                crate::host::load::Layer::Full(_) => Qf::SIZE,
                crate::host::load::Layer::Sliding(_) => Qs::SIZE,
            };
            let skip = 2 * H::SIZE + L::SIZE;
            let (row, next) = rest.split_at_checked(skip + o_width)
                .ok_or("truncated INT8 smoothing table")?;
            factors.push(Self { qkv: &row[..H::SIZE], o: &row[skip..] });
            rest = next;
        }
        if !rest.is_empty() {
            return Err("trailing bytes in INT8 smoothing table".into());
        }
        Ok(factors)
    }
}

fn pow2(e: &[u8], inverse: bool) -> Result<Vec<bf16>, BoxError> {
    e.iter().map(|&v| {
        let k = v as i8 as i32 * if inverse { -1 } else { 1 };
        let x = 2f32.powi(k);
        let b = bf16::from_f32(x);
        if !x.is_finite() || x == 0.0 || b.to_f32() != x {
            Err(format!("int8s exponent {k} is not an exact finite nonzero BF16 factor").into())
        } else { Ok(b) }
    }).collect()
}

/// Fold a power-of-two factor into norm gamma and reject values that would round.
async fn gamma(device: &mut Device, g: &HbmTensor<bf16, Chip, m![H]>, exp: &[u8]) -> Result<HbmTensor<bf16, Chip, m![Rc, H]>, BoxError> {
    if exp.len() != H::SIZE { return Err("int8s gamma exponent width".into()); }
    let old = g.to_host::<m![H]>(&mut device.pdma).await?.into_vec();
    let factors = pow2(exp, false)?;
    let mut result = Vec::with_capacity(H::SIZE);
    for (j, (g, s)) in old.into_iter().zip(factors).enumerate() {
        let x = g.to_f32() / s.to_f32();
        let b = bf16::from_f32(x);
        if !x.is_finite() || b.to_f32() != x || b.to_f32() * s.to_f32() != g.to_f32() {
            return Err(format!("int8s gamma pow2 fold loses value at channel {j}").into());
        }
        result.push(b);
    }
    Ok(up(device, copies(&result)).await?)
}

async fn factors<Q: AxisName>(device: &mut Device, exp: &[u8], inverse: bool) -> Result<HbmTensor<bf16, Chip, m![Rc, Q]>, BoxError> {
    if exp.len() != Q::SIZE { return Err("int8s smoothing width".into()); }
    Ok(up(device, copies(&pow2(exp, inverse)?)).await?)
}

async fn scales<N: AxisName>(device: &mut Device, rows: &HbmTensor<bf16, Chip, m![N]>) -> Result<HbmTensor<bf16, Chip, m![N, ScalePad]>, Error> {
    let rows = rows.to_host::<m![N]>(&mut device.pdma).await?.into_vec();
    up(device, rows.into_iter().flat_map(|v| std::iter::repeat_n(v, ScalePad::SIZE))).await
}

#[derive(Default)]
pub(super) struct Layer {
    pub(super) fp8_o: Option<Fp8O>,
    pub(super) fp8_gate_up: Option<Fp8GateUp>,
    pub(super) fp8_down: Option<Fp8Down>,
    pub(super) fp8_qkv: Option<Fp8Qkv>,
    /// Static FP8 Full Q/K carrier used only by the RSF resident bodies.
    pub(super) fp8_qk_full: Option<Fp8FullQk>,
    pub(super) o_full: Option<OFullInt8>,
    pub(super) qkv: Option<QkvInt8>,
    /// Q4 Gate/Up weights for this layer.
    pub(super) q4: Option<super::q4serve::Q4>,
}

pub(super) async fn prep_o_full(device: &mut Device, c: &Consts, s: &FullLayer, exp: &[u8]) -> Result<OFullInt8, BoxError> {
    let f = factors::<Qf>(device, exp, false).await?;
    let rs = scales(device, &s.o_weight_scale).await?;
    let (wo, ws) = launch(kernels::prep_o_full, (&mut *device, &s.o_weight, &rs, &f, &c.eye)).await?;
    let d_o = factors::<Qf>(device, exp, true).await?;
    Ok(OFullInt8 { wo, ws, d_o })
}

pub(super) async fn prep_qkv(device: &mut Device, c: &Consts, s: &SlidingLayer, exp: &[u8]) -> Result<QkvInt8, BoxError> {
    let input_signed = gamma(device, &s.input_norm, exp).await?;
    let f = factors::<H>(device, exp, false).await?;
    let rs = scales(device, &s.qkv_weight_scale).await?;
    let (wq, ws) = launch(kernels::prep_qkv, (&mut *device, &s.qkv_weight, &rs, &f, &c.eye)).await?;
    Ok(QkvInt8 { input_signed, wq, ws })
}

use crate::device::tokenwise_int8::ffn_fast::fp8c::Lp;
use crate::ops_tokenwise_int8::{SlidingOStaticF8, StaticF8FfnWeights, StaticF8SlidingQkvWeights};
pub(super) struct Fp8O {
    wo: HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    ws: HbmTensor<f32, Chip, m![H]>,
    sigma: HbmTensor<f32, Chip, m![1 # 8]>,
    unused: HbmTensor<bf16, Chip, m![Rc, Qs]>,
}
pub(super) struct Fp8Qkv {
    wq: HbmTensor<f8e4m3, Chip, m![Nq, H]>,
    ws: HbmTensor<f32, Chip, m![Nq]>,
    sigma: HbmTensor<f32, Chip, m![1 # 8]>,
    gamma: HbmTensor<bf16, Chip, m![Rc, H]>,
}
pub(super) struct Fp8FullQk {
    input_rms_weight: HbmTensor<bf16, Chip, m![Rc, H]>,
    wq: HbmTensor<f8e4m3, Chip, m![Qf, H]>,
    wsq: HbmTensor<bf16, Chip, m![Qf]>,
    wk: HbmTensor<f8e4m3, Chip, m![Pf, H]>,
    wsk: HbmTensor<bf16, Chip, m![Pf]>,
}
pub(super) struct Fp8GateUp {
    wug: HbmTensor<f8e4m3, Chip, m![Lp, H]>,
    wug_scale: HbmTensor<f32, Chip, m![Lu]>,
    sigma: HbmTensor<f32, Chip, m![1 # 8]>,
    gamma: HbmTensor<bf16, Chip, m![Rc, H]>,
}

pub(super) struct Fp8Down {
    wd: HbmTensor<f8e4m3, Chip, m![Kq4, H, Kd]>,
    wd_scale: HbmTensor<f32, Chip, m![H]>,
    sigma_down: HbmTensor<f32, Chip, m![1 # 8]>,
}
async fn fp8_sigma(device: &mut Device, s: f32) -> Result<HbmTensor<f32, Chip, m![1 # 8]>, Error> {
    up::<f32, m![1 # 8]>(device, [s; 8]).await
}
async fn fp8_rows<N: AxisName>(device: &mut Device, rows: &HbmTensor<bf16, Chip, m![N]>) -> Result<HbmTensor<f32, Chip, m![N]>, Error> {
    let values = rows.to_host::<m![N]>(&mut device.pdma).await?.into_vec();
    up(device, values.into_iter().map(|x| x.to_f32())).await
}
pub(super) async fn prep_fp8_o(device: &mut Device, s: &mut SlidingLayer, sigma: f32) -> Result<Fp8O, BoxError> {
    let wo = std::mem::replace(&mut s.o_weight, HbmTensor::new());
    let ws = fp8_rows(device, &s.o_weight_scale).await?;
    Ok(Fp8O { wo, ws, sigma: fp8_sigma(device, sigma).await?, unused: up(device, vec![bf16::from_f32(1.); Rc::SIZE * Qs::SIZE]).await? })
}
pub(super) async fn prep_fp8_qkv(device: &mut Device, s: &mut SlidingLayer, sigma: f32) -> Result<Fp8Qkv, BoxError> {
    let wq = std::mem::replace(&mut s.qkv_weight, HbmTensor::new());
    let ws = fp8_rows(device, &s.qkv_weight_scale).await?;
    let gamma = gamma(device, &s.input_norm, &[0; H::SIZE]).await?;
    Ok(Fp8Qkv { wq, ws, gamma, sigma: fp8_sigma(device, sigma).await? })
}

/// Keep the supplied E4M3 rows and BF16 scales in their existing HBM layout.
pub(super) async fn prep_fp8_qk_full(device: &mut Device, s: &mut FullLayer) -> Result<Fp8FullQk, BoxError> {
    Ok(Fp8FullQk {
        input_rms_weight: gamma(device, &s.input_norm, &[0; H::SIZE]).await?,
        wq: std::mem::replace(&mut s.q_weight, HbmTensor::new()),
        wsq: std::mem::replace(&mut s.q_weight_scale, HbmTensor::new()),
        wk: std::mem::replace(&mut s.k_weight, HbmTensor::new()),
        wsk: std::mem::replace(&mut s.k_weight_scale, HbmTensor::new()),
    })
}
/// Replicate each exact BF16 ratio to satisfy per-slice DMA alignment.
fn ratio_values(values: &[u8]) -> Vec<bf16> {
    values.iter().flat_map(|&v| std::iter::repeat_n(bf16::from_f32(0.5 + v as f32 / 32.), ScalePad::SIZE)).collect()
}

pub(super) async fn prep_fp8_gate_up(
    device: &mut Device, c: &Consts, m: &MlpWeights, pre: &HbmTensor<bf16, Chip, m![H]>,
    sigma: f32, ratios: &super::fp8_row_mse::Ratios<'_>,
) -> Result<Fp8GateUp, BoxError> {
    let gamma = gamma(device, pre, &[0; H::SIZE]).await?;
    let identity_h = factors::<H>(device, &[0; H::SIZE], false).await?;
    let gate_ratio = up::<bf16, m![L,ScalePad]>(device, ratio_values(ratios.gate)).await?;
    let up_ratio = up::<bf16, m![L,ScalePad]>(device, ratio_values(ratios.up)).await?;
    let (wug, wug_scale) = {
        use furiosa_opt_std::{__private::DeviceOutput, runtime::Buffers};
        // The kernel writes the live prefix. Zero the tail of the final layout
        // in place so the reader's padded loads stay within the allocation.
        let f = furiosa_opt_std::backend::npu::function(device, kernels::prep_ug_fp8_mse.path(), &[]).await?;
        let mut allocated = Buffers::new();
        <HbmTensor<f8e4m3, Chip, m![Lp,H]>>::alloc_into(&f, &mut allocated)?;
        let buffer = allocated.pop().ok_or("FP8 UG output allocation missing")?;
        let live = WugRows::SIZE * H::SIZE;
        let padded = Lp::SIZE * H::SIZE;
        assert_eq!(buffer.size(), padded);
        assert_eq!(padded - live, 2048 * H::SIZE);
        let mut tail: HbmTensor<f8e4m3, Chip, m![2048,H]> = buffer.slice(live..padded).into();
        HostTensor::<f8e4m3, m![2048,H]>::from_vec(vec![f8e4m3::from_f32(0.); 2048 * H::SIZE])
            .to_hbm(&mut device.pdma).output(&mut tail).await?;
        <HbmTensor<f32, Chip, WugRows>>::alloc_into(&f, &mut allocated)?;
        let scale = <HbmTensor<f32, Chip, WugRows>>::take(&mut allocated.into_iter());
        let prefix: HbmTensor<f8e4m3, Chip, m![{WugRows},H]> = buffer.slice(0..live).into();
        let mut output = (prefix, scale);
        launch(kernels::prep_ug_fp8_mse, (&mut *device, &m.up_weight_packed, &m.gate_weight_packed,
            &m.up_weight_scale, &m.gate_weight_scale, &m.up_global_scale, &m.gate_global_scale,
            &identity_h, &c.ones, &c.ones, &c.eye, &up_ratio, &gate_ratio)).output(&mut output).await?;
        (HbmTensor::<f8e4m3, Chip, m![Lp,H]>::from(buffer), output.1)
    };
    Ok(Fp8GateUp { wug, wug_scale: unsafe { wug_scale.reshape() }, sigma: fp8_sigma(device, sigma).await?, gamma })
}

pub(super) async fn prep_fp8_down(
    device: &mut Device, c: &Consts, m: &MlpWeights, sigma_down: f32, ratios: &[u8],
) -> Result<Fp8Down, BoxError> {
    let identity_l = factors::<L>(device, &[0; L::SIZE], false).await?;
    let down_ratio = up::<bf16, m![H,ScalePad]>(device, ratio_values(ratios)).await?;
    let (wd, wd_scale) = launch(kernels::prep_down_fp8_mse, (&mut *device, &m.down_weight_packed, &m.down_weight_scale,
        &m.down_global_scale, &identity_l, &c.eye, &down_ratio)).await?;
    Ok(Fp8Down { wd: unsafe { wd.reshape() }, wd_scale, sigma_down: fp8_sigma(device, sigma_down).await? })
}
impl Fp8O {
    pub(super) fn args<'a>(&'a self, post: &'a HbmTensor<bf16, Chip,m![Rc,H]>) -> SlidingOStaticF8<'a> {
        SlidingOStaticF8 { sigma: &self.sigma, wo: &self.wo, ws: &self.ws, o_signs: &self.unused, post_attn_rms_weight: post }
    }
}
impl Fp8Down {
    pub(super) fn args<'a>(
        &'a self, post: &'a HbmTensor<bf16, Chip, m![Rc, H]>, gate_up: &'a Fp8GateUp,
    ) -> StaticF8FfnWeights<'a> {
        StaticF8FfnWeights { sigma: &gate_up.sigma, sigma_down: &self.sigma_down, pre_ff_rms_weight: &gate_up.gamma,
            wug: &gate_up.wug, wug_scale: &gate_up.wug_scale, wd: &self.wd, wd_scale: &self.wd_scale, post_ff_rms_weight: post }
    }
}
impl Fp8Qkv {
    pub(super) fn args<'a>(&'a self, layer: &'a SlidingLayer, rope: &'a Rope) -> StaticF8SlidingQkvWeights<'a> {
        StaticF8SlidingQkvWeights { sigma: &self.sigma, input_rms_weight: &self.gamma, wq: &self.wq, ws: &self.ws,
            q_rms_weight: &layer.q_norm, k_rms_weight: &layer.k_norm, cos: &rope.cos_s, sin: &rope.sin_s }
    }
}
impl Fp8FullQk {
    pub(super) fn args<'a>(&'a self, next: &'a FullLayer, rope: &'a Rope) -> crate::ops_core::full_resident::FullQkvStaticF8<'a> {
        crate::ops_core::full_resident::FullQkvStaticF8 {
            input_rms_weight: &self.input_rms_weight,
            wq: &self.wq,
            wsq: &self.wsq,
            wk: &self.wk,
            wsk: &self.wsk,
            q_rms_weight: &next.q_norm,
            k_rms_weight: &next.k_norm,
            cos: &rope.cos_f,
            sin: &rope.sin_f,
        }
    }
}
