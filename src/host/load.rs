use std::fs::File;
use std::path::{Path, PathBuf};

use furiosa_opt_std::prelude::*;

use crate::axes::{Aa, Df, Ds, Dummy2, H, L, Mv, Ov, Pf, Ps, Pv, Qf, Qs, Rv, W};
use crate::device::tokenwise_int8::qkv::Nq;
use crate::{Chip, EMBED_SCALE, LAYERS};

use safetensors::{Dtype, SafeTensors, tensor::TensorView};

type BoxError = Box<dyn std::error::Error>;

pub(crate) struct MlpWeights {
    pub up_weight_packed: HbmTensor<f4e2m1, Chip, m![L, H]>,
    pub gate_weight_packed: HbmTensor<f4e2m1, Chip, m![L, H]>,
    pub down_weight_packed: HbmTensor<f4e2m1, Chip, m![H, L]>,
    pub up_weight_scale: HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    pub gate_weight_scale: HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    pub down_weight_scale: HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    pub up_global_scale: HbmTensor<f32, Chip, m![1]>,
    pub gate_global_scale: HbmTensor<f32, Chip, m![1]>,
    pub down_global_scale: HbmTensor<f32, Chip, m![1]>,
}

pub(crate) struct SlidingLayer {
    pub input_norm: HbmTensor<bf16, Chip, m![H]>,
    pub qkv_weight: HbmTensor<f8e4m3, Chip, m![Nq, H]>,
    pub qkv_weight_scale: HbmTensor<bf16, Chip, m![Nq]>,
    pub q_norm: HbmTensor<bf16, Chip, m![Ds]>,
    pub k_norm: HbmTensor<bf16, Chip, m![Ds]>,
    pub post_attention_norm: HbmTensor<bf16, Chip, m![H]>,
    pub o_weight: HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    pub o_weight_scale: HbmTensor<bf16, Chip, m![H]>,
    pub pre_feedforward_norm: HbmTensor<bf16, Chip, m![H]>,
    pub post_feedforward_norm: HbmTensor<bf16, Chip, m![H]>,
    pub layer_scalar: HbmTensor<bf16, Chip, m![1 # 8]>,
    pub mlp: MlpWeights,
}

pub(crate) struct FullLayer {
    pub input_norm: HbmTensor<bf16, Chip, m![H]>,
    pub q_weight: HbmTensor<f8e4m3, Chip, m![Qf, H]>,
    pub k_weight: HbmTensor<f8e4m3, Chip, m![Pf, H]>,
    pub q_weight_scale: HbmTensor<bf16, Chip, m![Qf]>,
    pub k_weight_scale: HbmTensor<bf16, Chip, m![Pf]>,
    pub q_norm: HbmTensor<bf16, Chip, m![Df]>,
    pub k_norm: HbmTensor<bf16, Chip, m![Df]>,
    pub post_attention_norm: HbmTensor<bf16, Chip, m![H]>,
    pub o_weight: HbmTensor<f8e4m3, Chip, m![H, Qf]>,
    pub o_weight_scale: HbmTensor<bf16, Chip, m![H]>,
    pub pre_feedforward_norm: HbmTensor<bf16, Chip, m![H]>,
    pub post_feedforward_norm: HbmTensor<bf16, Chip, m![H]>,
    pub layer_scalar: HbmTensor<bf16, Chip, m![1 # 8]>,
    pub mlp: MlpWeights,
}

pub(crate) enum Layer {
    Sliding(SlidingLayer),
    Full(FullLayer),
}

pub(crate) struct VisionWeights {
    pub patch_ln1_weight: HbmTensor<bf16, Chip, m![Rv]>,
    pub patch_ln1_bias: HbmTensor<bf16, Chip, m![Rv]>,
    pub patch_dense_weight: HbmTensor<bf16, Chip, m![Mv, Rv]>,
    pub patch_dense_bias: HbmTensor<bf16, Chip, m![Mv]>,
    pub patch_ln2_weight: HbmTensor<bf16, Chip, m![Mv]>,
    pub patch_ln2_bias: HbmTensor<bf16, Chip, m![Mv]>,
    pub pos_norm_weight: HbmTensor<bf16, Chip, m![Mv]>,
    pub pos_norm_bias: HbmTensor<bf16, Chip, m![Mv]>,
    pub embedding_projection_weight: HbmTensor<bf16, Chip, m![H, Ov]>,
    pub pos_embedding: Vec<bf16>,
}

pub(crate) struct AudioWeights {
    pub embedding_projection_weight: HbmTensor<bf16, Chip, m![H, Aa]>,
}

pub struct Weights {
    /// The tied embedding rows after scaling and rounding to BF16.
    /// All embedding consumers gather this pre-scaled table; the engine's W4 LM head
    /// is prepared independently from the source safetensors, so no second HBM copy is
    /// required.
    pub(crate) embedding_table: HbmTensor<bf16, Chip, m![W, H]>,
    pub(crate) final_norm: HbmTensor<bf16, Chip, m![H]>,
    pub(crate) layers: Vec<Layer>,
    pub(crate) vision: VisionWeights,
    pub(crate) audio: AudioWeights,
}

pub(crate) async fn upload<D: MaterializableScalar, E: M>(
    device: &mut Device,
    values: impl IntoIterator<Item = D>,
) -> Result<HbmTensor<D, Chip, E>, Error> {
    HostTensor::<D, E>::from_vec(values).to_hbm(&mut device.pdma).await
}

fn host_tensor<D: MaterializableScalar + ScalarBytes, E: M>(
    tensors: &SafeTensors<'_>,
    name: &str,
) -> Result<HostTensor<D, E>, BoxError> {
    HostTensor::from_safetensors(&tensors.tensor(name)?).map_err(|error| format!("{name}: {error}").into())
}

async fn load<D: MaterializableScalar + ScalarBytes, E: M>(
    device: &mut Device,
    tensors: &SafeTensors<'_>,
    name: &str,
) -> Result<HbmTensor<D, Chip, E>, BoxError> {
    Ok(host_tensor::<D, E>(tensors, name)?.to_hbm(&mut device.pdma).await?)
}

pub(crate) fn checked_tensor<'a>(
    tensors: &SafeTensors<'a>,
    name: &str,
    dtype: Dtype,
    shape: &[usize],
) -> Result<TensorView<'a>, BoxError> {
    let view = tensors.tensor(name)?;
    if view.dtype() != dtype || view.shape() != shape {
        return Err(format!(
            "{name} is {:?} {:?}, expected {dtype:?} {shape:?}",
            view.dtype(), view.shape(),
        ).into());
    }
    Ok(view)
}

pub(crate) fn embedding<'a>(tensors: &SafeTensors<'a>) -> Result<TensorView<'a>, BoxError> {
    checked_tensor(tensors, "model.language_model.embed_tokens.weight", Dtype::BF16, &[W::SIZE, H::SIZE])
}

/// Lookup for finite E4M3 codes. Callers check the two NaN encodings.
pub(crate) fn e4m3_lut() -> [f32; 256] {
    std::array::from_fn(|code| {
        let exponent = (code >> 3) & 15;
        let mantissa = (code & 7) as f32;
        let magnitude = if exponent == 0 {
            mantissa / 512.
        } else {
            (1. + mantissa / 8.) * 2f32.powi(exponent as i32 - 7)
        };
        if code & 128 == 0 { magnitude } else { -magnitude }
    })
}

/// Match the vector unit's input denormal flushing and canonical NaN.
fn scale_embedding_rows(bytes: &mut [u8]) {
    for row in bytes.chunks_exact_mut(2) {
        let bits = u16::from_le_bytes([row[0], row[1]]);
        let scaled = if bits & 0x7f80 == 0 {
            bits & 0x8000
        } else if bits & 0x7fff > 0x7f80 {
            0x7fc0
        } else {
            bf16::from_f32(bf16::from_bits(bits).to_f32() * EMBED_SCALE).to_bits()
        };
        row.copy_from_slice(&scaled.to_le_bytes());
    }
}

async fn load_embedding_scaled(
    device: &mut Device,
    tensors: &SafeTensors<'_>,
) -> Result<HbmTensor<bf16, Chip, m![W, H]>, BoxError> {
    let mut bytes = embedding(tensors)?.data().to_vec();
    scale_embedding_rows(&mut bytes);
    Ok(HostTensor::<bf16, m![W, H]>::from_buf(bytes).to_hbm(&mut device.pdma).await?)
}

async fn load_scalar_bf16(
    device: &mut Device,
    tensors: &SafeTensors<'_>,
    name: &str,
) -> Result<HbmTensor<bf16, Chip, m![1 # 8]>, BoxError> {
    let value = host_tensor::<bf16, m![1]>(tensors, name)?.into_vec()[0];
    Ok(upload::<bf16, m![1 # 8]>(device, [value; 8]).await?)
}

async fn load_packed<D: MaterializableScalar, E: M>(
    device: &mut Device,
    tensors: &SafeTensors<'_>,
    name: &str,
    dtype: Dtype,
    shape: &[usize],
) -> Result<HbmTensor<D, Chip, E>, BoxError> {
    let view = checked_tensor(tensors, name, dtype, shape)?;
    Ok(HostTensor::<D, E>::from_buf(view.data().to_vec()).to_hbm(&mut device.pdma).await?)
}

async fn load_bf16_column<E: M>(
    device: &mut Device,
    tensors: &SafeTensors<'_>,
    name: &str,
) -> Result<HbmTensor<bf16, Chip, E>, BoxError> {
    load_packed(device, tensors, name, Dtype::BF16, &[E::SIZE, 1]).await
}

pub(crate) fn sliding_projection<'a>(
    tensors: &SafeTensors<'a>, prefix: &str, part: &str, rows: usize,
) -> Result<(TensorView<'a>, TensorView<'a>), BoxError> {
    let proj = format!("{prefix}.self_attn.{part}_proj");
    Ok((
        checked_tensor(tensors, &format!("{proj}.weight"), Dtype::F8_E4M3, &[rows, H::SIZE])?,
        checked_tensor(tensors, &format!("{proj}.weight_scale"), Dtype::BF16, &[rows, 1])?,
    ))
}

async fn load_sliding_qkv(
    device: &mut Device,
    tensors: &SafeTensors<'_>,
    prefix: &str,
) -> Result<(HbmTensor<f8e4m3, Chip, m![Nq, H]>, HbmTensor<bf16, Chip, m![Nq]>), BoxError> {
    let mut codes = Vec::with_capacity(Nq::SIZE * H::SIZE);
    let mut scales = Vec::with_capacity(Nq::SIZE * 2);
    for (part, rows) in [("q", Qs::SIZE), ("k", Ps::SIZE), ("v", Ps::SIZE)] {
        let (weight, scale) = sliding_projection(tensors, prefix, part, rows)?;
        codes.extend_from_slice(weight.data());
        scales.extend_from_slice(scale.data());
    }
    Ok((
        HostTensor::<f8e4m3, m![Nq, H]>::from_buf(codes).to_hbm(&mut device.pdma).await?,
        HostTensor::<bf16, m![Nq]>::from_buf(scales).to_hbm(&mut device.pdma).await?,
    ))
}

fn read_f32(tensors: &SafeTensors<'_>, name: &str) -> Result<f32, BoxError> {
    let view = tensors.tensor(name)?;
    if view.dtype() != safetensors::Dtype::F32 || view.data().len() != 4 {
        return Err(format!("{name} is {:?}, not a scalar f32 tensor", view.dtype()).into());
    }
    Ok(f32::from_le_bytes(view.data().try_into()?))
}

pub(crate) fn reciprocal_global_scale(
    tensors: &SafeTensors<'_>,
    name: &str,
) -> Result<f32, BoxError> {
    let value = read_f32(tensors, name)?;
    if value == 0.0 || !value.is_finite() {
        return Err(format!("invalid NVFP4 global scale {name}: {value}").into());
    }
    Ok(1.0 / value)
}

async fn load_global_scale(
    device: &mut Device,
    tensors: &SafeTensors<'_>,
    name: &str,
) -> Result<HbmTensor<f32, Chip, m![1]>, BoxError> {
    let value = reciprocal_global_scale(tensors, name)?;
    Ok(upload(device, [value]).await?)
}

pub fn model_dir() -> Result<PathBuf, BoxError> {
    let root = std::env::var_os("RNGD_MODEL_DIR")
        .ok_or("RNGD_MODEL_DIR is not set; point it at the model directory or its parent")?;
    let root = PathBuf::from(root);
    let candidates = [
        root.clone(),
        root.join("Gemma4-12B-it"),
        root.join("gemma-4-12b-it-NVFP4"),
    ];
    candidates
        .into_iter()
        .find(|p| p.join("model.safetensors").is_file())
        .ok_or_else(|| format!("no model.safetensors found under {}", root.display()).into())
}

async fn load_mlp_projection<E: M, Es: M>(
    device: &mut Device,
    tensors: &SafeTensors<'_>,
    prefix: &str,
    name: &str,
    shape: [usize; 2],
) -> Result<
    (
        HbmTensor<f4e2m1, Chip, E>,
        HbmTensor<f8e4m3, Chip, Es>,
        HbmTensor<f32, Chip, m![1]>,
    ),
    BoxError,
> {
    Ok((
        load_packed(device, tensors, &format!("{prefix}.{name}.weight_packed"), Dtype::U8, &[shape[0], shape[1] / 2]).await?,
        load(device, tensors, &format!("{prefix}.{name}.weight_scale")).await?,
        load_global_scale(device, tensors, &format!("{prefix}.{name}.weight_global_scale")).await?,
    ))
}

async fn load_mlp(
    device: &mut Device,
    tensors: &SafeTensors<'_>,
    prefix: &str,
    uses_q4: bool,
) -> Result<MlpWeights, BoxError> {
    let mut gate_up = async |name| {
        if uses_q4 {
            Ok((HbmTensor::new(), HbmTensor::new(), HbmTensor::new()))
        } else {
            load_mlp_projection::<m![L, H], m![L, H / 16]>(device, tensors, prefix, name, [L::SIZE, H::SIZE]).await
        }
    };
    let (up_weight_packed, up_weight_scale, up_global_scale) = gate_up("up_proj").await?;
    let (gate_weight_packed, gate_weight_scale, gate_global_scale) = gate_up("gate_proj").await?;
    let (down_weight_packed, down_weight_scale, down_global_scale) =
        load_mlp_projection::<m![H, L], m![H, L / 16]>(device, tensors, prefix, "down_proj", [H::SIZE, L::SIZE])
            .await?;
    Ok(MlpWeights {
        up_weight_packed,
        gate_weight_packed,
        down_weight_packed,
        up_weight_scale,
        gate_weight_scale,
        down_weight_scale,
        up_global_scale,
        gate_global_scale,
        down_global_scale,
    })
}

async fn load_common(
    device: &mut Device,
    tensors: &SafeTensors<'_>,
    prefix: &str,
    layer: usize,
) -> Result<
    (
        HbmTensor<bf16, Chip, m![H]>,
        HbmTensor<bf16, Chip, m![H]>,
        HbmTensor<bf16, Chip, m![H]>,
        MlpWeights,
    ),
    BoxError,
> {
    Ok((
        load(device, tensors, &format!("{prefix}.pre_feedforward_layernorm.weight")).await?,
        load(device, tensors, &format!("{prefix}.post_feedforward_layernorm.weight")).await?,
        load(device, tensors, &format!("{prefix}.input_layernorm.weight")).await?,
        load_mlp(device, tensors, &format!("{prefix}.mlp"),
            crate::host::engine::int8::q4serve::LAYERS.contains(&layer)).await?,
    ))
}

pub(crate) async fn load_sliding(
    device: &mut Device,
    tensors: &SafeTensors<'_>,
    layer: usize,
) -> Result<SlidingLayer, BoxError> {
    let prefix = format!("model.language_model.layers.{layer}");
    let (pre_feedforward_norm, post_feedforward_norm, input_norm, mlp) = load_common(device, tensors, &prefix, layer).await?;
    let (qkv_weight, qkv_weight_scale) = load_sliding_qkv(device, tensors, &prefix).await?;
    Ok(SlidingLayer {
        input_norm,
        qkv_weight,
        qkv_weight_scale,
        q_norm: load(device, tensors, &format!("{prefix}.self_attn.q_norm.weight")).await?,
        k_norm: load(device, tensors, &format!("{prefix}.self_attn.k_norm.weight")).await?,
        post_attention_norm: load(device, tensors, &format!("{prefix}.post_attention_layernorm.weight")).await?,
        o_weight: load(device, tensors, &format!("{prefix}.self_attn.o_proj.weight")).await?,
        o_weight_scale: load_bf16_column(device, tensors, &format!("{prefix}.self_attn.o_proj.weight_scale")).await?,
        pre_feedforward_norm,
        post_feedforward_norm,
        layer_scalar: load_scalar_bf16(device, tensors, &format!("{prefix}.layer_scalar")).await?,
        mlp,
    })
}

pub(crate) async fn load_full(
    device: &mut Device,
    tensors: &SafeTensors<'_>,
    layer: usize,
) -> Result<FullLayer, BoxError> {
    let prefix = format!("model.language_model.layers.{layer}");
    let (pre_feedforward_norm, post_feedforward_norm, input_norm, mlp) = load_common(device, tensors, &prefix, layer).await?;
    Ok(FullLayer {
        input_norm,
        q_weight: load(device, tensors, &format!("{prefix}.self_attn.q_proj.weight")).await?,
        k_weight: load(device, tensors, &format!("{prefix}.self_attn.k_proj.weight")).await?,
        q_weight_scale: load_bf16_column(device, tensors, &format!("{prefix}.self_attn.q_proj.weight_scale")).await?,
        k_weight_scale: load_bf16_column(device, tensors, &format!("{prefix}.self_attn.k_proj.weight_scale")).await?,
        q_norm: load(device, tensors, &format!("{prefix}.self_attn.q_norm.weight")).await?,
        k_norm: load(device, tensors, &format!("{prefix}.self_attn.k_norm.weight")).await?,
        post_attention_norm: load(device, tensors, &format!("{prefix}.post_attention_layernorm.weight")).await?,
        o_weight: load(device, tensors, &format!("{prefix}.self_attn.o_proj.weight")).await?,
        o_weight_scale: load_bf16_column(device, tensors, &format!("{prefix}.self_attn.o_proj.weight_scale")).await?,
        pre_feedforward_norm,
        post_feedforward_norm,
        layer_scalar: load_scalar_bf16(device, tensors, &format!("{prefix}.layer_scalar")).await?,
        mlp,
    })
}

async fn load_vision(
    device: &mut Device,
    tensors: &SafeTensors<'_>,
) -> Result<VisionWeights, BoxError> {
    let pos_embedding_host = host_tensor::<bf16, m![Pv, Dummy2, Mv]>(tensors, "model.vision_embedder.pos_embedding")?;

    Ok(VisionWeights {
        patch_ln1_weight: load(device, tensors, "model.vision_embedder.patch_ln1.weight").await?,
        patch_ln1_bias: load(device, tensors, "model.vision_embedder.patch_ln1.bias").await?,
        patch_dense_weight: load(device, tensors, "model.vision_embedder.patch_dense.weight").await?,
        patch_dense_bias: load(device, tensors, "model.vision_embedder.patch_dense.bias").await?,
        patch_ln2_weight: load(device, tensors, "model.vision_embedder.patch_ln2.weight").await?,
        patch_ln2_bias: load(device, tensors, "model.vision_embedder.patch_ln2.bias").await?,
        pos_norm_weight: load(device, tensors, "model.vision_embedder.pos_norm.weight").await?,
        pos_norm_bias: load(device, tensors, "model.vision_embedder.pos_norm.bias").await?,
        embedding_projection_weight: load(device, tensors, "model.embed_vision.embedding_projection.weight")
            .await?,
        pos_embedding: pos_embedding_host.into_vec(),
    })
}

async fn load_audio(
    device: &mut Device,
    tensors: &SafeTensors<'_>,
) -> Result<AudioWeights, BoxError> {
    Ok(AudioWeights {
        embedding_projection_weight: load(device, tensors, "model.embed_audio.embedding_projection.weight")
            .await?,
    })
}

pub(crate) fn map_checkpoint() -> Result<memmap2::Mmap, BoxError> {
    let file = File::open(model_dir()?.join("model.safetensors"))?;
    Ok(unsafe { memmap2::Mmap::map(&file)? })
}

pub(crate) async fn load_weights(device: &mut Device, checkpoint: &[u8]) -> Result<Weights, BoxError> {
    let tensors = safetensors::SafeTensors::deserialize(checkpoint)?;

    let embedding_table = load_embedding_scaled(device, &tensors).await?;
    let final_norm = load(device, &tensors, "model.language_model.norm.weight").await?;
    let vision = load_vision(device, &tensors).await?;
    let audio = load_audio(device, &tensors).await?;

    let mut layers = Vec::with_capacity(LAYERS);
    for layer in 0..LAYERS {
        if layer % 6 == 5 {
            layers.push(Layer::Full(load_full(device, &tensors, layer).await?));
        } else {
            layers.push(Layer::Sliding(load_sliding(device, &tensors, layer).await?));
        }
    }

    Ok(Weights {
        embedding_table,
        final_norm,
        layers,
        vision,
        audio,
    })
}

pub fn tokenizer_path_in(model_dir: &Path) -> PathBuf {
    model_dir.join("tokenizer.json")
}
