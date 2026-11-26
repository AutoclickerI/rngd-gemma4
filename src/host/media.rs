//! Request-local media uploads, encoding and gathering.
//!
//! `resident` uploads each request once when admitted (waiting arrivals can upload while a
//! held step runs), queues one encoder per complete image/clip, and keeps its soft tokens in
//! disjoint HBM row ranges, released when the prompt finishes. The token-embedding launch gathers only this step's media rows.
//! Both encoders round to 32-row groups and retain two contraction tiles.
//! No encoder output, input upload, or request identity is shared between requests.
//!
//! The per-step fallback encodes soft tokens into a staging tensor, then copies
//! their rows into `x` with `put_rows`.

use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Aa, H, Mv, Rv, Tok1024, Tok1536};
use crate::host::engine::RowInput;
use crate::host::load::Weights;
use crate::ops_audio_rows::{AudioBatch, AudioRows};
use crate::ops_media::PutRows;
use crate::host::engine::buckets::{Bucket, Dense};
use crate::ops_vision_rows::{PvRows, VisionBatch, VisionRows};

mod resident;

pub(crate) struct MediaRequest<'a> {
    pub id: usize,
    pub owner: Option<usize>,
    pub stage:bool,
    pub image: Option<&'a crate::host::image::ImagePatches>,
    pub audio: Option<&'a crate::host::audio::AudioFrames>,
}

type BoxError = Box<dyn std::error::Error>;

/// Device inputs of the batched encoders, allocated once and overwritten per launch.
struct Inputs {
    pixels: HbmTensor<bf16, Chip, m![VisionRows, Rv]>,
    cols: HbmTensor<i32, Chip, m![VisionRows]>,
    rows: HbmTensor<i32, Chip, m![VisionRows]>,
    vision_dst: HbmTensor<i32, Chip, m![VisionRows]>,
    frames: HbmTensor<bf16, Chip, m![AudioRows, Aa]>,
    audio_dst: HbmTensor<i32, Chip, m![AudioRows]>,
    put_index: HbmTensor<i32, Chip, m![PutRows]>,
    put_source: HbmTensor<i32, Chip, m![PutRows]>,
}

pub struct MediaEncoder {
    resident: resident::Resident,
    /// The `[Pv, 2, Mv]` position embedding as `[PvRows, Mv]` rows (column `c` at row `2c`,
    /// row `r` at row `2r + 1`).
    /// Uploaded from `Weights` at the first image row (`MediaRows::new` has no weights).
    pos_table: Option<HbmTensor<bf16, Chip, m![PvRows, Mv]>>,
    /// Encoder output by step row, copied into the step's `x` by `put_rows`.
    staged: HbmTensor<bf16, Chip, m![Tok1024, H]>,
    inputs: Inputs,
}

async fn zeros<D: MaterializableScalar + num_traits::Zero, E: M>(device: &mut Device) -> Result<HbmTensor<D, Chip, E>, Error> {
    HostTensor::<D, E>::zero().to_hbm(&mut device.pdma).await
}

async fn upload<D: MaterializableScalar, E: M>(device: &mut Device, values: Vec<D>, into: &mut HbmTensor<D, Chip, E>) -> Result<(), Error> {
    HostTensor::<D, E>::from_vec(values).to_hbm(&mut device.pdma).output(into).await
}

/// Byte offset of step row `i` in a `[_, H]` tensor.
fn row_bytes(i: usize) -> i32 {
    (i * H::SIZE * 2) as i32
}

/// One media row of a step: its step row and its patch (pixels, `(col, row)`) or frame.
#[derive(Clone, Copy)]
enum MediaRow<'a> {
    Image(usize, &'a [bf16], (usize, usize)),
    Audio(usize, &'a [bf16]),
}

impl MediaEncoder {

    /// The encoder's device buffers, including the 240 MiB resident pool. The pool's
    /// one-time placement upload occurs before requests; no media step allocates it.
    pub async fn new(device: &mut Device) -> Result<Self, Error> {
        let mut resident = resident::Resident::new();
        resident.allocate(device).await?;
        Ok(Self {
            resident,
            pos_table: None,
            staged: zeros(device).await?,
            inputs: Inputs {
                pixels: zeros(device).await?,
                cols: zeros(device).await?,
                rows: zeros(device).await?,
                vision_dst: zeros(device).await?,
                frames: zeros(device).await?,
                audio_dst: zeros(device).await?,
                put_index: zeros(device).await?,
                put_source: zeros(device).await?,
            },
        })
    }

    pub(crate) async fn prepare(&mut self, device: &mut Device, weights:&Weights, requests: &[MediaRequest<'_>]) -> Result<(), BoxError> {
        self.load_positions(device,weights).await?;
        self.resident.prepare_encoded(device,weights,self.pos_table.as_ref().unwrap(),requests).await
    }

    pub(crate) fn ready(&self, spans: &[crate::host::engine::Span<'_>]) -> bool { self.resident.ready(spans) }

    pub(crate) fn begin(&mut self, spans: &[crate::host::engine::Span<'_>]) {
        self.resident.begin(spans);
    }

    pub(crate) async fn embed_resident<B: crate::host::engine::buckets::Bucket>(
        &mut self, device: &mut Device, weights: &Weights,
        spans: &[crate::host::engine::Span<'_>], dense: &mut crate::host::engine::buckets::Dense<B::T>, logical_to_physical: &[usize],
    ) -> Result<(), BoxError> {
        self.load_positions(device, weights).await?;
        self.resident.embed::<B>(device, weights, spans, dense, logical_to_physical).await
    }

    /// Uploads the position-embedding table (once).
    pub async fn load_positions(&mut self, device: &mut Device, weights: &Weights) -> Result<(), BoxError> {
        if self.pos_table.is_none() {
            let table = weights.vision.pos_embedding.clone();
            if table.len() != PvRows::SIZE * Mv::SIZE {
                return Err(format!("position embedding has {} values, expected {}", table.len(), PvRows::SIZE * Mv::SIZE).into());
            }
            self.pos_table = Some(HostTensor::<bf16, m![PvRows, Mv]>::from_vec(table).to_hbm(&mut device.pdma).await?);
        }
        Ok(())
    }

    /// Writes the soft token of every media row of `inputs` (the step's rows in order) into its
    /// row of `x`; text rows of `x` keep their contents. The encoders write a staging tensor by
    /// step row and one `put_rows` launch copies the media rows into `x` (any bucket).
    pub(crate) async fn embed<B: Bucket>(
        &mut self,
        device: &mut Device,
        weights: &Weights,
        inputs: &[RowInput<'_>],
        dense: &mut Dense<B::T>,
        logical_to_physical: Option<&[usize]>,
    ) -> Result<(), BoxError> {
        let mut images: Vec<MediaRow> = Vec::new();
        let mut audios: Vec<MediaRow> = Vec::new();
        for (logical, input) in inputs.iter().enumerate() {
            let i = logical_to_physical.map_or(logical, |rows| rows[logical]);
            match *input {
                RowInput::Token(_) | RowInput::DeviceToken(_) => {}
                RowInput::Image(image, k) => images.push(MediaRow::Image(i, &image.pixels[k], image.positions[k])),
                RowInput::Audio(audio, k) => audios.push(MediaRow::Audio(i, &audio.frames[k])),
            }
        }
        if images.is_empty() && audios.is_empty() {
            return Ok(());
        }
        if inputs.len() > B::T::SIZE || B::T::SIZE > Tok1536::SIZE {
            return Err(format!("{} step rows in a {}-row bucket (media rows support up to {})", inputs.len(), B::T::SIZE, Tok1536::SIZE).into());
        }
        if !images.is_empty() {
            self.load_positions(device, weights).await?;
        }
        let Self { pos_table, staged, inputs: buffers, .. } = self;
        if B::T::SIZE > Tok1024::SIZE {
            for (rows, image) in [(&images, true), (&audios, false)] {
                for chunk in rows.chunks(PutRows::SIZE) {
                    let staged_rows: Vec<_> = chunk.iter().enumerate().map(|(i, row)| match *row {
                        MediaRow::Image(_, pixels, position) => MediaRow::Image(i, pixels, position),
                        MediaRow::Audio(_, frame) => MediaRow::Audio(i, frame),
                    }).collect();
                    let (images, audios) = if image { (staged_rows.as_slice(), &[][..]) } else { (&[][..], staged_rows.as_slice()) };
                    encode(device, weights, pos_table.as_ref(), buffers, images, audios, staged).await?;
                    let mut source: Vec<_> = (0..chunk.len()).map(row_bytes).collect();
                    let mut destination: Vec<_> = chunk.iter().map(|r| row_bytes(r.row())).collect();
                    source.resize(PutRows::SIZE, source[chunk.len() - 1]);
                    destination.resize(PutRows::SIZE, destination[chunk.len() - 1]);
                    upload(device, source, &mut buffers.put_source).await?;
                    upload(device, destination, &mut buffers.put_index).await?;
                    B::put_media(device, staged, &buffers.put_source, &buffers.put_index, dense).await?;
                }
            }
            return Ok(());
        }
        encode(device, weights, pos_table.as_ref(), buffers, &images, &audios, staged).await?;
        let mut index: Vec<i32> = images.iter().chain(&audios).map(|r| row_bytes(r.row())).collect();
        let last = *index.last().expect("a media row");
        index.resize(PutRows::SIZE, last);
        upload(device, index, &mut buffers.put_index).await?;
        B::put_media(device, staged, &buffers.put_source, &buffers.put_index, dense).await?;
        Ok(())
    }
}

impl MediaRow<'_> {
    fn row(&self) -> usize {
        match *self {
            MediaRow::Image(row, ..) | MediaRow::Audio(row, _) => row,
        }
    }
}

/// The encoder launches of one step into `out` (`[Tok1024, H]`, rows by step row).
#[allow(clippy::too_many_arguments)]
async fn encode(
    device: &mut Device,
    weights: &Weights,
    pos_table: Option<&HbmTensor<bf16, Chip, m![PvRows, Mv]>>,
    buffers: &mut Inputs,
    images: &[MediaRow<'_>],
    audios: &[MediaRow<'_>],
    out: &mut HbmTensor<bf16, Chip, m![Tok1024, H]>,
) -> Result<(), BoxError> {
    let v = &weights.vision;
    for chunk in images.chunks(VisionRows::SIZE) {
        let pos_table = pos_table.ok_or("the position table is not loaded")?;
        let batch = VisionBatch::assemble(chunk.iter().map(|r| match *r {
            MediaRow::Image(row, pixels, position) => (pixels, position, row_bytes(row)),
            MediaRow::Audio(..) => unreachable!("image rows only"),
        }))
        .map_err(|e| format!("vision batch: {e}"))?;
        upload(device, batch.pixels, &mut buffers.pixels).await?;
        upload(device, batch.col_offsets, &mut buffers.cols).await?;
        upload(device, batch.row_offsets, &mut buffers.rows).await?;
        upload(device, batch.destination, &mut buffers.vision_dst).await?;
        crate::host::prebuilt::launch_kernel!(
            plain,
            device,
            ops_vision_rows::vision_rows,
            (
                &buffers.pixels,
                pos_table,
                &buffers.cols,
                &buffers.rows,
                &v.patch_ln1_weight,
                &v.patch_ln1_bias,
                &v.patch_dense_weight,
                &v.patch_dense_bias,
                &v.patch_ln2_weight,
                &v.patch_ln2_bias,
                &v.pos_norm_weight,
                &v.pos_norm_bias,
                &v.embedding_projection_weight,
                &buffers.vision_dst,
                &mut *out,
            ),
        )?;
    }
    for chunk in audios.chunks(AudioRows::SIZE) {
        let batch = AudioBatch::assemble(chunk.iter().map(|r| match *r {
            MediaRow::Audio(row, frame) => (frame, row_bytes(row)),
            MediaRow::Image(..) => unreachable!("audio rows only"),
        }))
        .map_err(|e| format!("audio batch: {e}"))?;
        upload(device, batch.frames, &mut buffers.frames).await?;
        upload(device, batch.destination, &mut buffers.audio_dst).await?;
        crate::host::prebuilt::launch_kernel!(
            plain,
            device,
            ops_audio_rows::audio_rows,
            (&buffers.frames, &weights.audio.embedding_projection_weight, &buffers.audio_dst, &mut *out),
        )?;
    }
    Ok(())
}
