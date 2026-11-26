//! Each owner holds only its own media, until that owner is reused at prompt position zero.
//! No host buffer read by a queued encoder is overwritten. The queue owns bound HBM regions;
//! inputs can be released after launch binding, and encoded rows live through the prompt.
use super::*;
use crate::axes::MediaPool;
use crate::host::engine::{Span, buckets::{Bucket, Dense}, tables};
use crate::ops_media::PutRows;

// The preprocessing contract: 280 image patches and at most 30 s * 16000 / 640 audio frames.
const IMAGE_CAP: usize = crate::host::image::MAX_SOFT_TOKENS;
const AUDIO_CAP: usize = crate::host::audio::MAX_SAMPLES / crate::host::audio::SAMPLES_PER_TOKEN;

const _: () = assert!(MediaPool::SIZE * H::SIZE * 2 < i32::MAX as usize);

struct VisionUpload<R:AxisName> {
    pixels: HbmTensor<i8, Chip, m![R, Rv]>,
    cols: HbmTensor<i32, Chip, m![R]>,
    rows: HbmTensor<i32, Chip, m![R]>,
    dst: HbmTensor<i32, Chip, m![R]>,
}
enum VisionInput {
    R32(VisionUpload<crate::ops_audio_rows::sized32::AudioRows>),
    R64(VisionUpload<crate::ops_audio_rows::sized64::AudioRows>),
    R96(VisionUpload<crate::ops_audio_rows::sized96::AudioRows>),
    R128(VisionUpload<crate::ops_audio_rows::sized128::AudioRows>),
    R160(VisionUpload<crate::ops_audio_rows::sized160::AudioRows>),
    R192(VisionUpload<crate::ops_audio_rows::sized192::AudioRows>),
    R224(VisionUpload<crate::ops_audio_rows::sized224::AudioRows>),
    R256(VisionUpload<crate::ops_audio_rows::sized256::AudioRows>),
    R288(VisionUpload<crate::ops_audio_rows::sized288::AudioRows>),
}
struct AudioUpload<R: AxisName> {
    frames: HbmTensor<bf16, Chip, m![R, Aa]>,
    dst: HbmTensor<i32, Chip, m![R]>,
}
enum AudioInput {
    R32(AudioUpload<crate::ops_audio_rows::sized32::AudioRows>),
    R64(AudioUpload<crate::ops_audio_rows::sized64::AudioRows>),
    R96(AudioUpload<crate::ops_audio_rows::sized96::AudioRows>),
    R128(AudioUpload<crate::ops_audio_rows::sized128::AudioRows>),
    R160(AudioUpload<crate::ops_audio_rows::sized160::AudioRows>),
    R192(AudioUpload<crate::ops_audio_rows::sized192::AudioRows>),
    R224(AudioUpload<crate::ops_audio_rows::sized224::AudioRows>),
    R256(AudioUpload<crate::ops_audio_rows::sized256::AudioRows>),
    R288(AudioUpload<crate::ops_audio_rows::sized288::AudioRows>),
    R320(AudioUpload<crate::ops_audio_rows::sized320::AudioRows>),
    R352(AudioUpload<crate::ops_audio_rows::sized352::AudioRows>),
    R384(AudioUpload<crate::ops_audio_rows::sized384::AudioRows>),
    R416(AudioUpload<crate::ops_audio_rows::sized416::AudioRows>),
    R448(AudioUpload<crate::ops_audio_rows::sized448::AudioRows>),
    R480(AudioUpload<crate::ops_audio_rows::sized480::AudioRows>),
    R512(AudioUpload<crate::ops_audio_rows::sized512::AudioRows>),
    R544(AudioUpload<crate::ops_audio_rows::sized544::AudioRows>),
    R576(AudioUpload<crate::ops_audio_rows::sized576::AudioRows>),
    R608(AudioUpload<crate::ops_audio_rows::sized608::AudioRows>),
    R640(AudioUpload<crate::ops_audio_rows::sized640::AudioRows>),
    R672(AudioUpload<crate::ops_audio_rows::sized672::AudioRows>),
    R704(AudioUpload<crate::ops_audio_rows::sized704::AudioRows>),
    R736(AudioUpload<crate::ops_audio_rows::sized736::AudioRows>),
    R768(AudioUpload<crate::ops_audio_rows::sized768::AudioRows>),
}

#[derive(Default)]
struct Owner {
    id: Option<usize>,
    range: Option<(usize,usize)>,
    image_rows: usize,
    image: Option<VisionInput>,
    audio: Option<AudioInput>,
    image_encoded: bool,
    audio_encoded: bool,
}

pub(super) struct Resident {
    pool: Option<HbmTensor<bf16, Chip, m![MediaPool, H]>>,
    owners: Vec<Owner>,
    pending: std::collections::BTreeMap<usize, Owner>,
    gather: Vec<(HbmTensor<i32, Chip, m![PutRows]>, HbmTensor<i32, Chip, m![PutRows]>)>,
}
impl Resident {
    pub fn new() -> Self {
        Self { pool: None, pending: Default::default(), owners: (0..tables::OWNERS).map(|_| Owner::default()).collect(), gather: Vec::new() }
    }
    fn place(&self, rows: usize) -> Option<usize> {
        first_fit(self.owners.iter().filter_map(|o|o.range).collect(),rows,MediaPool::SIZE)
    }
    pub fn ready(&self, spans: &[Span<'_>]) -> bool {
        spans.iter().all(|s| s.inputs.iter().all(|i|matches!(i,RowInput::Token(_) | RowInput::DeviceToken(_))) || self.owners[s.owner].range.is_some())
    }
    pub async fn allocate(&mut self, device: &mut Device) -> Result<(), Error> {
        // Host-side HbmTensor::new is unplaced on the NPU backend. Place the pool
        // once during engine initialization, before accepting any requests.
        self.pool = Some(zeros(device).await?);
        Ok(())
    }
    /// Clear stale owner slots when a new request begins.
    pub fn begin(&mut self, spans: &[Span<'_>]) {
        for span in spans {
            if span.start == 0 && span.owner < self.owners.len() && self.owners[span.owner].id.is_none() {
                self.owners[span.owner] = Owner::default();
            }
        }
    }
    /// Stages arrivals during the held step, and assigns admitted requests to their owner slots.
    /// IDs are Driver submission IDs, never addresses or media hashes. Pending IDs disappear
    /// when the request leaves the driver; a reused owner always gets a fresh request state.
    pub async fn prepare_encoded(&mut self, device:&mut Device,weights:&Weights,positions:&HbmTensor<bf16,Chip,m![PvRows,Mv]>,requests:&[super::MediaRequest<'_>])->Result<(),BoxError>{
        let result=self.stage(device,Some((weights,positions)),requests).await;
        result
    }
    async fn stage(&mut self, device: &mut Device, encoding:Option<(&Weights,&HbmTensor<bf16,Chip,m![PvRows,Mv]>)>, requests: &[super::MediaRequest<'_>]) -> Result<(), BoxError> {
        self.pending.retain(|id, _| requests.iter().any(|r| r.id == *id));
        for (slot, owner) in self.owners.iter_mut().enumerate() {
            if !requests.iter().any(|r| r.owner == Some(slot) && Some(r.id) == owner.id) {
                *owner = Owner::default();
            }
        }
        for request in requests {
            if !request.stage {continue;}
            if let Some(slot) = request.owner {
                if self.owners[slot].id == Some(request.id) && self.owners[slot].range.is_some() {
                    continue;
                }
                let image_rows=request.image.map_or(0,|i|i.pixels.len());
                let rows=image_rows+request.audio.map_or(0,|a|a.frames.len());
                let Some(base)=self.place(rows) else {
                    self.owners[slot]=Owner{id:Some(request.id),..Owner::default()};
                    self.pending.remove(&request.id);
                    continue;
                };
                let mut owner = match self.pending.remove(&request.id) {
                    Some(mut owner) => {
                        // Only the tiny scatter offsets depend on admission. The large input
                        // uploads are immutable and are never sent a second time.
                        if let (Some(v), Some(image)) = (owner.image.as_mut(), request.image) {
                            match v {
                                VisionInput::R32(v)=>upload(device,offsets::<crate::ops_audio_rows::sized32::AudioRows>(image.pixels.len(),base),&mut v.dst).await?,
                                VisionInput::R64(v)=>upload(device,offsets::<crate::ops_audio_rows::sized64::AudioRows>(image.pixels.len(),base),&mut v.dst).await?,
                                VisionInput::R96(v)=>upload(device,offsets::<crate::ops_audio_rows::sized96::AudioRows>(image.pixels.len(),base),&mut v.dst).await?,
                                VisionInput::R128(v)=>upload(device,offsets::<crate::ops_audio_rows::sized128::AudioRows>(image.pixels.len(),base),&mut v.dst).await?,
                                VisionInput::R160(v)=>upload(device,offsets::<crate::ops_audio_rows::sized160::AudioRows>(image.pixels.len(),base),&mut v.dst).await?,
                                VisionInput::R192(v)=>upload(device,offsets::<crate::ops_audio_rows::sized192::AudioRows>(image.pixels.len(),base),&mut v.dst).await?,
                                VisionInput::R224(v)=>upload(device,offsets::<crate::ops_audio_rows::sized224::AudioRows>(image.pixels.len(),base),&mut v.dst).await?,
                                VisionInput::R256(v)=>upload(device,offsets::<crate::ops_audio_rows::sized256::AudioRows>(image.pixels.len(),base),&mut v.dst).await?,
                                VisionInput::R288(v)=>upload(device,offsets::<crate::ops_audio_rows::sized288::AudioRows>(image.pixels.len(),base),&mut v.dst).await?,
                            }
                        }
                        if let (Some(a), Some(audio)) = (owner.audio.as_mut(), request.audio) {
                            match a {
                                AudioInput::R32(a) => upload(device, offsets::<crate::ops_audio_rows::sized32::AudioRows>(audio.frames.len(), base + image_rows), &mut a.dst).await?,
                                AudioInput::R64(a) => upload(device, offsets::<crate::ops_audio_rows::sized64::AudioRows>(audio.frames.len(), base + image_rows), &mut a.dst).await?,
                                AudioInput::R96(a) => upload(device, offsets::<crate::ops_audio_rows::sized96::AudioRows>(audio.frames.len(), base + image_rows), &mut a.dst).await?,
                                AudioInput::R128(a) => upload(device, offsets::<crate::ops_audio_rows::sized128::AudioRows>(audio.frames.len(), base + image_rows), &mut a.dst).await?,
                                AudioInput::R160(a) => upload(device, offsets::<crate::ops_audio_rows::sized160::AudioRows>(audio.frames.len(), base + image_rows), &mut a.dst).await?,
                                AudioInput::R192(a) => upload(device, offsets::<crate::ops_audio_rows::sized192::AudioRows>(audio.frames.len(), base + image_rows), &mut a.dst).await?,
                                AudioInput::R224(a) => upload(device, offsets::<crate::ops_audio_rows::sized224::AudioRows>(audio.frames.len(), base + image_rows), &mut a.dst).await?,
                                AudioInput::R256(a) => upload(device, offsets::<crate::ops_audio_rows::sized256::AudioRows>(audio.frames.len(), base + image_rows), &mut a.dst).await?,
                                AudioInput::R288(a) => upload(device, offsets::<crate::ops_audio_rows::sized288::AudioRows>(audio.frames.len(), base + image_rows), &mut a.dst).await?,
                                AudioInput::R320(a) => upload(device, offsets::<crate::ops_audio_rows::sized320::AudioRows>(audio.frames.len(), base + image_rows), &mut a.dst).await?,
                                AudioInput::R352(a) => upload(device, offsets::<crate::ops_audio_rows::sized352::AudioRows>(audio.frames.len(), base + image_rows), &mut a.dst).await?,
                                AudioInput::R384(a) => upload(device, offsets::<crate::ops_audio_rows::sized384::AudioRows>(audio.frames.len(), base + image_rows), &mut a.dst).await?,
                                AudioInput::R416(a) => upload(device, offsets::<crate::ops_audio_rows::sized416::AudioRows>(audio.frames.len(), base + image_rows), &mut a.dst).await?,
                                AudioInput::R448(a) => upload(device, offsets::<crate::ops_audio_rows::sized448::AudioRows>(audio.frames.len(), base + image_rows), &mut a.dst).await?,
                                AudioInput::R480(a) => upload(device, offsets::<crate::ops_audio_rows::sized480::AudioRows>(audio.frames.len(), base + image_rows), &mut a.dst).await?,
                                AudioInput::R512(a) => upload(device, offsets::<crate::ops_audio_rows::sized512::AudioRows>(audio.frames.len(), base + image_rows), &mut a.dst).await?,
                                AudioInput::R544(a) => upload(device, offsets::<crate::ops_audio_rows::sized544::AudioRows>(audio.frames.len(), base + image_rows), &mut a.dst).await?,
                                AudioInput::R576(a) => upload(device, offsets::<crate::ops_audio_rows::sized576::AudioRows>(audio.frames.len(), base + image_rows), &mut a.dst).await?,
                                AudioInput::R608(a) => upload(device, offsets::<crate::ops_audio_rows::sized608::AudioRows>(audio.frames.len(), base + image_rows), &mut a.dst).await?,
                                AudioInput::R640(a) => upload(device, offsets::<crate::ops_audio_rows::sized640::AudioRows>(audio.frames.len(), base + image_rows), &mut a.dst).await?,
                                AudioInput::R672(a) => upload(device, offsets::<crate::ops_audio_rows::sized672::AudioRows>(audio.frames.len(), base + image_rows), &mut a.dst).await?,
                                AudioInput::R704(a) => upload(device, offsets::<crate::ops_audio_rows::sized704::AudioRows>(audio.frames.len(), base + image_rows), &mut a.dst).await?,
                                AudioInput::R736(a) => upload(device, offsets::<crate::ops_audio_rows::sized736::AudioRows>(audio.frames.len(), base + image_rows), &mut a.dst).await?,
                                AudioInput::R768(a) => upload(device, offsets::<crate::ops_audio_rows::sized768::AudioRows>(audio.frames.len(), base + image_rows), &mut a.dst).await?,
                            }
                        }
                        owner
                    }
                    None => prepared_owner(device, request, base).await?,
                };
                owner.id = Some(request.id);
                owner.range=Some((base,rows));owner.image_rows=image_rows;
                if let Some((weights,positions))=encoding {
                    encode_owner(device,weights,positions,self.pool.as_mut().ok_or("media pool unavailable")?,&mut owner,request.image,request.audio).await?;
                }
                self.owners[slot] = owner;
            } else if !self.pending.contains_key(&request.id) {
                let owner = prepared_owner(device, request, 0).await?;
                self.pending.insert(request.id, owner);
            }
        }
        Ok(())
    }

    pub async fn embed<B: Bucket>(
        &mut self, device: &mut Device, weights: &Weights,
        spans: &[Span<'_>], dense: &mut Dense<B::T>, logical_to_physical: &[usize],
    ) -> Result<(), BoxError> {
        let pool = self.pool.as_mut().ok_or("resident media pool was not initialized")?;
        let mut source = Vec::new();
        let mut destination = Vec::new();
        let mut step_row = 0;
        for span in spans {
            let owner = &mut self.owners[span.owner];
            let base = owner.range.map_or(0,|r|r.0);
            for input in &span.inputs {
                let row = match *input {
                    RowInput::Token(_) | RowInput::DeviceToken(_) => { step_row += 1; continue; }
                    RowInput::Image(image, k) => {
                        if image.pixels.len() > IMAGE_CAP || image.pixels.len() != image.positions.len() || k >= image.pixels.len() {
                            return Err("resident image exceeds preprocessing contract or has invalid positions/index".into());
                        }
                        assert!(owner.image_encoded,"admission queued this owner's image");
                        base + k
                    }
                    RowInput::Audio(audio, k) => {
                        if audio.frames.len() > AUDIO_CAP || k >= audio.frames.len() {
                            return Err("resident audio exceeds preprocessing contract or has invalid index".into());
                        }
                        let base = base + owner.image_rows;
                        assert!(owner.audio_encoded,"admission queued this owner's audio");
                        base + k
                    }
                };
                source.push(row_bytes(row));
                destination.push(row_bytes(logical_to_physical[step_row]));
                step_row += 1;
            }
        }
        if source.is_empty() || source.len() != destination.len() || source.len() > B::T::SIZE {
            return Err(format!("invalid resident media gather: {} sources, {} destinations in {} rows", source.len(), destination.len(), B::T::SIZE).into());
        }
        for (batch, (s, d)) in source.chunks(PutRows::SIZE).zip(destination.chunks(PutRows::SIZE)).enumerate() {
            let (mut s, mut d) = (s.to_vec(), d.to_vec());
            pad_gather(&mut s, &mut d)?;
            let source: HbmTensor<i32, Chip, m![PutRows]> = post(device, s).await?;
            let destination: HbmTensor<i32, Chip, m![PutRows]> = post(device, d).await?;
            if batch == 0 {
                B::embed_media(device, dense, weights, pool, &source, &destination).await?;
            } else {
                B::put_resident_tail(device, pool, &source, &destination, dense).await?;
            }
            self.gather.push((source, destination));
        }
        Ok(())
    }
}

async fn encode_owner(device:&mut Device,weights:&Weights,positions:&HbmTensor<bf16,Chip,m![PvRows,Mv]>,pool:&mut HbmTensor<bf16,Chip,m![MediaPool,H]>,owner:&mut Owner,image:Option<&crate::host::image::ImagePatches>,audio:Option<&crate::host::audio::AudioFrames>)->Result<(),BoxError>{
    let base=owner.range.expect("encoded owner has a range").0;
    if let Some(image)=image {
                        if !owner.image_encoded {
                            if owner.image.is_none() {
                                owner.image = Some(vision_upload(device, image, base).await?);
                            }
                            let w=&weights.vision;
                            macro_rules! encode {
                                ($op:expr,$v:ident) => { crate::host::engine::queue::queued!(device, $op,
                                    (&$v.pixels, positions, &$v.cols, &$v.rows, &w.patch_ln1_weight, &w.patch_ln1_bias,
                                     &w.patch_dense_weight, &w.patch_dense_bias, &w.patch_ln2_weight, &w.patch_ln2_bias,
                                     &w.pos_norm_weight, &w.pos_norm_bias, &w.embedding_projection_weight, &$v.dst, &mut *pool))?; };
                            }
                            match owner.image.as_ref().unwrap() {
                                VisionInput::R32(v) => { encode!(crate::ops_vision_rows::sized32::encode_u8, v); },
                                VisionInput::R64(v) => { encode!(crate::ops_vision_rows::sized64::encode_u8, v); },
                                VisionInput::R96(v) => { encode!(crate::ops_vision_rows::sized96::encode_u8, v); },
                                VisionInput::R128(v) => { encode!(crate::ops_vision_rows::sized128::encode_u8, v); },
                                VisionInput::R160(v) => { encode!(crate::ops_vision_rows::sized160::encode_u8, v); },
                                VisionInput::R192(v) => { encode!(crate::ops_vision_rows::sized192::encode_u8, v); },
                                VisionInput::R224(v) => { encode!(crate::ops_vision_rows::sized224::encode_u8, v); },
                                VisionInput::R256(v) => { encode!(crate::ops_vision_rows::sized256::encode_u8, v); },
                                VisionInput::R288(v) => { encode!(crate::ops_vision_rows::sized288::encode_u8, v); },
                            }
                            owner.image_encoded = true;
                            // queue::launch binds owned buffer regions before returning.
                            // The queued future retains these inputs through completion;
                            // the request needs only its encoded rows from this point.
                            owner.image = None;
                        }

    }
    let base=base+owner.image_rows;
    if let Some(audio)=audio {
                        if !owner.audio_encoded {
                            if owner.audio.is_none() {
                                owner.audio = Some(audio_input(device, audio, base).await?);
                            }
                            match owner.audio.as_ref().unwrap() {
                                AudioInput::R32(a) => {
                                    crate::host::engine::queue::queued!(device, crate::ops_audio_rows::sized32::encode,
                                        (&a.frames, &weights.audio.embedding_projection_weight, &a.dst, &mut *pool))?;
                                }
                                AudioInput::R64(a) => {
                                    crate::host::engine::queue::queued!(device, crate::ops_audio_rows::sized64::encode,
                                        (&a.frames, &weights.audio.embedding_projection_weight, &a.dst, &mut *pool))?;
                                }
                                AudioInput::R96(a) => {
                                    crate::host::engine::queue::queued!(device, crate::ops_audio_rows::sized96::encode,
                                        (&a.frames, &weights.audio.embedding_projection_weight, &a.dst, &mut *pool))?;
                                }
                                AudioInput::R128(a) => {
                                    crate::host::engine::queue::queued!(device, crate::ops_audio_rows::sized128::encode,
                                        (&a.frames, &weights.audio.embedding_projection_weight, &a.dst, &mut *pool))?;
                                }
                                AudioInput::R160(a) => {
                                    crate::host::engine::queue::queued!(device, crate::ops_audio_rows::sized160::encode,
                                        (&a.frames, &weights.audio.embedding_projection_weight, &a.dst, &mut *pool))?;
                                }
                                AudioInput::R192(a) => {
                                    crate::host::engine::queue::queued!(device, crate::ops_audio_rows::sized192::encode,
                                        (&a.frames, &weights.audio.embedding_projection_weight, &a.dst, &mut *pool))?;
                                }
                                AudioInput::R224(a) => {
                                    crate::host::engine::queue::queued!(device, crate::ops_audio_rows::sized224::encode,
                                        (&a.frames, &weights.audio.embedding_projection_weight, &a.dst, &mut *pool))?;
                                }
                                AudioInput::R256(a) => {
                                    crate::host::engine::queue::queued!(device, crate::ops_audio_rows::sized256::encode,
                                        (&a.frames, &weights.audio.embedding_projection_weight, &a.dst, &mut *pool))?;
                                }
                                AudioInput::R288(a) => {
                                    crate::host::engine::queue::queued!(device, crate::ops_audio_rows::sized288::encode,
                                        (&a.frames, &weights.audio.embedding_projection_weight, &a.dst, &mut *pool))?;
                                }
                                AudioInput::R320(a) => {
                                    crate::host::engine::queue::queued!(device, crate::ops_audio_rows::sized320::encode,
                                        (&a.frames, &weights.audio.embedding_projection_weight, &a.dst, &mut *pool))?;
                                }
                                AudioInput::R352(a) => {
                                    crate::host::engine::queue::queued!(device, crate::ops_audio_rows::sized352::encode,
                                        (&a.frames, &weights.audio.embedding_projection_weight, &a.dst, &mut *pool))?;
                                }
                                AudioInput::R384(a) => {
                                    crate::host::engine::queue::queued!(device, crate::ops_audio_rows::sized384::encode,
                                        (&a.frames, &weights.audio.embedding_projection_weight, &a.dst, &mut *pool))?;
                                }
                                AudioInput::R416(a) => {
                                    crate::host::engine::queue::queued!(device, crate::ops_audio_rows::sized416::encode,
                                        (&a.frames, &weights.audio.embedding_projection_weight, &a.dst, &mut *pool))?;
                                }
                                AudioInput::R448(a) => {
                                    crate::host::engine::queue::queued!(device, crate::ops_audio_rows::sized448::encode,
                                        (&a.frames, &weights.audio.embedding_projection_weight, &a.dst, &mut *pool))?;
                                }
                                AudioInput::R480(a) => {
                                    crate::host::engine::queue::queued!(device, crate::ops_audio_rows::sized480::encode,
                                        (&a.frames, &weights.audio.embedding_projection_weight, &a.dst, &mut *pool))?;
                                }
                                AudioInput::R512(a) => {
                                    crate::host::engine::queue::queued!(device, crate::ops_audio_rows::sized512::encode,
                                        (&a.frames, &weights.audio.embedding_projection_weight, &a.dst, &mut *pool))?;
                                }
                                AudioInput::R544(a) => {
                                    crate::host::engine::queue::queued!(device, crate::ops_audio_rows::sized544::encode,
                                        (&a.frames, &weights.audio.embedding_projection_weight, &a.dst, &mut *pool))?;
                                }
                                AudioInput::R576(a) => {
                                    crate::host::engine::queue::queued!(device, crate::ops_audio_rows::sized576::encode,
                                        (&a.frames, &weights.audio.embedding_projection_weight, &a.dst, &mut *pool))?;
                                }
                                AudioInput::R608(a) => {
                                    crate::host::engine::queue::queued!(device, crate::ops_audio_rows::sized608::encode,
                                        (&a.frames, &weights.audio.embedding_projection_weight, &a.dst, &mut *pool))?;
                                }
                                AudioInput::R640(a) => {
                                    crate::host::engine::queue::queued!(device, crate::ops_audio_rows::sized640::encode,
                                        (&a.frames, &weights.audio.embedding_projection_weight, &a.dst, &mut *pool))?;
                                }
                                AudioInput::R672(a) => {
                                    crate::host::engine::queue::queued!(device, crate::ops_audio_rows::sized672::encode,
                                        (&a.frames, &weights.audio.embedding_projection_weight, &a.dst, &mut *pool))?;
                                }
                                AudioInput::R704(a) => {
                                    crate::host::engine::queue::queued!(device, crate::ops_audio_rows::sized704::encode,
                                        (&a.frames, &weights.audio.embedding_projection_weight, &a.dst, &mut *pool))?;
                                }
                                AudioInput::R736(a) => {
                                    crate::host::engine::queue::queued!(device, crate::ops_audio_rows::sized736::encode,
                                        (&a.frames, &weights.audio.embedding_projection_weight, &a.dst, &mut *pool))?;
                                }
                                AudioInput::R768(a) => {
                                    crate::host::engine::queue::queued!(device, crate::ops_audio_rows::sized768::encode,
                                        (&a.frames, &weights.audio.embedding_projection_weight, &a.dst, &mut *pool))?;
                                }
                            }

                            owner.audio_encoded = true;
                            owner.audio = None;
                        }

    }
    Ok(())
}

fn first_fit(mut ranges:Vec<(usize,usize)>,rows:usize,capacity:usize)->Option<usize>{
    if rows==0 || rows>capacity {return None;}
    ranges.sort_unstable();let mut at=0;
    for (start,n) in ranges {
        if at+rows<=start {return Some(at);}
        at=at.max(start+n);
    }
    (at+rows<=capacity).then_some(at)
}

fn offsets<R: AxisName>(n: usize, base: usize) -> Vec<i32> {
    assert!(n > 0 && n <= R::SIZE);
    (0..R::SIZE).map(|r| row_bytes(base + r.min(n - 1))).collect()
}

async fn prepared_owner(device: &mut Device, r: &super::MediaRequest<'_>, base: usize) -> Result<Owner, BoxError> {
    let mut owner = Owner { id: Some(r.id), ..Owner::default() };
    if let Some(image) = r.image {
        owner.image = Some(vision_upload(device, image, base).await?);
    }
    if let Some(audio) = r.audio {
        owner.audio = Some(audio_input(device, audio, base + r.image.map_or(0,|i|i.pixels.len())).await?);
    }
    Ok(owner)
}

async fn vision_upload(device:&mut Device,image:&crate::host::image::ImagePatches,base:usize)->Result<VisionInput,BoxError>{
    if image.pixels.is_empty() || image.pixels.len()>IMAGE_CAP || image.positions.len()!=image.pixels.len() || image.pixels.iter().any(|p|p.len()!=Rv::SIZE) {return Err("invalid resident image".into());}
    Ok(match image.pixels.len().div_ceil(32) {
        1=>VisionInput::R32(vision_upload_sized(device,image,base).await?),
        2=>VisionInput::R64(vision_upload_sized(device,image,base).await?),
        3=>VisionInput::R96(vision_upload_sized(device,image,base).await?),
        4=>VisionInput::R128(vision_upload_sized(device,image,base).await?),
        5=>VisionInput::R160(vision_upload_sized(device,image,base).await?),
        6=>VisionInput::R192(vision_upload_sized(device,image,base).await?),
        7=>VisionInput::R224(vision_upload_sized(device,image,base).await?),
        8=>VisionInput::R256(vision_upload_sized(device,image,base).await?),
        9=>VisionInput::R288(vision_upload_sized(device,image,base).await?),
        _=>return Err("invalid resident image row count".into()),
    })
}
async fn vision_upload_sized<R:AxisName>(device:&mut Device,image:&crate::host::image::ImagePatches,base:usize)->Result<VisionUpload<R>,BoxError>{
    let n=image.pixels.len();assert!(n>0 && n<=R::SIZE);
    let mut cols=Vec::with_capacity(R::SIZE);let mut rows=Vec::with_capacity(R::SIZE);
    for k in 0..R::SIZE {let k=k.min(n-1);let(c,r)=crate::ops_vision_rows::position_offsets(image.positions[k].0,image.positions[k].1);cols.push(c);rows.push(r);}
    let bytes = &image.rgb8;
    assert_eq!(bytes.len(), n * Rv::SIZE, "cached pixels match the preprocessing geometry");
    let mut padded = Vec::with_capacity(R::SIZE * Rv::SIZE);
    padded.extend_from_slice(bytes);
    for _ in n..R::SIZE {
        padded.extend_from_slice(&bytes[(n - 1) * Rv::SIZE..n * Rv::SIZE]);
    }
    let pixels = post(device, padded).await?;
    Ok(VisionUpload{pixels,cols:post(device,cols).await?,rows:post(device,rows).await?,dst:post(device,offsets::<R>(n,base)).await?})
}

async fn audio_input(device: &mut Device, audio: &crate::host::audio::AudioFrames, base: usize) -> Result<AudioInput, BoxError> {
    if audio.frames.len() > AUDIO_CAP {
        return Err("resident audio exceeds preprocessing contract".into());
    }
    Ok(match audio.frames.len().div_ceil(32) {
        1 => AudioInput::R32(audio_upload(device,audio,base).await?),
        2 => AudioInput::R64(audio_upload(device,audio,base).await?),
        3 => AudioInput::R96(audio_upload(device,audio,base).await?),
        4 => AudioInput::R128(audio_upload(device,audio,base).await?),
        5 => AudioInput::R160(audio_upload(device,audio,base).await?),
        6 => AudioInput::R192(audio_upload(device,audio,base).await?),
        7 => AudioInput::R224(audio_upload(device,audio,base).await?),
        8 => AudioInput::R256(audio_upload(device,audio,base).await?),
        9 => AudioInput::R288(audio_upload(device,audio,base).await?),
        10 => AudioInput::R320(audio_upload(device,audio,base).await?),
        11 => AudioInput::R352(audio_upload(device,audio,base).await?),
        12 => AudioInput::R384(audio_upload(device,audio,base).await?),
        13 => AudioInput::R416(audio_upload(device,audio,base).await?),
        14 => AudioInput::R448(audio_upload(device,audio,base).await?),
        15 => AudioInput::R480(audio_upload(device,audio,base).await?),
        16 => AudioInput::R512(audio_upload(device,audio,base).await?),
        17 => AudioInput::R544(audio_upload(device,audio,base).await?),
        18 => AudioInput::R576(audio_upload(device,audio,base).await?),
        19 => AudioInput::R608(audio_upload(device,audio,base).await?),
        20 => AudioInput::R640(audio_upload(device,audio,base).await?),
        21 => AudioInput::R672(audio_upload(device,audio,base).await?),
        22 => AudioInput::R704(audio_upload(device,audio,base).await?),
        23 => AudioInput::R736(audio_upload(device,audio,base).await?),
        24 => AudioInput::R768(audio_upload(device,audio,base).await?),
        _ => return Err("invalid resident audio row count".into()),
    })
}

async fn post<D: MaterializableScalar, E: M>(device: &mut Device, values: Vec<D>) -> Result<HbmTensor<D, Chip, E>, Error> {
    let result = HostTensor::<D, E>::from_vec(values).to_hbm(&mut device.pdma).await;
    result
}

async fn audio_upload<R: AxisName>(device: &mut Device, audio: &crate::host::audio::AudioFrames, base: usize) -> Result<AudioUpload<R>, BoxError> {
    let n = audio.frames.len();
    if n == 0 || n > R::SIZE || audio.frames.iter().any(|f| f.len() != Aa::SIZE) {
        return Err("invalid resident audio frames".into());
    }
    let frames = (0..R::SIZE).flat_map(|r| audio.frames[r.min(n - 1)].iter().copied()).collect();
    let destination = (0..R::SIZE).map(|r| row_bytes(base + r.min(n - 1))).collect();
    Ok(AudioUpload { frames: post(device, frames).await?, dst: post(device, destination).await? })
}

fn pad_gather(source: &mut Vec<i32>, destination: &mut Vec<i32>) -> Result<(), BoxError> {
    if source.is_empty() || source.len() != destination.len() || source.len() > PutRows::SIZE {
        return Err("invalid resident media gather".into());
    }
    source.resize(PutRows::SIZE, *source.last().unwrap());
    destination.resize(PutRows::SIZE, *destination.last().unwrap());
    Ok(())
}
