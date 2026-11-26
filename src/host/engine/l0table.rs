//! Build the full vocabulary table from the supplied checkpoint before serving.
use crate::ops_l0table::{post::prepare_norm_t1024, prepare_project_t1024, serve::*};
use crate::device::tokenwise_int8::qkv::Nq;
use crate::{Chip, EMBED_SCALE, EPS, axes::*};
use super::queue;
use crate::host::load::{checked_tensor, e4m3_lut, embedding, sliding_projection, upload};
use safetensors::Dtype;
use furiosa_opt_std::prelude::*;
use std::{sync::{Arc, mpsc}, thread};

type BoxError = Box<dyn std::error::Error>;
mod inputs;
mod route;
pub(crate) use route::MediaSource;

pub(super) struct Table {
    pub(super) media: route::MediaScratch,
    q: HbmTensor<bf16, Chip, m![W, Qs]>,
    k: HbmTensor<bf16, Chip, m![W, Ps]>,
    v: HbmTensor<f8e4m3, Chip, m![W, Ps]>,
}
impl Table {
    pub(super) fn args(&self) -> Tables<'_> {
        Tables { q: &self.q, k: &self.k, v: &self.v }
    }
    pub(super) async fn build(device: &mut Device, checkpoint: Arc<memmap2::Mmap>) -> Result<Self, BoxError> {
        let prep = Preparation::start(checkpoint)?;
        let queueing = queue::queueing();
        queue::set_enabled(true);
        let result = prep.finish(device).await;
        let drained = queue::drain().await;
        queue::set_enabled(queueing);
        let table = result?;
        drained?;
        Ok(table)
    }
}

struct WeightRows {
    w: Vec<bf16>,
    scale: Vec<bf16>,
    qg: Vec<bf16>,
    kg: Vec<bf16>,
}
/// One chunk's split3 terms as packed BF16 byte images (little-endian, element order: what
/// `HostTensor::from_vec` would pack), so the upload wraps them with `from_buf` without a copy.
struct Chunk {
    first: usize,
    hi: Vec<u8>,
    mid: Vec<u8>,
    lo: Vec<u8>,
}
/// Byte buffers handed back after their upload, so the worker refills resident pages instead of
/// faulting in fresh ones every chunk.
type Buffers = [Vec<u8>; 3];
/// Input lanes: lane k computes chunks k, k + LANES, ... on its own long-lived thread (one chunk per
/// thread, no per-chunk spawn or join), and the device loop takes chunk c from lane c % LANES, in order.
const MAX_LANES: usize = 7;
/// One lane's ends: its finished chunks (in its own order) and the buffers handed back to it.
type Lane = (mpsc::SyncSender<Result<Chunk, String>>, mpsc::Receiver<Buffers>);

struct Preparation {
    rx: mpsc::Receiver<Result<WeightRows, String>>,
    chunks: Vec<mpsc::Receiver<Result<Chunk, String>>>,
    free: Vec<mpsc::Sender<Buffers>>,
    worker: thread::JoinHandle<()>,
}
impl Preparation {
    fn start(checkpoint: Arc<memmap2::Mmap>) -> Result<Self, BoxError> {
        let (tx, rx) = mpsc::sync_channel(1);
        let (mut chunks, mut free, mut lanes) = (Vec::new(), Vec::new(), Vec::new());
        let workers = thread::available_parallelism().map_or(2, |n| n.get()).saturating_sub(1).clamp(1, MAX_LANES);
        for _ in 0..workers {
            let (chunk_tx, chunk_rx) = mpsc::sync_channel(1);
            let (free_tx, free_rx) = mpsc::channel();
            chunks.push(chunk_rx);
            free.push(free_tx);
            lanes.push((chunk_tx, free_rx));
        }
        let worker = thread::Builder::new()
            .name("l0table-inputs".into())
            .spawn(move || {
                if let Err(e) = input_worker(&checkpoint, &tx, lanes) {
                    let _ = tx.send(Err(e));
                }
            })?;
        Ok(Self { rx, chunks, free, worker })
    }
    /// Chunk `chunk` from its lane; a closed lane reports the worker's own error when it sent one.
    fn chunk(&self, chunk: usize) -> Result<Chunk, BoxError> {
        match self.chunks[chunk % self.chunks.len()].recv() {
            Ok(c) => c.map_err(Into::into),
            Err(e) => Err(match self.rx.try_recv() {
                Ok(Err(why)) => why,
                _ => format!("layer0 preparation lane {}: {e}", chunk % self.chunks.len()),
            }
            .into()),
        }
    }
    fn recv(&self) -> Result<WeightRows, BoxError> {
        self.rx
            .recv()
            .map_err(|e| format!("layer0 preparation channel: {e}"))?
            .map_err(Into::into)
    }

    async fn finish(self, device: &mut Device) -> Result<Table, BoxError> {
        let WeightRows { w, scale, qg, kg } = self.recv()?;
        let w = upload(device, w).await?;
        let ws = upload(device, scale).await?;
        let qg = upload(device, qg).await?;
        let kg = upload(device, kg).await?;
        let (tq, tk, tv) = allocate_table(device).await?;
        let mut table = Table { q: tq, k: tk, v: tv, media: route::MediaScratch::new(device).await? };
        let mut raw = HostTensor::<f32, m![Tok1024, Nq]>::zero().to_hbm(&mut device.pdma).await?;
        let mut q = HostTensor::<bf16, m![Tok1024, Qs]>::zero().to_hbm(&mut device.pdma).await?;
        let mut k = HostTensor::<bf16, m![Tok1024, Ps]>::zero().to_hbm(&mut device.pdma).await?;
        let mut v = HostTensor::<f8e4m3, m![Tok1024, Ps]>::zero().to_hbm(&mut device.pdma).await?;
        // Two input sets: chunk i's uploads go into set i % 2 while the device still runs chunk i - 1's
        // launches from the other set (queued; the device runs them in submission order, so `raw` and the
        // staged q/k/v pass between them as in the awaited form). A set is rewritten only after the fence
        // behind its last reader (`publish_t1024`, two chunks back) is answered.
        let mut sets = [Inputs::new(), Inputs::new()];
        for chunk in 0..(W::SIZE / 1024) {
            let c = self.chunk(chunk)?;
            if c.first != chunk * 1024 {
                return Err(format!("layer0 chunk {} != {}", c.first, chunk * 1024).into());
            }
            let set = &mut sets[chunk % 2];
            if let Some(read) = set.read.take() {
                queue::drain_through(&read).await?;
            }
            // The packed images go up as they are (`from_buf`: no repacking copy); once each awaited
            // transfer is done the bytes are free, and go back to the worker for a later chunk.
            let mut recycled = [c.hi, c.mid, c.lo];
            for (bytes, dst) in recycled.iter_mut().zip([&mut set.hi, &mut set.mid, &mut set.lo]) {
                let host = HostTensor::<bf16, m![Tok1024, H]>::from_buf(std::mem::take(bytes));
                host.to_hbm(&mut device.pdma).output(dst).await?;
                *bytes = host.into_inner().into_buf();
            }
            let _ = self.free[chunk % self.chunks.len()].send(recycled);
            for (stride, dst) in [(8192usize, &mut set.qi), (4096, &mut set.ki), (2048, &mut set.vi)] {
                let offsets = (c.first..c.first + 1024)
                    .map(|i| {
                        i.checked_mul(stride)
                            .and_then(|x| i32::try_from(x).ok())
                            .ok_or("layer0 table offset overflow")
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                HostTensor::<i32, m![Tok1024]>::from_vec(offsets)
                    .to_hbm(&mut device.pdma)
                    .output(dst)
                    .await?;
            }
            queue::queued!(device, prepare_project_t1024, (&set.hi, &set.mid, &set.lo, &w, &ws, &mut raw))?;
            queue::queued!(device, prepare_norm_t1024, (&raw, &qg, &kg, &mut q, &mut k, &mut v))?;
            queue::queued!(device, publish_t1024,
                (&q, &k, &v, &set.qi, &set.ki, &set.vi, &mut table.q, &mut table.k, &mut table.v))?;
            set.read = Some(queue::fence());
        }
        queue::drain().await?;
        self.worker
            .join()
            .map_err(|_| "layer0 input worker panicked")?;
        Ok(table)
    }
}

/// One chunk's device inputs: the split3 terms and the table row offsets, with the fence behind the last
/// queued launch that reads them.
struct Inputs {
    hi: HbmTensor<bf16, Chip, m![Tok1024, H]>,
    mid: HbmTensor<bf16, Chip, m![Tok1024, H]>,
    lo: HbmTensor<bf16, Chip, m![Tok1024, H]>,
    qi: HbmTensor<i32, Chip, m![Tok1024]>,
    ki: HbmTensor<i32, Chip, m![Tok1024]>,
    vi: HbmTensor<i32, Chip, m![Tok1024]>,
    read: Option<queue::Fence>,
}
impl Inputs {
    fn new() -> Self {
        Self { hi: HbmTensor::new(), mid: HbmTensor::new(), lo: HbmTensor::new(),
            qi: HbmTensor::new(), ki: HbmTensor::new(), vi: HbmTensor::new(), read: None }
    }
}

/// Allocate and zero the tables on the device, uploading one zero tile per table.
async fn allocate_table(device: &mut Device) -> Result<(
    HbmTensor<bf16, Chip, m![W, Qs]>,
    HbmTensor<bf16, Chip, m![W, Ps]>,
    HbmTensor<f8e4m3, Chip, m![W, Ps]>,
), BoxError> {
    let zq = HostTensor::<bf16, m![Tok1024, Qs]>::zero().to_hbm(&mut device.pdma).await?;
    let zk = HostTensor::<bf16, m![Tok1024, Ps]>::zero().to_hbm(&mut device.pdma).await?;
    let zv = HostTensor::<f8e4m3, m![Tok1024, Ps]>::zero().to_hbm(&mut device.pdma).await?;
    Ok(launch(allocate_t1024, (&mut *device, &zq, &zk, &zv)).await?)
}

fn decode_bf16(bytes: &[u8]) -> impl Iterator<Item = bf16> + '_ {
    bytes.chunks_exact(2).map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])))
}

fn input_worker(
    checkpoint: &[u8],
    tx: &mpsc::SyncSender<Result<WeightRows, String>>,
    lanes: Vec<Lane>,
) -> Result<(), String> {
    let ts = safetensors::SafeTensors::deserialize(checkpoint).map_err(|e| e.to_string())?;
    let tensor = |name: &str, dtype, shape: &[usize]| {
        checked_tensor(&ts, name, dtype, shape).map_err(|e| e.to_string())
    };
    let bf = |name: &str, width| -> Result<Vec<bf16>, String> {
        let view = tensor(name, Dtype::BF16, &[width])?;
        Ok(decode_bf16(view.data()).collect())
    };
    let lut = e4m3_lut().map(bf16::from_f32);
    let mut w = Vec::with_capacity(Nq::SIZE * H::SIZE);
    let mut scale = Vec::with_capacity(Nq::SIZE);
    let prefix = "model.language_model.layers.0";
    for (part, rows) in [("q", Qs::SIZE), ("k", Ps::SIZE), ("v", Ps::SIZE)] {
        let (weight, rows) = sliding_projection(&ts, prefix, part, rows).map_err(|e| e.to_string())?;
        if weight.data().iter().any(|&c| c & 127 == 127) {
            return Err(format!("nonfinite layer-0 {part} projection weight"));
        }
        w.extend(weight.data().iter().map(|&c| lut[c as usize]));
        scale.extend(decode_bf16(rows.data()));
    }
    let qg = bf(&format!("{prefix}.self_attn.q_norm.weight"), Ds::SIZE)?;
    let kg = bf(&format!("{prefix}.self_attn.k_norm.weight"), Ds::SIZE)?;
    tx.send(Ok(WeightRows { w, scale, qg, kg }))
        .map_err(|e| e.to_string())?;
    let gamma: Vec<f32> = bf(&format!("{prefix}.input_layernorm.weight"), H::SIZE)?
        .into_iter().map(|g| g.to_f32()).collect();
    let emb = embedding(&ts).map_err(|e| e.to_string())?;
    let (emb, gamma) = (emb.data(), &gamma);
    let workers = lanes.len();
    thread::scope(|scope| {
        for (lane, (chunks, recycled)) in lanes.into_iter().enumerate() {
            scope.spawn(move || {
                for chunk in (lane..W::SIZE / 1024).step_by(workers) {
                    let first = chunk * 1024;
                    let [mut hi, mut mid, mut lo] = recycled.try_recv().unwrap_or_else(|_| {
                        let n = 1024 * 3840 * 2;
                        [vec![0u8; n], vec![0u8; n], vec![0u8; n]]
                    });
                    let made = inputs::chunk_into(emb, gamma, first, EMBED_SCALE, EPS, [&mut hi, &mut mid, &mut lo])
                        .map(|()| Chunk { first, hi, mid, lo });
                    let failed = made.is_err();
                    if chunks.send(made).is_err() || failed {
                        return;
                    }
                }
            });
        }
    });
    Ok(())
}
