//! Device tokens and one outstanding host result. Every token buffer has a FIFO completion fence.
use super::*;

pub type TokenTable = HbmTensor<i32, Chip, m![Tok256, Dummy8]>;
type Pending = (Option<resident::Step>, queue::Fence, usize, Vec<Option<usize>>);
pub(crate) type Retire<'a> = dyn FnMut(Vec<Option<usize>>) + 'a;
pub(crate) type Prefetch<'a> = dyn FnMut() -> Vec<driver::pipeline::PipelineMedia> + 'a;
#[derive(Default)]
pub(crate) struct Work<'a> {
    pub retire: Option<&'a mut Retire<'a>>,
    pub prefetch: Option<&'a mut Prefetch<'a>>,
}
pub(crate) struct State {
    pub write: usize,
    pub tables: [TokenTable; 2],
    pub(super) alt_outputs: OutputRows,
    pending: std::collections::VecDeque<Pending>,
}
impl State {
    pub async fn new(device: &mut Device) -> Result<Self, Error> {
        Ok(Self { write: 0,
            tables: [HostTensor::<i32, m![Tok256, Dummy8]>::zero().to_hbm(&mut device.pdma).await?,
                     HostTensor::<i32, m![Tok256, Dummy8]>::zero().to_hbm(&mut device.pdma).await?],
            alt_outputs: OutputRows::new(device).await?,
            pending: Default::default() })
    }
}

impl Engine {
    pub(crate) async fn pipeline_submit(&mut self, device: &mut Device, spans: &[Span<'_>], mut early: Option<&mut ahead::Early<'_>>,
        work: Work<'_>) -> Result<(), BoxError> {
        let p = &mut self.pipeline;
        assert!(p.pending.len() < 2, "host can lag by only one step");
        p.write ^= 1;
        std::mem::swap(&mut self.outputs, &mut p.alt_outputs);
        let kept = self.step_launch_with_work(device, spans, early.as_deref_mut(), work).await?;
        let prepared = match early {
            Some(early) => self.prepare_ahead(device, early).await,
            None => Ok(()),
        };
        let owners = spans.iter().map(|s| s.banned.map(|_| s.owner)).collect();
        let p = &mut self.pipeline;
        p.pending.push_back((kept, queue::fence(), p.write, owners));
        prepared
    }

    pub(crate) fn pipeline_end(&mut self) {
        self.end_step(self.steps - 1);
    }

    pub async fn pipeline_complete(&mut self, device: &mut Device) -> Result<Vec<Option<usize>>, BoxError> {
        self.pipeline.complete(device).await
    }

}

impl State {
    pub(crate) async fn retire(&mut self, device: &mut Device, retire: &mut Retire<'_>) -> Result<(), BoxError> {
        let picks = self.complete(device).await?;
        retire(picks);
        Ok(())
    }

    async fn complete(&mut self, device: &mut Device) -> Result<Vec<Option<usize>>, BoxError> {
        let p = self;
        let (held, fence, bank, owners) = p.pending.pop_front().ok_or("pipeline has no result")?;
        queue::drain_through(&fence).await?;
        // Later-step launches remain queued throughout this asynchronous PDMA read.
        let picks = if owners.iter().any(Option::is_some) {
            let tokens: HostTensor<i32, m![Tok256, Dummy8]> = p.tables[bank].to_host(&mut device.pdma).await?;
            let tokens = tokens.into_vec();
            owners.into_iter().map(|o| o.map(|o| tokens[o*Dummy8::SIZE] as usize)).collect()
        } else {
            (0..owners.len()).map(|_| None).collect()
        };
        drop(held);
        Ok(picks)
    }

}

/// Raw-bit selector buffers, in one of two banks.
pub(super) struct Exact {
    raw: HbmTensor<i16, Chip, m![Out128, Nseg, Lseg]>,
    keys: HbmTensor<i32, Chip, m![Nseg, Out128]>,
    picks: HbmTensor<i32, Chip, m![Out128]>,
    bases: HbmTensor<i32, Chip, m![Nseg]>,
    kept: Vec<HbmTensor<i32, Chip, m![Out128]>>,
}
impl Exact {
    pub(super) async fn new(device: &mut Device, logits: &HbmTensor<bf16, Chip, m![Out128, Nseg, Lseg]>) -> Result<Self, Error> {
        let raw = HostTensor::<i16, m![Out128, Nseg, Lseg]>::zero().to_hbm(&mut device.pdma).await?;
        {
            use furiosa_opt_std::runtime::DeviceSend;
            let mut b = furiosa_opt_std::runtime::Buffers::new();
            logits.bind(&mut b)?;
            uploads::alias(&raw, b.pop().expect("logits buffer"))?;
        }
        Ok(Self { raw, keys: HostTensor::<i32, m![Nseg, Out128]>::zero().to_hbm(&mut device.pdma).await?,
            picks: HostTensor::<i32, m![Out128]>::zero().to_hbm(&mut device.pdma).await?,
            bases: HostTensor::<i32, m![Nseg]>::from_vec((0..Nseg::SIZE).map(|g| (g*Lseg::SIZE) as i32)).to_hbm(&mut device.pdma).await?,
            kept: Vec::new() })
    }
}
impl OutputRows {
    pub async fn queue_exact(&mut self, device: &mut Device, bans: &[&HashSet<usize>], owners: &[usize], into: &mut TokenTable) -> Result<(), BoxError> {
        self.exact.kept.clear();
        let mut sets: Vec<HashSet<usize>> = Vec::new();
        for &set in bans {
            if !sets.iter().any(|s| s == set) { sets.push(set.clone()); }
        }
        for set in sets {
            let all_banned = (0..W::SIZE).all(|i| set.contains(&i));
            let empty = HashSet::new();
            let mask = self.mask(device, if all_banned { &empty } else { &set }).await?;
            let targets: Vec<i32> = (0..Out128::SIZE).map(|r| {
                let owner = if bans.get(r).copied() == Some(&set) { owners[r] } else { 128+r };
                (owner*Dummy8::SIZE*4) as i32
            }).collect();
            let targets = HostTensor::<i32, m![Out128]>::from_vec(targets).to_hbm(&mut device.pdma).await?;
            let exact = &mut self.exact;
            exact.kept.push(targets);
            queue::queued!(device, crate::ops_sampler::exact::segment, (&exact.raw, &self.masks[mask].1, &mut exact.keys))?;
            queue::queued!(device, crate::ops_sampler::exact::finish, (&exact.keys, &exact.bases, &mut exact.picks))?;
            queue::queued!(device, crate::ops_sampler::feedback::publish, (&exact.picks, exact.kept.last().unwrap(), &mut *into))?;
        }
        Ok(())
    }
}
