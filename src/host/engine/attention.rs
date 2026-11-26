//! Each layer runs attention over E4M3 pools. The step planner selects the DSL kernels,
//! and the attention state owns their device row spaces and table buffers.

use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;
use crate::host::engine::dsl;
use crate::host::engine::attention_geometry::{self, Family};

type BoxError = Box<dyn std::error::Error>;

pub use dsl::Plan as StepPlan;

impl StepPlan {
    /// Keep the original last writer of each emitting row's head half.
    /// Regions include padding. Walk backwards so earlier overlapping writes that
    /// the plan overwrites are dead too. No query, kernel, table or order changes.
    pub fn live_launches(&self, family: Family, emitting: &[(usize, usize)]) -> Result<Vec<bool>, String> {
        let bytes = family.row_bytes();
        let mut needed = Vec::with_capacity(2 * emitting.len());
        for &(_, row) in emitting {
            for half in 0..2 {
                let offset = *self.rows.get(2 * row + half).ok_or("emitting row absent from attention plan")?;
                if offset < 0 || offset as usize % bytes != 0 {
                    return Err("unaligned emitting attention row".into());
                }
                needed.push(offset as usize / bytes);
            }
        }
        let regions: Vec<_> = self.launches.iter().map(|l| (l.row, 2 * l.kernel.rows())).collect();
        let mut keep = vec![false; regions.len()];
        for (i, &(start, count)) in regions.iter().enumerate().rev() {
            let n = needed.len();
            needed.retain(|&r| !(start <= r && r - start < count));
            keep[i] = needed.len() != n;
        }
        if !needed.is_empty() {
            return Err("an emitting row has no attention writer".into());
        }
        Ok(keep)
    }
}

/// One family's row space: the query halves the launches read and the output halves they write.
pub struct RowSpace<HQ: AxisName> {
    pub qin: HbmTensor<bf16, Chip, m![Rq, HQ]>,
    pub out: HbmTensor<bf16, Chip, m![Rq, HQ]>,
}

impl<HQ: AxisName> RowSpace<HQ> {
    async fn new(device: &mut Device) -> Result<Self, Error> {
        Ok(Self {
            qin: HostTensor::<bf16, m![Rq, HQ]>::zero().to_hbm(&mut device.pdma).await?,
            out: HostTensor::<bf16, m![Rq, HQ]>::zero().to_hbm(&mut device.pdma).await?,
        })
    }
}

/// A family's per-step table words (a DSL plan's words), uploaded in power-of-two size classes so a step uploads
/// at most twice what it uses and pads every region a launch reads; a launch addresses its part by
/// byte offset. Each class's device buffer is allocated on its first use.
///
/// The merged upload (`AttentionState::words_m`) carries both families' words in one
/// transfer, in the finer half-octave classes (`FINE`): the Sliding words at word 0, the Full words from
/// `Merged::base_f`, each followed by at least `attention_geometry::MASK_PAD_WORDS` zero words, as in its own upload.
#[derive(Default)]
pub struct Words {
    slots: Vec<(usize, Box<dyn ClassBuf>)>,
    /// The class of the latest upload (its word count).
    pub class: usize,
}

/// One class buffer, whatever its word count.
trait ClassBuf {
    fn tensor(&self) -> &dyn DeviceSend;
    /// `bytes` (exactly the class's words) into the buffer, as a write still to be awaited.
    fn write<'a>(&'a mut self, device: &'a Device, bytes: Vec<u8>) -> crate::host::engine::uploads::Write<'a>;
}

impl<E: M + 'static> ClassBuf for HbmTensor<i32, Chip, E> {
    fn tensor(&self) -> &dyn DeviceSend {
        self
    }

    fn write<'a>(&'a mut self, device: &'a Device, bytes: Vec<u8>) -> crate::host::engine::uploads::Write<'a> {
        crate::host::engine::uploads::write(device, HostTensor::<i32, E>::from_buf(bytes), self)
    }
}

/// A zeroed class buffer of `class` words.
async fn zeroed(device: &mut Device, class: usize) -> Result<Box<dyn ClassBuf>, BoxError> {
    async fn of<E: M + 'static>(device: &mut Device) -> Result<Box<dyn ClassBuf>, BoxError> {
        let buf: HbmTensor<i32, Chip, E> = HostTensor::<i32, E>::zero().to_hbm(&mut device.pdma).await?;
        Ok(Box::new(buf))
    }
    match class {
        c if c == Iw16::SIZE => of::<m![Iw16]>(device).await,
        c if c == Iw16h::SIZE => of::<m![Iw16h]>(device).await,
        c if c == Iw17::SIZE => of::<m![Iw17]>(device).await,
        c if c == Iw17h::SIZE => of::<m![Iw17h]>(device).await,
        c if c == Iw18::SIZE => of::<m![Iw18]>(device).await,
        c if c == Iw18h::SIZE => of::<m![Iw18h]>(device).await,
        c if c == Iw19::SIZE => of::<m![Iw19]>(device).await,
        c if c == Iw19h::SIZE => of::<m![Iw19h]>(device).await,
        c if c == Iw20::SIZE => of::<m![Iw20]>(device).await,
        c if c == Iw20h::SIZE => of::<m![Iw20h]>(device).await,
        c if c == Iw21::SIZE => of::<m![Iw21]>(device).await,
        c if c == Iw21h::SIZE => of::<m![Iw21h]>(device).await,
        c if c == Iw22::SIZE => of::<m![Iw22]>(device).await,
        c if c == Iw22h::SIZE => of::<m![Iw22h]>(device).await,
        c if c == Iw23::SIZE => of::<m![Iw23]>(device).await,
        c if c == Iw23h::SIZE => of::<m![Iw23h]>(device).await,
        c if c == Iw24::SIZE => of::<m![Iw24]>(device).await,
        c => Err(format!("{c} words is no table class").into()),
    }
}

/// The merged upload's classes, smallest first: the power-of-two classes and the half-octave ones between them.
const FINE: [usize; 17] = [
    Iw16::SIZE,
    Iw16h::SIZE,
    Iw17::SIZE,
    Iw17h::SIZE,
    Iw18::SIZE,
    Iw18h::SIZE,
    Iw19::SIZE,
    Iw19h::SIZE,
    Iw20::SIZE,
    Iw20h::SIZE,
    Iw21::SIZE,
    Iw21h::SIZE,
    Iw22::SIZE,
    Iw22h::SIZE,
    Iw23::SIZE,
    Iw23h::SIZE,
    Iw24::SIZE,
];

fn put_words(bytes: &mut [u8], words: &[i32]) {
    for (dst, word) in bytes.chunks_exact_mut(4).zip(words) {
        dst.copy_from_slice(&word.to_le_bytes());
    }
}

/// Where the Full family's words start in the merged upload: past the Sliding words and their padding, on a
/// 64 KiB boundary (the Full words' regions keep their offsets modulo 64 KiB).
const MERGED_ALIGN: usize = 16384;

/// The merged upload's layout for a step: the Full family's first word and the words both families use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Merged {
    pub base_f: usize,
    pub used: usize,
}

impl Merged {
    pub fn of(plans: &[StepPlan; 2]) -> Self {
        let base_f = (plans[0].words.len() + attention_geometry::MASK_PAD_WORDS).next_multiple_of(MERGED_ALIGN);
        Self { base_f, used: base_f + plans[1].words.len() }
    }
}

/// The merged image: the Sliding words at word 0, the Full words at `base_f`, zeros elsewhere.
pub fn merged_bytes(plans: &[StepPlan; 2], merged: Merged, class: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; class * 4];
    put_words(&mut bytes, &plans[0].words);
    put_words(&mut bytes[merged.base_f * 4..], &plans[1].words);
    bytes
}

/// `pack` (`uploads`): where the step's small tables go in the merged upload: each at a 256-byte boundary (64 words, the
/// device allocator's alignment) from the first past the merged words and their mask padding. Returns each table's
/// first word and the words the upload then uses.
pub fn packed_layout(used: usize, lens: &[usize]) -> (Vec<usize>, usize) {
    let mut at = (used + attention_geometry::MASK_PAD_WORDS).next_multiple_of(64);
    let starts = lens
        .iter()
        .map(|&n| {
            let start = at;
            at = (at + n).next_multiple_of(64);
            start
        })
        .collect();
    (starts, at)
}

impl Words {
    fn slot(&self) -> Result<&dyn ClassBuf, BoxError> {
        self.slots
            .iter()
            .find(|(class, _)| *class == self.class)
            .map(|(_, buf)| &**buf)
            .ok_or_else(|| "the step's table words were not uploaded".into())
    }

    fn slot_mut(&mut self, class: usize) -> Result<&mut dyn ClassBuf, BoxError> {
        match self.slots.iter_mut().find(|(c, _)| *c == class) {
            Some((_, buf)) => Ok(&mut **buf),
            None => Err("table words: the class buffer was not placed (`ensure`)".into()),
        }
    }

    /// The current upload whole (the DSL launches address it by byte).
    pub fn tensor(&self) -> Result<&dyn DeviceSend, BoxError> {
        Ok(self.slot()?.tensor())
    }

    /// The class `used` words go into with the padding a launch's mask region may read past its last byte
    /// (`attention_geometry::MASK_PAD_WORDS`): the smallest of `classes` that holds them.
    fn class_of(used: usize, classes: &[usize]) -> Result<usize, BoxError> {
        let n = used + attention_geometry::MASK_PAD_WORDS;
        match classes.iter().find(|&&c| n <= c) {
            Some(&class) => Ok(class),
            None => Err(format!("{n} table words exceed the largest upload class").into()),
        }
    }

    /// Allocates (zeroed) the buffer of `class` words when there is none yet.
    async fn place(&mut self, device: &mut Device, class: usize) -> Result<(), BoxError> {
        if !self.slots.iter().any(|(c, _)| *c == class) {
            let buf = zeroed(device, class).await?;
            self.slots.push((class, buf));
        }
        Ok(())
    }

    /// `ensure` for a merged upload of `used` words with the small tables (`packed_layout`).
    pub async fn ensure_packed(&mut self, device: &mut Device, used: usize) -> Result<(), BoxError> {
        let class = Self::class_of(used, &FINE)?;
        self.place(device, class).await
    }

    /// Write the merged attention words and the step's small tables. Return the write and its buffer.
    pub fn write_packed<'a>(
        &'a mut self,
        device: &'a Device,
        plans: &[StepPlan; 2],
        merged: Merged,
        tables: &[Vec<i32>],
        starts: &[usize],
        used: usize,
    ) -> Result<(crate::host::engine::uploads::Write<'a>, crate::host::engine::uploads::Buffer), BoxError> {
        let class = Self::class_of(used, &FINE)?;
        let mut bytes = merged_bytes(plans, merged, class);
        for (table, &at) in tables.iter().zip(starts) {
            for (dst, word) in bytes[at * 4..(at + table.len()) * 4].chunks_exact_mut(4).zip(table) {
                dst.copy_from_slice(&word.to_le_bytes());
            }
        }
        self.class = class;
        let mut bound = furiosa_opt_std::runtime::Buffers::new();
        self.slot_mut(class)?.tensor().bind(&mut bound)?;
        let buffer = bound.pop().ok_or("the words upload binds no buffer")?;
        let pending = self.slot_mut(class)?.write(device, bytes);
        Ok((pending, buffer))
    }
}

/// One family's region in the merged table upload.
#[derive(Clone, Copy)]
pub struct WordsAt<'a> {
    pub words: &'a Words,
    pub base: usize,
}

impl<'a> WordsAt<'a> {
    /// The upload whole and the byte the family's words start at.
    pub fn tensor(&self) -> Result<(&'a dyn DeviceSend, usize), BoxError> {
        Ok((self.words.tensor()?, self.base * 4))
    }
}

/// The attention's device state: the families' row spaces, the per-step table-word uploads and the DSL kernels.
pub struct AttentionState {
    pub sliding: RowSpace<Hqs>,
    pub full: RowSpace<Hqf>,
    pub words_m: Words,
    pub merged: Merged,
    /// Next-step buffers, swapped in while queued launches hold the current buffers.
    alt_words: Words,
    /// The DSL kernels' constants and loaded kernels.
    pub dsl: dsl::Dsl,
}

impl AttentionState {
    pub async fn new(device: &mut Device) -> Result<Self, BoxError> {
        let dsl = dsl::Dsl::new(device)
            .await
            .map_err(|why| format!("DSL attention: {why}"))?;
        Ok(Self {
            sliding: RowSpace::new(device).await?,
            full: RowSpace::new(device).await?,
            words_m: Words::default(),
            merged: Merged::default(),
            alt_words: Words::default(),
            dsl,
        })
    }

    pub fn swap_words(&mut self) {
        std::mem::swap(&mut self.words_m, &mut self.alt_words);
    }

    pub fn words(&self, family: Family) -> WordsAt<'_> {
        words_at(self.merged, &self.words_m, family)
    }
}

fn words_at(merged: Merged, words: &Words, family: Family) -> WordsAt<'_> {
    let base = match family {
        Family::Sliding => 0,
        Family::Full => merged.base_f,
    };
    WordsAt { words, base }
}

/// Runs one layer's planned launches of `family` into the family's row space.
#[allow(clippy::too_many_arguments)]
pub async fn run_planned<E: M>(
    state: &mut AttentionState,
    family: Family,
    plan: &StepPlan,
    fused: usize,
    skip_first: usize,
    live: Option<&[bool]>,
    k_pool: &HbmTensor<f8e4m3, Chip, E>,
    v_pool: &HbmTensor<f8e4m3, Chip, E>,
) -> Result<(), BoxError> {
    let AttentionState { sliding, full, words_m, merged, dsl, .. } = state;
    let words = words_at(*merged, words_m, family);
    match family {
        Family::Sliding => dsl::run(dsl, family, &plan.launches, fused, skip_first, live, sliding, words, k_pool, v_pool).await,
        Family::Full => {
            let launches = if fused == 0 && skip_first == 0 {
                plan.full_standalone.as_deref().unwrap_or(&plan.launches)
            } else {
                &plan.launches
            };
            dsl::run(dsl, family, launches, fused, skip_first, live, full, words, k_pool, v_pool).await
        },
    }
}
