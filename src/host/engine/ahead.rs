//! Prepare the next step's attention plans and tables while the device runs the current step.
//! The driver predicts the next spans before reading the current picks. Preparing runs in two
//! pieces: first plan the predicted spans, then upload their tables to a second set of buffers.
//! The queue can make time for these pieces between layers. Unfinished work runs after the
//! current step queues its token selection.
//!
//! The next step reuses the prepared buffers when its routing inputs match `TableKey`. It
//! uploads token IDs separately when they differ. A change in active requests or row packing
//! makes the engine plan and upload tables for the actual step.

use furiosa_opt_std::prelude::*;

use crate::host::engine::attention::{self, Merged, StepPlan};
use crate::host::engine::buckets::{B128, B1024, B1536, Bucket, DenseSet};
use crate::host::engine::tables::{self, AttnRow};
use crate::host::engine::{Engine, dsl, plan_row_attention, post_attention, queue, resident, schedule};

type BoxError = Box<dyn std::error::Error>;

/// One request's rows in the step after the running one, as `Driver::predict` expects them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AheadSpan {
    pub owner: usize,
    pub pages: Vec<usize>,
    pub start: usize,
    /// Per row, its embedding-table row (`RowInput::token`); `None` for the decode row whose token is the running
    /// step's pick.
    pub tokens: Vec<Option<usize>>,
    pub decode: bool,
    /// Its last row consumes logits (decode or a prompt chunk completing its prompt).
    pub emits: bool,
}

/// Routing inputs that must match before the engine reuses prepared tables.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TableKey {
    bucket: usize,
    rows: Vec<(Option<usize>, usize, usize, usize, bool)>,
    pages: Vec<Vec<usize>>,
    floor: bool,
    placement: Option<&'static [[u8; 3]]>,
    row_order: Vec<usize>,
}

impl TableKey {
    pub fn with_row_order(mut self, order: &schedule::RowOrder) -> Self {
        self.row_order = order.logical_to_physical.clone();
        self
    }

    pub fn of(bucket: usize, attn: &[AttnRow<'_>], prompt: &[bool], floor: bool, placement: Option<&'static [[u8; 3]]>) -> Self {
        let rows = attn.iter().zip(prompt).map(|(r, &p)| (r.owner, r.position, r.first_position, r.first_row, p)).collect();
        let pages = attn.iter().enumerate().filter(|&(i, r)| i == 0 || r.first_row != attn[i - 1].first_row).map(|(_, r)| r.pages.to_vec()).collect();
        Self { bucket, rows, pages, floor, placement, row_order: Vec::new() }
    }
}

/// A step's tables, posted into the second table set during the step before it.
pub(crate) struct Prepared {
    /// The engine's number of the step it is for.
    pub step: u64,
    pub key: TableKey,
    pub plans: [StepPlan; 2],
    pub merged: Merged,
    /// The valid rows' tokens as posted (an unknown token as 0).
    pub tokens: Vec<usize>,
}

/// Prepared state for the next step.
#[derive(Default)]
pub(crate) struct State {
    pub prepared: Option<Prepared>,
    failures: u64,
}

impl State {
    /// Reuse the prepared set when the prediction matches this step.
    pub fn take(&mut self, step: u64, key: impl FnOnce() -> TableKey) -> Option<Prepared> {
        let p = self.prepared.take()?;
        if p.step != step {
            return None;
        }
        if p.key == key() {
            Some(p)
        } else {
            None
        }
    }

}


/// Piece 1's result: the predicted step's spans, bucket, key and plans.
pub(crate) struct Planned {
    spans: Vec<AheadSpan>,
    bucket: usize,
    key: TableKey,
    plans: [StepPlan; 2],
}

enum Stage {
    Start,
    Planned(Planned),
    Done,
}

/// One step's prepare of the next (`Engine::ahead_early`): the prediction (`Driver::predict` behind the worker's
/// closure) and its pieces.
pub struct Early<'a> {
    predict: &'a mut dyn FnMut() -> Option<Vec<AheadSpan>>,
    at: Option<[usize; 4]>,
    stage: Stage,
    /// Whether this preparation suspended the machine gate.
    suspended: bool,
}

impl Early<'_> {
    /// After layer `l` of `n` is queued, establish eligibility `fill` layers before the end.
    /// Run the next piece when it is due and at least `full` launches are unanswered.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn at_layer(
        &mut self,
        l: usize,
        n: usize,
        device: &mut Device,
        resident: &resident::Resident,
        official: &mut attention::AttentionState,
        dense: &mut DenseSet,
        state: &mut State,
        next: u64,
    ) -> Result<(), BoxError> {
        let Some([fill, p1, p2, full]) = self.at else { return Ok(()) };
        let left = n - 1 - l;
        if !self.suspended && left <= fill && !matches!(self.stage, Stage::Done) && queue::queueing() {
            // The select boundary still allows preparation when the gate is off.
            if !queue::napping() {
                self.at = None;
                return Ok(());
            }
            self.suspended = true;
            queue::suspend_nap(true);
        }
        let due = match self.stage {
            Stage::Start => left <= p1,
            Stage::Planned(_) => left <= p2,
            Stage::Done => false,
        };
        // Use the queue length after the latest submit's reap. The last layer has short
        // attention launches, so leave its remaining pieces for the select boundary.
        if !due || left == 0 || queue::queued_len() < full {
            return Ok(());
        }
        self.piece(device, resident, official, dense, state, next).await?;
        if matches!(self.stage, Stage::Done) {
            self.resume();
        }
        Ok(())
    }

    /// Restore the machine gate after preparation.
    fn resume(&mut self) {
        if std::mem::take(&mut self.suspended) {
            queue::suspend_nap(false);
        }
    }

    /// Runs the next piece using the plan settings for step `next`. Only
    /// a failed queued launch is an error; a prepare that fails leaves nothing prepared (the step plans as before).
    #[allow(clippy::too_many_arguments)]
    async fn piece(
        &mut self,
        device: &mut Device,
        resident: &resident::Resident,
        official: &mut attention::AttentionState,
        dense: &mut DenseSet,
        state: &mut State,
        next: u64,
    ) -> Result<(), BoxError> {
        match std::mem::replace(&mut self.stage, Stage::Done) {
            Stage::Start => {
                if let Some(spans) = (self.predict)() {
                    match plan_piece(resident, spans) {
                        Ok(Some(planned)) => self.stage = Stage::Planned(planned),
                        Ok(None) => {}
                        Err(why) => failed(state, why),
                    }
                }
            }
            Stage::Planned(planned) => {
                match post_piece(device, official, dense, planned, next).await {
                    Ok(prepared) => {
                        state.prepared = Some(prepared);
                    }
                    Err(why) => failed(state, why),
                }
            }
            Stage::Done => {
                return Ok(());
            }
        }
        queue::reap().await?;
        Ok(())
    }
}

fn failed(state: &mut State, why: BoxError) {
    state.failures += 1;
    if state.failures <= 3 {
        eprintln!("ENGINE_PLANAHEAD prepare failed (the step plans as before): {why}");
    }
}

/// The rows of `spans` as `run` lays them out: attention rows, tokens (an unknown one as 0), row classes, prompt and
/// decode rows; `None` when a span cannot be laid out.
#[allow(clippy::type_complexity)]
fn layout(spans: &[AheadSpan]) -> Option<(Vec<AttnRow<'_>>, Vec<usize>, Vec<bool>, usize, usize)> {
    let mut attn: Vec<AttnRow> = Vec::new();
    let mut tokens: Vec<usize> = Vec::new();
    let mut prompt: Vec<bool> = Vec::new();
    let (mut prompt_rows, mut decode_rows) = (0, 0);
    for span in spans {
        if span.tokens.is_empty() || span.owner >= tables::OWNERS {
            return None;
        }
        let first_row = attn.len();
        for (j, token) in span.tokens.iter().enumerate() {
            attn.push(AttnRow { owner: Some(span.owner), position: span.start + j, first_position: span.start, first_row, pages: &span.pages });
            tokens.push(token.unwrap_or(0));
            prompt.push(!span.decode);
        }
        match span.decode {
            true => decode_rows += span.tokens.len(),
            false => prompt_rows += span.tokens.len(),
        }
    }
    Some((attn, tokens, prompt, prompt_rows, decode_rows))
}

fn row_order(spans: &[AheadSpan], bucket: usize) -> Result<schedule::RowOrder, BoxError> {
    Ok(schedule::RowOrder::of(&spans.iter().map(|s| schedule::RowSpan {
        rows: s.tokens.len(), emits: s.emits,
    }).collect::<Vec<_>>(), bucket)?)
}

/// Plan the predicted step with the same bucket and resident geometry as `run`.
fn plan_piece(resident: &resident::Resident, spans: Vec<AheadSpan>) -> Result<Option<Planned>, BoxError> {
    let (key, plans, bucket) = {
        let Some((mut attn, _, prompt, prompt_rows, decode_rows)) = layout(&spans) else { return Ok(None) };
        let valid = attn.len();
        let Some(bucket) = crate::host::engine::step_bucket_for(valid, prompt_rows) else { return Ok(None) };
        let rss = resident.peek(bucket, prompt_rows, decode_rows);
        let floor = rss.is_some();
        let placement = rss.as_ref().and_then(|(_, images)| *images);
        attn.resize(bucket, AttnRow::PADDING);
        let order = row_order(&spans, bucket)?;
        let key = TableKey::of(bucket, &attn[..valid], &prompt, floor, placement).with_row_order(&order);
        let guard_floor = floor.then(|| dsl::SlidingDecodeFloor::new(crate::ops_dsl_attention::window::GROUP_WINDOW));
        let guard_placement = placement.map(|i| dsl::PromptPlacement::new(i, crate::ops_core::resident::PROMPT_ROW));
        let guard_full = dsl::FullGroups::new(bucket);
        let plans = plan_row_attention(&attn, &prompt, &order);
        drop(guard_full);
        drop(guard_placement);
        drop(guard_floor);
        (key, plans?, bucket)
    };
    Ok(Some(Planned { spans, bucket, key, plans }))
}

/// Piece 2: posts `planned`'s tables into its bucket's second table set and the second word set.
async fn post_piece(device: &mut Device, official: &mut attention::AttentionState, dense: &mut DenseSet, planned: Planned, next: u64) -> Result<Prepared, BoxError> {
    let Planned { spans, bucket, key, plans } = planned;
    let (mut attn, tokens, _, _, _) = layout(&spans).ok_or("planahead: the planned spans lost their layout")?;
    attn.resize(bucket, AttnRow::PADDING);
    let order = row_order(&spans, bucket)?;
    let attn = order.physical(&attn);
    let mut tokens = tokens;
    tokens.resize(bucket, 0);
    let tokens = order.physical(&tokens);
    let positions: Vec<usize> = attn.iter().map(|r| r.position).collect();
    let rows = Rows { attn: &attn, tokens: &tokens, positions: &positions };
    let (plans, merged) = match bucket {
        128 => post::<B128>(device, official, dense, &rows, plans).await?,
        1024 => post::<B1024>(device, official, dense, &rows, plans).await?,
        1536 => post::<B1536>(device, official, dense, &rows, plans).await?,
        _ => unreachable!("validated bucket"),
    };
    Ok(Prepared { step: next, key, plans, merged, tokens })
}

impl Engine {

    /// Prepare the next step during the upcoming pipeline launch.
    pub fn ahead_early<'a>(&self, predict: &'a mut dyn FnMut() -> Option<Vec<AheadSpan>>) -> Early<'a> {
        Early { predict, at: Some([40, 7, 4, 4]), stage: Stage::Start, suspended: false }
    }

    /// Finish preparing the next step while the current step runs, then restore the machine gate.
    pub(super) async fn prepare_ahead(&mut self, device: &mut Device, early: &mut Early<'_>) -> Result<(), BoxError> {
        let next = self.steps;
        let Engine { official, resident, dense, ahead, .. } = self;
        let mut result = Ok(());
        for _ in 0..2 {
            if matches!(early.stage, Stage::Done) {
                break;
            }
            result = early.piece(device, resident, official, dense, ahead, next).await;
            if result.is_err() {
                break;
            }
        }
        early.resume();
        result
    }
}

/// The rows `post` posts.
struct Rows<'a> {
    attn: &'a [AttnRow<'a>],
    tokens: &'a [usize],
    positions: &'a [usize],
}

/// Posts `plans` of `rows` into bucket `B`'s second table set and the second word set, each swapped in for the call and
/// out again. The running step keeps its buffers and merged layout. Return the plans
/// and the next step's word layout.
async fn post<B: Bucket>(
    device: &mut Device,
    official: &mut attention::AttentionState,
    dense_set: &mut DenseSet,
    rows: &Rows<'_>,
    plans: [StepPlan; 2],
) -> Result<([StepPlan; 2], Merged), BoxError> {
    let dense = B::dense(dense_set);
    let merged_before = official.merged;
    dense.swap_tables();
    official.swap_words();
    let result = post_attention(
        device, official, dense, plans, rows.attn,
        (rows.tokens, rows.positions),
    ).await;
    let merged = official.merged;
    official.merged = merged_before;
    official.swap_words();
    dense.swap_tables();
    result.map(|plans| (plans, merged))
}
