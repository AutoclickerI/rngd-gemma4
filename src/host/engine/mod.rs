//! Token-packed inference over prompt chunks and decode tokens from active requests.
//! `driver` admits requests and allocates their KV pages. `schedule` packs contiguous
//! spans per request, with at most one emitting row per span.
//! `attention` and `dsl` plan attention tables using `attention_geometry`.
//! `core`, `carriers` and `segments` launch the embedding and decoder segments.
//! A step runs `pre` for layer 0's QKV, then attention and `mid` segments that publish
//! KV and produce QKV for the next layer. `resident` and `resident_full` can fuse the
//! next attention into a segment. `last` produces the output rows for `sampler`.
//! The driver retires selections through `host::generate` and releases finished
//! requests' KV pages.
use std::collections::HashSet;

use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;
use crate::host::audio::AudioFrames;
use crate::host::image::ImagePatches;
use crate::host::load::{Layer, Weights};
use crate::ops_glue;

pub mod ahead;
pub mod attention;
pub(crate) mod buckets;
pub mod carriers;
pub(crate) mod core;
pub mod driver;
pub mod dsl;
pub(crate) mod resident;
pub(crate) mod resident_full;
pub(crate) mod int8;
mod l0table;
pub mod attention_geometry;
pub mod resolution;
pub(crate) mod queue;
pub(crate) mod launch_thread;
pub mod sampler;
pub mod pipeline;
pub(crate) mod feedback;
pub mod schedule;
pub(crate) mod segments;
pub mod step_image;
pub mod tables;
pub(crate) mod uploads;

use buckets::{B128, B1024, B1536, Bucket, Dense, DenseSet, retype, restore};
use attention::StepPlan;
use attention_geometry::Family;
use carriers::Carriers;
pub use int8::SlidingKv;
use sampler::OutputRows;
use segments::{MediaRows, Rope};
use tables::AttnRow;

type BoxError = Box<dyn std::error::Error>;

/// One row's input: a token id or an image/audio soft token.
#[derive(Clone, Copy)]
pub enum RowInput<'a> {
    Token(usize),
    /// The preceding queued step writes this owner's pick; the host has not read it.
    DeviceToken(usize),
    Image(&'a ImagePatches, usize),
    Audio(&'a AudioFrames, usize),
}

impl RowInput<'_> {
    /// The embedding-table row the step gathers for this row (media rows replace theirs).
    fn token(&self) -> usize {
        match *self {
            RowInput::Token(token) => token,
            _ => 0,
        }
    }

}

/// One request's rows in a step: positions `start.. start + inputs.len()` of owner `owner`.
pub struct Span<'a> {
    pub owner: usize,
    pub pages: &'a [usize],
    pub start: usize,
    pub inputs: Vec<RowInput<'a>>,
    pub banned: Option<&'a HashSet<usize>>,
    /// A decode row (the request's prompt is complete).
    pub decode: bool,
}

/// One layer's KV pool.
pub(crate) enum LayerPool {
    /// E4M3 elements (`int8::SlidingKv`).
    Sliding { k: HbmTensor<SlidingKv, Chip, m![Rs, Ns, Ds]>, v: HbmTensor<SlidingKv, Chip, m![Rs, Ns, Ds]> },
    Full { k: HbmTensor<f8e4m3, Chip, m![Rf, Df]>, v: HbmTensor<f8e4m3, Chip, m![Rf, Df]> },
}

pub struct Engine {
    weights: Weights,
    l0_table: l0table::Table,
    pools: Vec<LayerPool>,
    official: attention::AttentionState,
    /// Runs the next Sliding layer's attention groups inside `mid_ss`.
    resident: resident::Resident,
    /// Runs Full decode groups inside `mid_sf`.
    resident_full: resident_full::ResidentFull,
    carriers: Carriers,
    rope: Rope,
    media: MediaRows,
    dense: DenseSet,
    outputs: OutputRows,
    pipeline: pipeline::State,
    steps: u64,
    /// The next step's tables, prepared during this step.
    ahead: ahead::State,
}

/// The engine's first launch copies 16 KiB. The carried image first cleans the PE
/// D-cache over the firmware's shared fields. The reference build compiles the copy from source.
async fn spm_clean(device: &mut Device) -> Result<(), BoxError> {
    use crate::ops_core::spm_clean::Rows;
    let x: HbmTensor<f32, Chip, Rows> = HostTensor::<f32, Rows>::zero().to_hbm(&mut device.pdma).await?;
    let mut out: HbmTensor<f32, Chip, Rows> = HostTensor::<f32, Rows>::zero().to_hbm(&mut device.pdma).await?;
    #[cfg(not(feature = "compile-carried"))]
    crate::host::prebuilt::carried_launch!(plain, device, (ops_core::spm_clean::spm_clean), (&x, &mut out))?;
    #[cfg(feature = "compile-carried")]
    crate::host::prebuilt::compiled_kernel!(plain, device, (ops_core::spm_clean::spm_clean), (&x, &mut out))?;
    Ok(())
}

impl Engine {

    pub(crate) async fn with_q4_prefetch(
        device: &mut Device,
        mut weights: Weights,
        checkpoint: std::sync::Arc<memmap2::Mmap>,
        q4_prefetch: int8::q4serve::Prefetch,
    ) -> Result<Self, BoxError> {
        if !matches!(weights.layers.first(), Some(Layer::Sliding(_))) || !matches!(weights.layers.last(), Some(Layer::Full(_))) {
            return Err("the segments need a Sliding first layer and a Full last layer".into());
        }
        if weights.layers.windows(2).any(|w| matches!((&w[0], &w[1]), (Layer::Full(_), Layer::Full(_)))) {
            return Err("two consecutive Full layers have no segment".into());
        }
        // A payload out of step with the carried set stops the engine here, not at a first launch.
        crate::host::prebuilt::check_installed()?;
        // the first launch of the process, before any weight preparation, cleans the D-cache lines the PE
        // firmware left dirty over the shared fields (`ops_core::spm_clean`; the clean is in its carried image).
        spm_clean(device).await.map_err(|why| format!("spm_clean: {why}"))?;
        queue::reset();
        // The step families route launches of functions the modules below load and register (`step_image`).
        step_image::reset();
        let carriers = Carriers::prepare(device, &mut weights, checkpoint.clone(), q4_prefetch).await.map_err(|why| format!("carriers: {why}"))?;
        let official = attention::AttentionState::new(device).await.map_err(|why| format!("official attention: {why}"))?;
        let resident = resident::Resident::new(device).await.map_err(|why| format!("resident: {why}"))?;
        let resident_full = resident_full::ResidentFull::new(device).await.map_err(|why| format!("resident_full: {why}"))?;
        step_image::load(device).await.map_err(|why| format!("step_image: {why}"))?;
        let layers = &weights.layers;
        let mut pools = Vec::with_capacity(layers.len());
        for layer in layers {
            pools.push(match layer {
                Layer::Sliding(_) => {
                    let (k, v) = allocate_sliding_pool(device).await?;
                    LayerPool::Sliding { k, v }
                }
                Layer::Full(_) => {
                    let (k, v) = allocate_full_pool(device).await?;
                    LayerPool::Full { k, v }
                }
            });
        }
        let rope = Rope::new(device).await?;
        let media = MediaRows::new(device, tables::MAX_STEP_ROWS).await?;
        let mut dense = DenseSet::new(device).await?;
        let outputs = OutputRows::new(device).await?;
        dense.double(device).await?;
        // Place both select banks before serving so the first large head does not allocate at the tail.
        let pipeline = pipeline::State::new(device).await?;
        let l0_table = l0table::Table::build(device, checkpoint).await?;
        let engine = Self {
            l0_table,
            pools,
            official,
            resident,
            resident_full,
            carriers,
            rope,
            media,
            dense,
            outputs,
            weights,
            steps: 0,
            ahead: Default::default(),
            pipeline,
        };
        // Exercise every prepared argument constructor on host before the first request.
        engine.carriers.check_arguments(&engine.weights, &engine.rope)?;
        resolution::check()?;
        Ok(engine)
    }

    /// Full positions the largest attention kernel covers.
    pub fn full_reach(&self) -> usize {
        dsl::FULL_REACH
    }

    pub(crate) async fn prepare_media(&mut self, device: &mut Device, requests: &[crate::host::media::MediaRequest<'_>]) -> Result<(), BoxError> {
        queue::set_enabled(true);
        self.media.prepare(device, &self.weights, requests).await
    }

    async fn step_launch_with_work(&mut self, device: &mut Device, spans: &[Span<'_>], early: Option<&mut ahead::Early<'_>>,
        work: pipeline::Work<'_>) -> Result<Option<resident::Step>, BoxError> {
        let rows: usize = spans.iter().map(|s| s.inputs.len()).sum();
        let prompt_rows: usize = spans.iter().filter(|s| !s.decode).map(|s| s.inputs.len()).sum();
        let bucket = step_bucket_for(rows, prompt_rows).ok_or_else(|| format!("{rows} rows exceed the largest token bucket"))?;

        let step = self.steps;
        self.steps += 1;
        queue::set_enabled(true);
        // Keep early preparation suspended across the pipelined step boundary.
        queue::suspend_nap(true);
        let result = match bucket {
            128 => self.run::<B128>(device, spans, early, work).await,
            1024 => self.run::<B1024>(device, spans, early, work).await,
            1536 => self.run::<B1536>(device, spans, early, work).await,
            _ => return Err("engine: unsupported executable bucket".into()),
        };

        if result.is_err() {
            queue::clear();
            self.end_step(step);
        }
        result
    }

    fn end_step(&mut self, step: u64) {
        queue::suspend_nap(false);
        queue::nap_gate_step(step);
        step_image::step_ended();
    }

    #[allow(clippy::too_many_arguments)]
    async fn run<B: Bucket>(
        &mut self,
        device: &mut Device,
        spans: &[Span<'_>],
        mut early: Option<&mut ahead::Early<'_>>,
        mut work: pipeline::Work<'_>,
    ) -> Result<Option<resident::Step>, BoxError> {
        let next_step = self.steps;
        let Engine { weights, l0_table, pools, official, resident, resident_full, carriers, rope, media, dense: dense_set, outputs, ahead, pipeline, .. } = self;
        let bucket = B::T::SIZE;
        // A step that failed part-way may have left its family active.
        step_image::activate(None);

        // Rows: spans in order, then padding.
        let mut attn: Vec<AttnRow> = Vec::with_capacity(bucket);
        let mut inputs: Vec<RowInput> = Vec::with_capacity(bucket);
        let mut prompt = Vec::with_capacity(bucket);
        let mut emitting: Vec<(usize, usize)> = Vec::new(); // (span, row)
        for (s, span) in spans.iter().enumerate() {
            if span.inputs.is_empty() {
                return Err("a span has at least one row".into());
            }
            if span.owner >= tables::OWNERS {
                return Err(format!("owner slot {} out of range", span.owner).into());
            }
            let first_row = attn.len();
            for (j, input) in span.inputs.iter().enumerate() {
                attn.push(AttnRow { owner: Some(span.owner), position: span.start + j, first_position: span.start, first_row, pages: span.pages });
                inputs.push(*input);
                prompt.push(!span.decode);
            }
            if span.banned.is_some() {
                emitting.push((s, attn.len() - 1));
            }
        }
        let valid = attn.len();
        let prompt_rows: usize = spans.iter().filter(|s| !s.decode).map(|s| s.inputs.len()).sum();
        attn.resize(bucket, AttnRow::PADDING);
        let logical_attn = attn;
        let row_order = schedule::RowOrder::of(&spans.iter().map(|s| schedule::RowSpan {
            rows: s.inputs.len(), emits: s.banned.is_some(),
        }).collect::<Vec<_>>(), bucket)?;
        let attn = row_order.physical(&logical_attn);
        let logical_inputs = inputs;
        let mut padded_inputs = logical_inputs.clone();
        padded_inputs.resize(bucket, RowInput::Token(0));
        let inputs = row_order.physical(&padded_inputs);
        // Preserve logical span indices, including when a held step completes later.
        for (_, row) in &mut emitting { *row = row_order.logical_to_physical[*row]; }

        let layers = &weights.layers;
        let n = layers.len();
        if emitting.len() > Out128::SIZE {
            return Err(format!("{} emitting rows exceed {}", emitting.len(), Out128::SIZE).into());
        }
        // Resident segments can fuse the next layer's attention. Tok128 fuses Sliding
        // decode with Q4. Larger buckets also cover selected prompt and Full decode shapes.
        // The attention plan launches any remaining rounds separately.
        let rss_groups = resident.peek(bucket, prompt_rows, valid - prompt_rows);

        let rows_f: Vec<i32>;
        // Keep the resident tables alive until the queued launches finish.
        let kept;
        // Tracks whether `last`'s row tables went out right after `pre`.
        let mut rows_posted = false;
        // The previous step may still be reading the other bank's gather indices.
        let compact_bank = pipeline.write;
        {
            let dense = B::dense(dense_set);
            let tokens: Vec<usize> = inputs.iter().map(RowInput::token).collect();
            let positions: Vec<usize> = attn.iter().map(|r| r.position).collect();
            let floor_on = rss_groups.is_some();
            let images = rss_groups.as_ref().and_then(|(_, images)| *images);
            // Next-step preparation: the tables the previous step posted for this one, when they are this step's (`ahead`).
            let prepared = ahead.take(next_step - 1, || ahead::TableKey::of(bucket, &logical_attn[..valid], &prompt, floor_on, images).with_row_order(&row_order));
            let plans = match prepared {
                Some(p) => {
                    dense.swap_tables();
                    official.swap_words();
                    official.merged = p.merged;
                    if p.tokens != tokens {
                        dense.upload_tok(device, &tokens).await?;
                    }
                    p.plans
                }
                None => {
                    dense.swap_tables();
                    official.swap_words();
                    let floor = floor_on.then(|| dsl::SlidingDecodeFloor::new(crate::ops_dsl_attention::window::GROUP_WINDOW));
                    // Piece 2 of the resident path: the plan places the program's prompt image at its fixed row.
                    let placement = images.map(|i| dsl::PromptPlacement::new(i, crate::ops_core::resident::PROMPT_ROW));
                    let full_groups = dsl::FullGroups::new(bucket);
                    let planned = prepare_row_attention::<B>(
                        device, official, dense, &logical_attn, &attn, &prompt, &row_order,
                        (&tokens, &positions),
                    ).await?;
                    drop(full_groups);
                    drop(placement);
                    drop(floor);
                    planned
                }
            };
            let rss_step = match rss_groups {
                Some((g, _)) => resident.step(&plans[0], bucket, g),
                None => None,
            };
            let rsf_step = resident_full.step(device, &plans[1]).await?;
            // The Sliding -> Full segments that run the Full layer's decode groups (`resident_full`).
            let rsf_at = |l: usize| {
                rsf_step.is_some()
                    && l + 1 < n
                    && matches!((&layers[l], &layers[l + 1]), (Layer::Sliding(_), Layer::Full(_)))
            };
            // The layers whose segment runs the next layer's decode groups.
            let rss_at = |l: usize| {
                rss_step.is_some()
                    && l + 1 < n
                    && matches!((&layers[l], &layers[l + 1]), (Layer::Sliding(_), Layer::Sliding(_)))
            };
            // The layers whose hybrid Full -> Sliding segment runs the next (Sliding) layer's whole attention (`rfs`).
            let rfs_at = |l: usize| {
                rss_step.is_some()
                    && l + 1 < n
                    && matches!((&layers[l], &layers[l + 1]), (Layer::Full(_), Layer::Sliding(_)))
            };
            rows_f = plans[1].rows.clone();
            // The step family (`step_image`): when one holds every segment program and attention launch of the step,
            // its layers' launches of members run that one image.
            let family = step_image::choose(
                step_image::StepKey {
                    bucket,
                    all_resident: (0..n - 1).all(|l| rss_at(l) || rsf_at(l) || rfs_at(l)),
                    rss: rss_step.as_ref().map(resident::Step::paths),
                    rsf: rsf_step.as_ref().map(|s| s.path()),
                },
                || dsl::step_paths(&plans, rsf_step.as_ref().map_or(0, |step| step.served())),
            );

            // The embedded input: token rows, then the media rows' soft rows over them.
            if inputs.iter().any(|i| matches!(i, RowInput::DeviceToken(_))) {
                B::resolve_tokens(device, dense, &pipeline.tables[pipeline.write ^ 1], &inputs).await?;
            }
            media.begin(spans);
            let text_only = inputs.iter().all(|i| matches!(i, RowInput::Token(_) | RowInput::DeviceToken(_)));
            if !text_only && media.ready(spans) {
                media.embed_resident::<B>(device, weights, dense, spans, &row_order.logical_to_physical).await?;
            } else {
                B::embed(device, dense, weights).await?;
                if !text_only {
                    queue::drain().await?;
                    media.embed::<B>(device, weights, dense, &logical_inputs, &row_order.logical_to_physical).await?;
                }
            }

            let Layer::Sliding(layer0) = &layers[0] else { unreachable!("checked at engine initialization") };
            {
                let LayerPool::Sliding { k, v } = &mut pools[0] else { unreachable!() };
                B::table_pre(device, dense, l0_table.args(), rope, &mut official.sliding.qin, k, v).await?;
                if !text_only {
                    l0_table.media.run(device, B::table_media_source(dense), &inputs, &positions, &plans[0].rows,
                        &carriers.int8, layer0, rope, &mut official.sliding.qin, k, v).await?;
                }
            }
            // Post the output-row tables while the first segment runs. Only the final segment reads them.
            if !emitting.is_empty() && queue::queueing() {
                let mut rows: Vec<usize> = emitting.iter().map(|&(_, row)| row).collect();
                let a_rows: Vec<i32> = rows.iter().flat_map(|&r| [rows_f[2 * r], rows_f[2 * r + 1]]).collect();
                if bucket == 1536 {
                    post_compact_rows(device, &mut dense_set.compact_rows[compact_bank], &emitting).await?;
                    rows = (0..rows.len()).collect();
                }
                post_out_rows(device, outputs, &rows, &a_rows).await?;
                rows_posted = true;
            }
            // Only emitting rows reach `last`. Keep the original output regions and
            // table lifetimes; final attention reads Q/K/V but never another launch's
            // output.
            let final_live = plans[1].live_launches(Family::Full, &emitting)?;
            step_image::prepare(&*device, family).await?;
            step_image::activate(family);
            for l in 0..n {
                // Next-step preparation uses the table sets between layers (`ahead::Early::at_layer`).
                let dense = B::dense(&mut *dense_set);
                let (head, tail) = pools.split_at_mut(l + 1);
                let pool = &mut head[l];
                // Layer l's query halves and K/V are in place (the segment that produced them did
                // the moves); its output is gathered by the next segment.
                        // The previous segment ran this layer's last decode groups (`resident`), and in piece 2 its
                        // prompt launch (the plan's first).
                        let (first, last) = match (rss_step.as_ref(), rsf_step.as_ref()) {
                            (Some(step), _) if l > 0 && (rss_at(l - 1) || rfs_at(l - 1)) => step.skips(),
                            // `resident_full`: the previous segment ran this Full layer's last decode launches.
                            (_, Some(step)) if l > 0 && rsf_at(l - 1) => (0, step.served()),
                            _ => (0, 0),
                        };
                        let live = (l + 1 == n).then_some(final_live.as_slice());
                launches(&plans, official, pool, last, first, live).await?;
                if let Some(next) = layers.get(l + 1) {
                    let next_pool = &mut tail[0];
                    if let (true, Some(step)) = (rss_at(l), rss_step.as_ref()) {
                        let (i8w, c) = (&carriers.int8, &carriers.core);
                        match (&layers[l], next, pool, next_pool) {
                            (Layer::Sliding(this), Layer::Sliding(nx), LayerPool::Sliding { k: k_l, v: v_l }, LayerPool::Sliding { k: k_n, v: v_n }) => {
                                resident.mid_ss(device, step, &plans[0], dense, &this.layer_scalar, i8w.mid_ss(l, nx, c, rope)?, official, k_l, v_l, k_n, v_n).await?
                            }
                            _ => return Err(format!("resident mid({l}): layer and pool families disagree").into()),
                        }
                    } else if let (true, Some(step)) = (rfs_at(l), rss_step.as_ref()) {
                        let (i8w, c) = (&carriers.int8, &carriers.core);
                        match (&layers[l], next, next_pool) {
                            (Layer::Full(this), Layer::Sliding(nx), LayerPool::Sliding { k: k_n, v: v_n }) => {
                                resident.mid_fs(device, step, &plans[0], dense, &this.layer_scalar, i8w.mid_fs(l, nx, c, rope)?, official, k_n, v_n).await?
                            }
                            _ => return Err(format!("rfs mid({l}): layer and pool families disagree").into()),
                        }
                    } else if let (true, Some(step)) = (rsf_at(l), rsf_step.as_ref()) {
                        let (i8w, c) = (&carriers.int8, &carriers.core);
                        match (&layers[l], next, pool, next_pool) {
                            (Layer::Sliding(this), Layer::Full(nx), LayerPool::Sliding { k: k_l, v: v_l }, LayerPool::Full { k: k_n, v: v_n }) => {
                                resident_full.mid_sf(device, step, dense, &this.layer_scalar, i8w.mid_sf(l, nx, c, rope)?, official, k_l, v_l, k_n, v_n).await?
                            }
                            _ => return Err(format!("resident_full mid({l}): layer and pool families disagree").into()),
                        }
                    } else {
                        core::mid::<B>(device, carriers, l, dense, &layers[l], next, rope, official, pool, next_pool).await?;
                    }
                }
                // Submit several dependent layers before reading the previous token bank.
                // Their device work covers readback and host retirement.
                if l == 4 && let Some(retire) = work.retire.take() {
                    pipeline.retire(device, retire).await?;
                }
                // Stage arrivals while the queue still has layer work. At this point the
                // current step's media gather is already ordered before any new encoder.
                if l + 14 >= n && l + 8 < n && queue::queued_len() >= 4
                    && let Some(prefetch) = work.prefetch.take()
                {
                    let next = prefetch();
                    if !next.is_empty() {
                        media.prepare(device, weights, &next.iter().map(|m| m.request()).collect::<Vec<_>>()).await?;
                    }
                }
                if let Some(e) = early.as_deref_mut() {
                    e.at_layer(l, n, device, resident, official, dense_set, ahead, next_step).await?;
                }
                if l == 1 {
                    queue::suspend_nap(false);
                } else if l + 4 >= n {
                    // Leave the final layers and exact selection to the launch queue.
                    queue::suspend_nap(true);
                }
            }
            queue::flush_pending().await?;
            step_image::activate(None);
            kept = rss_step;
        }

        // `last` and selection for the emitting rows.
        if !emitting.is_empty() {
            let bans: Vec<_> = emitting.iter().map(|&(s, _)| spans[s].banned.expect("emitting row")).collect();
            let mut rows: Vec<usize> = emitting.iter().map(|&(_, row)| row).collect();
            B::widen(device, dense_set).await?;
            let Layer::Full(layer) = weights.layers.last().expect("layers") else { unreachable!("checked at engine initialization") };
            // The output rows' halves in the Full row space, where `last` gathers them (contract 2.3).
            let a_rows: Vec<i32> = rows.iter().flat_map(|&r| [rows_f[2 * r], rows_f[2 * r + 1]]).collect();
            if bucket == 1536 {
                // Compact only emitted residual rows. The Full attention table
                // above retains original row-space positions; only the residual
                // gather indices become 0..n in the unchanged W4 head ABI.
                if !rows_posted {
                    post_compact_rows(device, &mut dense_set.compact_rows[compact_bank], &emitting).await?;
                }
                crate::host::prebuilt::queued_kernel!(device, ops_step::wide_output::compact1536_out128,
                    (&dense_set.b1536.x, &dense_set.compact_rows[compact_bank], &mut dense_set.b1024.x))?;
                rows = (0..rows.len()).collect();
            }
            let last = LastInputs {
                layer, wide: &dense_set.b1024, a_rows: &a_rows, core: &carriers.core.layers[n - 1],
                w4_head: &carriers.core.w4_head, out_f: &official.full.out,
            };
            let owners: Vec<_> = emitting.iter().map(|&(s, _)| spans[s].owner).collect();
            post_last(device, &last, outputs, &rows, rows_posted).await?;
            outputs.queue_exact(device, &bans, &owners, &mut pipeline.tables[pipeline.write]).await?;
        }
        Ok(kept)
    }
}

/// Plan on the original request chunks, then upload operands in physical row-role order.
#[allow(clippy::too_many_arguments)]
async fn prepare_row_attention<B: Bucket>(
    device: &mut Device, official: &mut attention::AttentionState,
    dense: &mut Dense<B::T>, logical: &[AttnRow<'_>], physical: &[AttnRow<'_>], prompt: &[bool],
    order: &schedule::RowOrder, meta: (&[usize], &[usize]),
) -> Result<[StepPlan; 2], BoxError> {
    let plans = plan_row_attention(logical, prompt, order)?;
    post_attention(
        device, official, dense, plans, physical,
        meta,
    ).await
}

/// Only stage addresses and dense-row lookup tables change; launch shapes/order remain the logical plan's.
fn plan_row_attention(logical: &[AttnRow<'_>], prompt: &[bool], order: &schedule::RowOrder) -> Result<[StepPlan; 2], BoxError> {
    let _stage_rows = dsl::StageRows::new(&order.logical_to_physical);
    let mut plans = plan_attention(logical, prompt)?;
    for plan in &mut plans { plan.permute_step_rows(&order.physical_to_logical); }
    Ok(plans)
}

/// Plan both attention families for `prepare_row_attention` and next-step preparation.
fn plan_attention(attn: &[AttnRow<'_>], prompt: &[bool]) -> Result<[StepPlan; 2], BoxError> {
    let plan = |family: Family| -> Result<StepPlan, BoxError> {
        let planned = dsl::plan(family, attn, prompt);
        planned.map_err(|why| format!("{family:?} attention has no plan: {why}").into())
    };
    Ok([plan(Family::Sliding)?, plan(Family::Full)?])
}

/// `prepare_row_attention`'s uploads of `plans` (next-step piece 2 posts them into the second set, `ahead`).
#[allow(clippy::too_many_arguments)]
async fn post_attention<T: AxisName>(
    device: &mut Device,
    official: &mut attention::AttentionState,
    dense: &mut Dense<T>,
    plans: [StepPlan; 2],
    attn: &[AttnRow<'_>],
    meta: (&[usize], &[usize]),
) -> Result<[StepPlan; 2], BoxError> {
    let merged = attention::Merged::of(&plans);
    official.merged = merged;
    let words = &mut official.words_m;
    let tables = Dense::<T>::table_images(attn, plans[0].rows.clone(), plans[1].rows.clone(), meta.0, meta.1);
    let sizes: Vec<_> = tables.iter().map(Vec::len).collect();
    let (starts, used) = attention::packed_layout(merged.used, &sizes);
    words.ensure_packed(device, used).await?;
    let (write, buffer) = words.write_packed(device, &plans, merged, &tables, &starts, used)?;
    uploads::join(vec![write]).await?;
    let tensors = dense.table_tensors();
    for ((tensor, &at), table) in tensors.into_iter().zip(&starts).zip(&tables) {
        uploads::alias(tensor, buffer.slice(at * 4..(at + table.len()) * 4))?;
    }
    Ok(plans)
}

/// One layer's attention launches: the launches of its family's plan but its last `fused` (the decode launches a fused
/// segment program runs, `fused8`).
async fn launches(
    plans: &[StepPlan; 2],
    state: &mut attention::AttentionState,
    pool: &mut LayerPool,
    fused: usize,
    skip_first: usize,
    live: Option<&[bool]>,
) -> Result<(), BoxError> {
    match pool {
        LayerPool::Sliding { k, v } => {
            attention::run_planned(state, Family::Sliding, &plans[0], fused, skip_first, None, k, v).await?;
        }
        LayerPool::Full { k, v } => {
            if skip_first > 0 {
                return Err("no program runs a Full prefill launch".into());
            }
            // `fused`: the last decode launches, already run by `resident_full`.
            // `live` indexes the original complete plan, including that resident suffix.
            attention::run_planned(state, Family::Full, &plans[1], fused, 0, live, k, v).await?;
        }
    }
    Ok(())
}

/// Allocate and zero a Sliding layer's E4M3 pools by copying a single zero ring to the device.
async fn allocate_sliding_pool(
    device: &mut Device,
) -> Result<(HbmTensor<SlidingKv, Chip, m![Rs, Ns, Ds]>, HbmTensor<SlidingKv, Chip, m![Rs, Ns, Ds]>), Error> {
    {
        let zero_ring = HostTensor::<SlidingKv, m![Ts, Ns, Ds]>::zero().to_hbm(&mut device.pdma).await?;
        launch(ops_glue::allocate_sliding_pool8, (&mut *device, &zero_ring)).await
    }
}

/// A Full layer's pools (see `allocate_sliding_pool`).
async fn allocate_full_pool(device: &mut Device) -> Result<(HbmTensor<f8e4m3, Chip, m![Rf, Df]>, HbmTensor<f8e4m3, Chip, m![Rf, Df]>), Error> {
    {
        let zero_page = HostTensor::<f8e4m3, m![Tf, Df]>::zero().to_hbm(&mut device.pdma).await?;
        launch(ops_glue::allocate_full_pool, (&mut *device, &zero_page)).await
    }
}

/// What `last` reads besides the output rows.
struct LastInputs<'a> {
    layer: &'a crate::host::load::FullLayer,
    /// The step tensors widened to Tok1024 (`last` reads `x`).
    wide: &'a Dense<Tok1024>,
    /// The output rows' halves in the Full row space.
    a_rows: &'a [i32],
    core: &'a core::CoreLayer,
    w4_head: &'a head4::Head,
    /// The Full row space's output, where the core's `last` gathers layer 47's halves.
    out_f: &'a HbmTensor<bf16, Chip, m![Rq, Hqf]>,
}

/// Map the emitting residual rows into the head's Tok1024 input.
async fn post_compact_rows(device: &mut Device, compact: &mut HbmTensor<i32, Chip, m![Out128]>, emitting: &[(usize, usize)]) -> Result<(), Error> {
    let mut selected: Vec<i32> = emitting.iter().map(|&(_, row)| row as i32).collect();
    selected.resize(Out128::SIZE, selected[0]);
    HostTensor::<i32, m![Out128]>::from_vec(selected).to_hbm(&mut device.pdma).output(compact).await?;
    Ok(())
}

/// `last`'s row tables for the output rows `rows` (step rows) and `a_rows` (their halves in the Full row space): padding
/// entries repeat the first output row.
async fn post_out_rows(device: &mut Device, out: &mut OutputRows, rows: &[usize], a_rows: &[i32]) -> Result<(), Error> {
    let mut index: Vec<i32> = rows.iter().map(|&r| r as i32).collect();
    index.resize(Out128::SIZE, rows[0] as i32);
    HostTensor::<i32, m![Out128]>::from_vec(index).to_hbm(&mut device.pdma).output(&mut out.rows).await?;
    let mut halves = a_rows.to_vec();
    let first = [a_rows[0], a_rows[1]];
    while halves.len() < Out128::SIZE * Hh::SIZE {
        halves.extend_from_slice(&first);
    }
    HostTensor::<i32, m![Out128, Hh]>::from_vec(halves).to_hbm(&mut device.pdma).output(&mut out.a_rows).await?;
    Ok(())
}

/// Posts the output-row table if needed and queues `last` into
/// `out.logits`.
async fn post_last(device: &mut Device, last: &LastInputs<'_>, out: &mut OutputRows, rows: &[usize], posted: bool) -> Result<(), BoxError> {
    if !posted {
        post_out_rows(device, out, rows, last.a_rows).await?;
    }
    let mut logits: HbmTensor<bf16, Chip, m![Out128, W]> = retype(&mut out.logits);
    let (layer, core, head) = (last.layer, last.core, last.w4_head);
    let launched = crate::host::prebuilt::queued_kernel!(
        device, ops_core::ends::last_bf16_out128,
        (last.out_f, &out.a_rows, &last.wide.x, &out.rows,
         segments::core_full_o(layer, core), segments::core_ffn(&layer.mlp, core),
         &layer.layer_scalar, &head.signed_final_rms, &head.codes, &head.scales,
         &head.h64, &head.h60, &head.eye, &mut logits),
    );
    restore(&mut out.logits, logits);
    launched?;
    Ok(())
}


fn step_bucket_for(rows: usize, prompt_rows: usize) -> Option<usize> {
    match rows {
        ..=128 if prompt_rows == 0 => Some(128),
        ..=1024 => Some(1024),
        ..=1536 => Some(1536),
        _ => None,
    }
}

pub(crate) mod head4;
