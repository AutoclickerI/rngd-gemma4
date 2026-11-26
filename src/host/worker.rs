//! Request scheduling for the token-packed engine.
//!
//! Continuous admission: a waiting request is admitted between steps when a KV owner slot and
//! its whole Full-page budget are free (first in, first out). Every step packs one decode row
//! per decoding request and prompt chunks of the others into one token-packed step
//! (`engine::driver`), selects each emitting row's token on the device and streams its text.
use std::future::Future;
use std::sync::{Arc, mpsc};
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

use crate::api::worker::Job;
use crate::host::engine::driver::{Driver, Limits, PendingRows, pipeline::PipelineSpan};
use crate::host::engine::{Engine, tables};
use crate::host::load;
use crate::host::tokenizer::Tokenizer;
use furiosa_opt_std::prelude::*;

/// Requests computed together; more wait in the queue.
pub const CAPACITY: usize = tables::OWNERS;
/// Prompt plus generated positions allowed per request: the attention kernels' Full reach
/// (`Engine::full_reach`, 4,096; `Worker::new` checks). A longer request fails alone at admission
/// with a 400 from the API (`api::handlers`) and, should one reach the worker, an error event from
/// `Driver::submit`; the worker and the other requests go on.
pub const CONTEXT: usize = 4096;

/// The step limits.
pub const LIMITS: Limits = Limits { owners: CAPACITY, pages: tables::FULL_PAGES, context: CONTEXT };

pub struct Worker {
    runtime: tokio::runtime::Runtime,
    device: Device,
    engine: Engine,
    tokenizer: Tokenizer,
}

impl Worker {
    pub fn new(tokenizer: Tokenizer) -> Result<Self, Box<dyn std::error::Error>> {
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        let mut device = Device::new(crate::ops_step::embed_rows_t128.topology())?;
        let engine = runtime.block_on(async {
            let checkpoint = Arc::new(load::map_checkpoint()?);
            let q4_prefetch = crate::host::engine::int8::q4serve::Prefetch::start(checkpoint.clone());
            let weights = load::load_weights(&mut device, &checkpoint).await?;
            let engine = Engine::with_q4_prefetch(&mut device, weights, checkpoint, q4_prefetch).await?;
            if CONTEXT > engine.full_reach() {
                return Err(format!("context {CONTEXT} passes the attention kernels' Full reach {}", engine.full_reach()).into());
            }
            let mut engine = engine;
            warm_up(&mut device, &mut engine, &tokenizer).await?;
            Ok::<_, Box<dyn std::error::Error>>(engine)
        })?;
        Ok(Self { runtime, device, engine, tokenizer })
    }

    pub fn run(mut self, jobs: mpsc::Receiver<Job>) {
        let jobs = crate::host::arrival::arrivals(jobs);
        let mut driver = Driver::new(LIMITS);
        let release = crate::host::arrival::Release::new();
        driver.set_stream_release(release);
        // The previous step's stream events, sent once the next step is in flight.
        let mut pending: Vec<(crate::host::arrival::Events, crate::api::worker::StreamEvent)> = Vec::new();
        let flush = |pending: &mut Vec<(crate::host::arrival::Events, crate::api::worker::StreamEvent)>| {
            for (events, event) in pending.drain(..) {
                let _ = events.send(event);
            }
        };
        // Flush before scheduling the next step so the stream wakes on the host-only tail.
        let spin = SPIN;
        // Flush deferred text while the next step's queued layers run, before retiring its picks.
        driver.set_defer_text(true);
        let mut text_job: Option<crate::host::engine::driver::TextJob> = None;
        let mut pipeline_rows: Option<PendingRows> = None;
        let mut prepared_spans: Option<(PendingRows, Vec<PipelineSpan>)> = None;
        loop {
            if driver.idle() {
                flush(&mut pending);
                let Ok((job, at)) = jobs.recv() else { break };
                driver.submit_at(&self.tokenizer, job, at);
            }
            for (job, at) in jobs.try_iter() {
                driver.submit_at(&self.tokenizer, job, at);
            }
            driver.admit();
            if driver.active() == 0 {
                flush(&mut pending);
                continue;
            }
            let prior = pipeline_rows.take();
            let empty = PendingRows::default();
            let outstanding = prior.as_ref().unwrap_or(&empty);
            let next = driver.pipeline_plan(outstanding);
            let retired = std::cell::Cell::new(false);
            let mut arrived = Vec::new();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                spin_block_on(&self.runtime, async {
                    let predicted = outstanding.followed_by(&next);
                    let media = driver.media_for_rows(&next);
                    self.engine.prepare_media(&mut self.device, &media).await?;
                    drop(media);
                    let prepared = prepared_spans.take();
                    if !next.0.is_empty() {
                        let snapshot = match prepared {
                            Some((rows, snapshot)) if rows == next => snapshot,
                            _ => driver.pipeline_snapshot(outstanding, &next)?,
                        };
                        let spans = snapshot.iter().map(|s| s.span()).collect::<Result<Vec<_>, _>>()?;
                        let driver = std::cell::RefCell::new(&mut driver);
                        let mut forecast_rows = None;
                        let mut predict = || {
                            let driver = driver.borrow();
                            for (job, at) in jobs.try_iter() {
                                arrived.push(driver.arrive_at(&self.tokenizer, job, at));
                            }
                            let (all, prior) = if retired.get() { (&next, &empty) } else { (&predicted, outstanding) };
                            let (rows, spans) = driver.pipeline_forecast_rows(all, prior, &next, &arrived)?;
                            forecast_rows = Some(rows);
                            Some(spans)
                        };
                        let mut retire = |picks| {
                            let mut driver = driver.borrow_mut();
                            if let Some(job) = text_job.take() { job.run(); }
                            flush(&mut pending);
                            let mut events = driver.pipeline_finish(outstanding, picks, &next);
                            flush(&mut events);
                            driver.text_job().run();
                            retired.set(true);
                        };
                        let mut prefetch = || {
                            let mut driver = driver.borrow_mut();
                            for (job, at) in jobs.try_iter() {
                                driver.submit_at(&self.tokenizer, job, at);
                            }
                            driver.admit();
                            driver.pipeline_media_preview(&next)
                        };
                        let mut early = self.engine.ahead_early(&mut predict);
                        let work = crate::host::engine::pipeline::Work {
                            retire: prior.is_some().then_some(&mut retire),
                            prefetch: Some(&mut prefetch),
                        };
                        self.engine.pipeline_submit(&mut self.device, &spans, Some(&mut early), work).await?;
                        drop(early);
                        if (retired.get() || prior.is_none()) && let Some(rows) = forecast_rows {
                            let driver = driver.into_inner();
                            for arrival in arrived.drain(..) { driver.enqueue(arrival); }
                            driver.admit();
                            // This is a preview, including ownership. Actual admission and packing
                            // still run after draining all arrivals at the next boundary.
                            prepared_spans = driver.pipeline_snapshot(&next, &rows).ok().map(|spans| (rows, spans));
                        }
                        self.engine.pipeline_end();
                    }
                    if prior.is_some() && !retired.get() {
                        self.engine.pipeline_complete(&mut self.device).await.map(Some)
                    } else { Ok(None) }
                }, spin)
            }));
            if let Some(job) = text_job.take() { job.run(); }
            for arrival in arrived {
                driver.enqueue(arrival);
            }
            match result {
                Ok(Ok(picks)) => {
                    flush(&mut pending);
                    if let (Some(prior), Some(picks)) = (prior, picks) {
                        pending = driver.pipeline_finish(&prior, picks, &next);
                    }
                    text_job = Some(driver.text_job());
                    flush(&mut pending);
                    pipeline_rows = (!next.0.is_empty()).then_some(next);
                }
                failed => {
                    let message = match failed { Ok(Err(error)) => error.to_string(), _ => "generation panicked; worker is offline".to_owned() };
                    eprintln!("ENGINE_GENERATION_ERROR {message}");
                    flush(&mut pending); driver.fail_all(&message); return;
                }
            }
        }
        flush(&mut pending);
    }
}

/// Prime the serving chunk geometry before serving requests.
async fn warm_up(device: &mut Device, engine: &mut Engine, tokenizer: &Tokenizer) -> Result<(), Box<dyn std::error::Error>> {
    let text = "Record: the quick brown fox jumps over the lazy dog while the river runs past the mill. ".repeat(300);
    let ids = tokenizer.encode_chat(&[serde_json::json!({ "role": "user", "content": text })], false)?;
    let mut driver = Driver::new(LIMITS);
    for len in [2040usize, 1600, 101, 21].into_iter().chain(std::iter::repeat_n(26, 29)) {
        let prompt_ids: Vec<usize> = ids.iter().copied().cycle().take(len).collect();
        let request = crate::host::generate::GenerationRequest {
            soft_at: vec![None; prompt_ids.len()],
            prompt_ids,
            image: None,
            audio: None,
            max_new_tokens: 2,
            sampling: crate::host::sampling::SamplingConfig::DEFAULT,
            stop_strings: Vec::new(),
            separate_reasoning: false,
            ignore_eos: true,
        };
        let (events, _rx) = mpsc::channel();
        driver.submit(tokenizer, Job { request, stream: false, events });
    }
    let mut prior = PendingRows::default();
    while !driver.idle() {
        driver.admit();
        let next = driver.pipeline_plan(&prior);
        if !next.0.is_empty() {
            let snapshot = driver.pipeline_snapshot(&prior, &next)?;
            let spans = snapshot.iter().map(PipelineSpan::span).collect::<Result<Vec<_>, _>>()?;
            engine.pipeline_submit(device, &spans, None, Default::default()).await?;
            engine.pipeline_end();
        }
        if !prior.0.is_empty() {
            let picks = engine.pipeline_complete(device).await?;
            driver.pipeline_finish(&prior, picks, &next);
        }
        prior = next;
    }
    Ok(())
}

/// How long the worker spins on a pending device wait before yielding.
const SPIN: Duration = Duration::from_micros(50);

/// Polls the future in the runtime context. Spin on short waits, then yield
/// while staying runnable so a device answer can advance the next segment promptly.
fn spin_block_on<F: Future>(runtime: &tokio::runtime::Runtime, future: F, spin: Duration) -> F::Output {
    let _context = runtime.enter();
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    let mut future = std::pin::pin!(future);
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut cx) {
            return value;
        }
        let pending = Instant::now();
        loop {
            if pending.elapsed() < spin {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
            if let Poll::Ready(value) = future.as_mut().poll(&mut cx) {
                return value;
            }
        }
    }
}
