//! The request side of the engine: admission, step planning and delivery, independent of the
//! device. The worker executes the resulting steps through `Engine`.
use std::collections::{HashSet, VecDeque};
use std::rc::Rc;
use std::time::Instant;
use crate::host::arrival::Events;

use crate::api::worker::{Job, StreamEvent};
use crate::host::engine::ahead::AheadSpan;
use crate::host::engine::schedule::{self, Allocator, Demand};
use crate::host::engine::{RowInput, Span};
use crate::host::generate::{Delta, Generation};
use crate::host::tokenizer::Tokenizer;

type BoxError = Box<dyn std::error::Error>;

pub(crate) mod pipeline;
pub use pipeline::PendingRows;

struct Session {
    retired: bool,
    /// Submission order (probes map spans back to their requests).
    id: usize,
    /// Arrival time stamped by the host.
    generation: Generation,
    owner: usize,
    pages: Vec<usize>,
    banned: Rc<HashSet<usize>>,
    events: Events,
    stream: bool,
}

/// A job tokenized but not yet queued (`Driver::arrive`, `Driver::enqueue`).
pub struct Arrival(Option<(Generation, Events, bool, Instant)>);

struct Waiting {
    id: usize,
    generation: Generation,
    events: Events,
    stream: bool,
}

/// Step limits.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Concurrent requests (KV owner slots in use).
    pub owners: usize,
    /// Full pages shared by the admitted requests.
    pub pages: usize,
    /// Prompt plus generated positions per request: `submit` fails a longer request at once.
    pub context: usize,
}

pub struct Driver {
    leased: HashSet<usize>,
    sessions: Vec<Session>,
    waiting: VecDeque<Waiting>,
    allocator: Allocator,
    limits: Limits,
    submitted: usize,
    stream_release: Option<crate::host::arrival::Release>,
    defer_text: bool,
}

/// Detokenize waiting tokens and send deltas in admission order.
pub struct TextJob {
    items: Vec<(std::rc::Rc<std::cell::RefCell<crate::host::generate::TextState>>, Option<Events>)>,
}

impl TextJob {
    pub fn run(self) {
        for (text, events) in self.items {
            let mut out: Vec<StreamEvent> = Vec::new();
            text.borrow_mut().catch_up_or_hold(|delta| {
                if events.is_some() {
                    out.push(match delta {
                        Delta::Content(text) => StreamEvent::Delta(text.to_owned()),
                        Delta::Reasoning(text) => StreamEvent::ReasoningDelta(text.to_owned()),
                    });
                }
                true
            });
            if let Some(events) = events {
                for event in out {
                    let _ = events.send(event);
                }
            }
        }
    }
}

impl Driver {
    pub fn new(limits: Limits) -> Self {
        Self {
            leased: HashSet::new(),
            sessions: Vec::new(),
            waiting: VecDeque::new(),
            allocator: Allocator::new(limits.owners, limits.pages),
            limits,
            submitted: 0,
            stream_release: None,
            defer_text: false,
        }
    }

    /// Requests without stop strings can leave token text for `text_job`.
    pub fn set_defer_text(&mut self, on: bool) {
        self.defer_text = on;
    }

    /// The text work the last step left (`detok`): every request with tokens still to detokenize.
    pub fn text_job(&self) -> TextJob {
        let items = self
            .sessions
            .iter()
            .filter_map(|s| s.generation.deferred_text().map(|t| (t, s.stream.then(|| s.events.clone()))))
            .collect();
        TextJob { items }
    }

    pub(crate) fn set_stream_release(&mut self, release: crate::host::arrival::Release) {
        self.stream_release = Some(release);
    }

    /// No admitted and no waiting request.
    pub fn idle(&self) -> bool {
        self.sessions.is_empty() && self.waiting.is_empty()
    }

    pub fn active(&self) -> usize {
        self.sessions.len()
    }

    /// Accepts a job: an invalid one (among them one whose prompt plus `max_new_tokens` passes
    /// `Limits::context`) fails at once with an error event, an empty one finishes at once, the
    /// rest wait for admission in arrival order. Other requests are not affected.
    pub fn submit(&mut self, tokenizer: &Tokenizer, job: Job) {
        self.submit_at(tokenizer, job, Instant::now());
    }

    pub fn submit_at(&mut self, tokenizer: &Tokenizer, job: Job, at: Instant) {
        let arrival = self.arrive_at(tokenizer, job, at);
        self.enqueue(arrival);
    }

    /// Tokenizes a job on arrival, immediately answering empty or invalid requests.
    pub fn arrive_at(&self, tokenizer: &Tokenizer, job: Job, at: Instant) -> Arrival {
        let events = match &self.stream_release {
            Some(release) if job.stream => release.events(job.events, Some(at + crate::host::arrival::FIRST_CHUNK_HOLD)),
            _ => Events::from(job.events),
        };
        let empty = job.request.max_new_tokens == 0;
        match Generation::new(tokenizer, job.request, self.limits.context) {
            Ok(mut generation) if empty => {
                let output = generation.advance(None, |_| true).unwrap().unwrap();
                let _ = events.send(StreamEvent::Done(output));
                Arrival(None)
            }
            Ok(generation) => Arrival(Some((generation, events, job.stream, at))),
            Err(error) => {
                let _ = events.send(StreamEvent::Error(error.to_string()));
                Arrival(None)
            }
        }
    }

    /// `submit`'s second half: the request numbered and queued, in arrival order.
    pub fn enqueue(&mut self, arrival: Arrival) {
        let id = self.submitted;
        self.submitted += 1;
        if let Arrival(Some((generation, events, stream, _))) = arrival {
            self.waiting.push_back(Waiting { id, generation, events, stream });
        }
    }

    /// Admits waiting requests, first in first out, while an owner slot and the request's whole
    /// Full-page budget are free.
    pub fn admit(&mut self) {
        while let Some(next) = self.waiting.front() {
            let Some((owner, pages)) = self.allocator.admit(next.generation.max_positions()) else { break };
            let next = self.waiting.pop_front().unwrap();
            let banned = Rc::new(next.generation.banned());
            self.sessions.push(Session {
                retired: false,
                id: next.id,
                generation: next.generation,
                owner,
                pages,
                banned,
                events: next.events,
                stream: next.stream,
            });
        }
    }

    pub(crate) fn media_requests(&self) -> Vec<crate::host::media::MediaRequest<'_>> {
        self.sessions.iter().map(|s| (s.id, Some(s.owner), &s.generation))
            .chain(self.waiting.iter().map(|s| (s.id, None, &s.generation)))
            .filter_map(|(id, owner, generation)| {
                if generation.remaining_prompt()==0 {return None;}
                let (image, audio) = generation.media_inputs();
                (image.is_some() || audio.is_some()).then_some(crate::host::media::MediaRequest { id, owner, stage:owner.is_some(), image, audio })
            }).collect()
    }

    /// The step's stream events (text deltas, then each finished
    /// request's `Done` or `Error`), in order, for the caller to send once the next step is in
    /// flight (`host::worker`). A request whose client left is noticed at the send and keeps
    /// generating to its limit.
    pub fn finish_step_deferred(&mut self, rows: &[usize], picks: Vec<Option<usize>>) -> Vec<(Events, StreamEvent)> {
        let mut out: Vec<(Events, StreamEvent)> = Vec::new();
        let mut picks = picks.into_iter();
        let mut index = 0;
        let allocator = &mut self.allocator;
        let defer = self.defer_text;
        let leased = &self.leased;
        self.sessions.retain_mut(|session| {
            session.generation.set_defer(defer);
            let n = rows[index];
            index += 1;
            if session.retired {
                if n > 0 { let _ = picks.next(); } // The one row already queued after EOS/cancellation.
                if leased.contains(&session.id) { return true; }
                allocator.release(session.owner, std::mem::take(&mut session.pages));
                return false;
            }
            if n == 0 { return true; }
            let last = picks.next().flatten();
            let (stream, events) = (session.stream, &session.events);
            let result = session.generation.advance_rows(n, last, |delta| {
                if stream {
                    out.push((
                        events.clone(),
                        match delta {
                            Delta::Content(text) => StreamEvent::Delta(text.to_owned()),
                            Delta::Reasoning(text) => StreamEvent::ReasoningDelta(text.to_owned()),
                        },
                    ));
                }
                true
            });
            let event = match result {
                Ok(None) => return true,
                Ok(Some(output)) => StreamEvent::Done(output),
                Err(error) => StreamEvent::Error(error.to_string()),
            };
            out.push((session.events.clone(), event));
            if leased.contains(&session.id) {
                session.retired = true;
                return true;
            }
            allocator.release(session.owner, std::mem::take(&mut session.pages));
            false
        });
        out
    }

    /// Fails every admitted and waiting request (the device is no longer usable).
    pub fn fail_all(&mut self, message: &str) {
        for session in self.sessions.drain(..) {
            if !session.retired { let _ = session.events.send(StreamEvent::Error(message.to_owned())); }
        }
        for waiting in self.waiting.drain(..) {
            let _ = waiting.events.send(StreamEvent::Error(message.to_owned()));
        }
    }
}
