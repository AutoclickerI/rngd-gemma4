//! Deadline-based first-chunk release, independent of device progress and detokenization.
use std::collections::HashMap;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use crate::api::worker::{Job, StreamEvent};

pub const FIRST_CHUNK_HOLD: Duration = Duration::from_millis(2000);

/// Timestamp each request before tokenization and admission.
pub fn arrivals(jobs: mpsc::Receiver<Job>) -> mpsc::Receiver<(Job, Instant)> {
    let (tx, rx) = mpsc::channel();
    std::thread::Builder::new().name("request-arrivals".into()).spawn(move || {
        for job in jobs {
            if let Err(mpsc::SendError((job, _))) = tx.send((job, Instant::now())) {
                let _ = job.events.send(StreamEvent::Error("generation worker stopped".into()));
                break;
            }
        }
    }).expect("start request arrival clock");
    rx
}

enum Command {
    Open(usize, mpsc::Sender<StreamEvent>, Instant),
    Event(usize, StreamEvent),
}

#[derive(Clone)]
pub struct Events {
    direct: mpsc::Sender<StreamEvent>,
    relay: Option<(mpsc::Sender<Command>, usize)>,
}

impl From<mpsc::Sender<StreamEvent>> for Events {
    fn from(direct: mpsc::Sender<StreamEvent>) -> Self { Self { direct, relay: None } }
}

impl Events {
    pub fn send(&self, event: StreamEvent) -> Result<(), mpsc::SendError<StreamEvent>> {
        match &self.relay {
            None => self.direct.send(event),
            Some((tx, id)) => tx.send(Command::Event(*id, event)).map_err(|_| {
                // A failed relay must be visible; forwarding the terminal event could conceal lost deltas.
                let message = "stream release thread stopped".to_owned();
                let _ = self.direct.send(StreamEvent::Error(message.clone()));
                mpsc::SendError(StreamEvent::Error(message))
            }),
        }
    }
}

pub struct Release {
    tx: mpsc::Sender<Command>,
    next: std::cell::Cell<usize>,
}

impl Release {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        std::thread::Builder::new().name("stream-release".into()).spawn(move || run(rx))
            .expect("start first-chunk release timer");
        Self { tx, next: std::cell::Cell::new(0) }
    }

    pub fn events(&self, direct: mpsc::Sender<StreamEvent>, deadline: Option<Instant>) -> Events {
        let Some(deadline) = deadline else { return Events::from(direct); };
        let id = self.next.get();
        self.next.set(id + 1);
        self.tx.send(Command::Open(id, direct.clone(), deadline)).expect("stream release thread stopped");
        Events { direct, relay: Some((self.tx.clone(), id)) }
    }
}

struct Pending {
    events: mpsc::Sender<StreamEvent>,
    deadline: Instant,
    released: bool,
    buffered: Vec<StreamEvent>,
}

impl Pending {
    fn flush(&mut self) -> bool {
        self.released = true;
        let mut done = false;
        for event in self.buffered.drain(..) {
            done |= matches!(event, StreamEvent::Done(_));
            let _ = self.events.send(event);
        }
        done
    }

    fn event(&mut self, event: StreamEvent, now: Instant) -> bool {
        let terminal = matches!(&event, StreamEvent::Done(_) | StreamEvent::Error(_));
        if matches!(event, StreamEvent::Error(_)) || now >= self.deadline {
            self.flush();
        }
        if self.released { let _ = self.events.send(event); } else { self.buffered.push(event); }
        terminal && self.released
    }
}

fn run(rx: mpsc::Receiver<Command>) {
    let mut pending: HashMap<usize, Pending> = HashMap::new();
    loop {
        let now = Instant::now();
        pending.retain(|_, p| p.released || now < p.deadline || !p.flush());
        let next = pending.values().filter(|p| !p.released).map(|p| p.deadline).min();
        let command = match next {
            Some(at) => match rx.recv_timeout(at.saturating_duration_since(Instant::now())) {
                Ok(command) => command,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            },
            None => match rx.recv() { Ok(command) => command, Err(_) => break },
        };
        match command {
            Command::Open(id, events, deadline) => {
                pending.insert(id, Pending { events, deadline, released: false, buffered: Vec::new() });
            }
            Command::Event(id, event) => {
                if pending.get_mut(&id).is_some_and(|p| p.event(event, Instant::now())) { pending.remove(&id); }
            }
        }
    }
    for (_, mut request) in pending {
        if !request.flush() {
            let _ = request.events.send(StreamEvent::Error("generation worker stopped before completion".into()));
        }
    }
}
