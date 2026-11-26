//! One outstanding step's scheduling state, identified by request id rather than vector position.
use super::*;
use std::borrow::Cow;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PendingRows(pub Vec<(usize, usize)>);
impl PendingRows {
    pub fn followed_by(&self, next: &Self) -> Self {
        let mut all = self.0.clone();
        for &(id, n) in &next.0 {
            if let Some((_, previous)) = all.iter_mut().find(|(i, _)| *i == id) { *previous += n; }
            else { all.push((id, n)); }
        }
        Self(all)
    }
    fn rows(&self, id: usize) -> usize { self.0.iter().find(|&&(i, _)| i == id).map_or(0, |&(_, n)| n) }
}

struct Projected<'a> {
    id: usize,
    generation: &'a Generation,
    owner: usize,
    pages: Cow<'a, [usize]>,
    pending: usize,
}

/// Host-owned row metadata for launching while the previous picks advance the driver.
pub(crate) struct PipelineSpan {
    owner: usize,
    pages: Vec<usize>,
    start: usize,
    input: PipelineInput,
    banned: Option<Rc<HashSet<usize>>>,
}

/// The request data needed to stage the next step's media while the live driver can advance.
pub(crate) struct PipelineMedia {
    id: usize,
    owner: Option<usize>,
    stage: bool,
    request: Rc<crate::host::generate::GenerationRequest>,
}

impl PipelineMedia {
    pub(crate) fn request(&self) -> crate::host::media::MediaRequest<'_> {
        crate::host::media::MediaRequest { id: self.id, owner: self.owner, stage: self.stage,
            image: self.request.image.as_ref(), audio: self.request.audio.as_ref() }
    }
}

enum PipelineInput {
    Prompt(Rc<crate::host::generate::GenerationRequest>, usize),
    Decode(usize),
    Feedback,
}

impl PipelineInput {
    fn project(g: &Generation, pending: usize, rows: usize) -> Result<(usize, Self, bool), BoxError> {
        let (first, position) = g.input();
        let remaining = g.remaining_prompt().saturating_sub(pending);
        if remaining > 0 {
            if rows > remaining { return Err("pipeline prompt plan exceeds pending inputs".into()); }
            return Ok((position + pending, Self::Prompt(g.shared_request(), rows), rows == remaining));
        }
        if rows != 1 || selections_after(g, pending) >= g.remaining_picks() {
            return Err("pipeline decoded past generation budget".into());
        }
        let input = if pending > 0 { Self::Feedback } else { Self::Decode(first) };
        Ok((position + pending, input, true))
    }

    fn decode(&self) -> bool {
        !matches!(self, Self::Prompt(..))
    }

    fn rows(&self, start: usize, owner: usize) -> Result<Vec<RowInput<'_>>, BoxError> {
        Ok(match self {
            Self::Decode(token) => vec![RowInput::Token(*token)],
            Self::Feedback => vec![RowInput::DeviceToken(owner)],
            Self::Prompt(req, rows) => (start..start + rows).map(|p| {
                Ok(match req.soft_at[p] {
                    None => RowInput::Token(req.prompt_ids[p]),
                    Some(crate::host::generate::SoftToken::Image(i)) => RowInput::Image(req.image.as_ref().ok_or("image is missing")?, i),
                    Some(crate::host::generate::SoftToken::Audio(i)) => RowInput::Audio(req.audio.as_ref().ok_or("audio is missing")?, i),
                })
            }).collect::<Result<Vec<_>, BoxError>>()?,
        })
    }
}

impl PipelineSpan {
    pub(crate) fn span(&self) -> Result<Span<'_>, BoxError> {
        Ok(Span { owner: self.owner, pages: &self.pages, start: self.start,
            inputs: self.input.rows(self.start, self.owner)?, banned: self.banned.as_deref(), decode: self.input.decode() })
    }
}

impl Driver {
    /// Plan after the outstanding rows, without using their picks. Retirement is delayed until its lease ends.
    pub fn pipeline_plan(&self, pending: &PendingRows) -> PendingRows {
        let live = self.projected(pending);
        let packed = self.pack_projected(&live);
        PendingRows(live.iter().zip(packed).filter(|(_, n)| *n > 0).map(|(s, n)| (s.id, n)).collect())
    }

    fn projected(&self, pending: &PendingRows) -> Vec<Projected<'_>> {
        self.sessions.iter().filter_map(|s| {
            let n = pending.rows(s.id);
            (!s.retired && selections_after(&s.generation, n) < s.generation.remaining_picks()).then(|| Projected {
                id: s.id, generation: &s.generation, owner: s.owner, pages: Cow::Borrowed(&s.pages),
                pending: n,
            })
        }).collect()
    }

    fn pack_projected(&self, live: &[Projected<'_>]) -> Vec<usize> {
        let demands: Vec<_> = live.iter().map(|s| {
            let g = s.generation;
            let n = s.pending;
            Demand { remaining_prompt: g.remaining_prompt().saturating_sub(n),
                position: (g.prompt_position() + n).min(g.prompt_len()) }
        }).collect();
        schedule::pack(&demands)
    }

    /// Predict the plan after retiring `prior`, with `next` still holding its
    /// leases. Include arrivals already seen by the worker during `next`.
    pub(crate) fn pipeline_forecast_rows(&self, pending: &PendingRows, prior: &PendingRows, next: &PendingRows, arrivals: &[Arrival])
        -> Option<(PendingRows, Vec<AheadSpan>)>
    {
        let mut live = self.projected(pending);
        if !self.waiting.is_empty() || !arrivals.is_empty() {
            let mut allocator = self.allocator.clone();
            for s in &self.sessions {
                if next.rows(s.id) == 0 && (s.retired
                    || selections_after(&s.generation, prior.rows(s.id)) >= s.generation.remaining_picks())
                {
                    allocator.release(s.owner, s.pages.clone());
                }
            }
            let waiting = self.waiting.iter().map(|w| (w.id, &w.generation));
            let extra = arrivals.iter().enumerate().filter_map(|(i, a)| {
                a.0.as_ref().map(|(g, _, _, _)| (self.submitted + i, g))
            });
            for (id, generation) in waiting.chain(extra) {
                let Some((owner, pages)) = allocator.admit(generation.max_positions()) else { break };
                live.push(Projected { id, generation, owner, pages: Cow::Owned(pages), pending: 0 });
            }
        }
        let packed = self.pack_projected(&live);
        let mut spans = Vec::new();
        let mut rows = Vec::new();
        for (s, count) in live.into_iter().zip(packed).filter(|(_, n)| *n != 0) {
            let (start, input, emits) = PipelineInput::project(s.generation, s.pending, count).ok()?;
            let inputs = input.rows(start, s.owner).ok()?;
            let tokens = inputs.iter().map(|i| match i { RowInput::DeviceToken(_) => None, _ => Some(i.token()) }).collect();
            rows.push((s.id, count));
            spans.push(AheadSpan { owner: s.owner, pages: s.pages.into_owned(), start, tokens,
                decode: input.decode(), emits });
        }
        (!spans.is_empty()).then_some((PendingRows(rows), spans))
    }

    pub(crate) fn media_for_rows(&self, rows: &PendingRows) -> Vec<crate::host::media::MediaRequest<'_>> {
        let mut requests = self.media_requests();
        for request in &mut requests {
            if request.owner.is_some() { request.stage = rows.rows(request.id) > 0; }
        }
        requests
    }

    pub(crate) fn pipeline_media_preview(&self, pending: &PendingRows) -> Vec<PipelineMedia> {
        let mut media: Vec<_> = self.sessions.iter().map(|s| (s.id, Some(s.owner), &s.generation))
            .chain(self.waiting.iter().map(|w| (w.id, None, &w.generation)))
            .filter(|(_, _, g)| g.remaining_prompt() > 0 && g.media() != (false, false))
            .map(|(id, owner, g)| PipelineMedia { id, owner, stage: owner.is_none(), request: g.shared_request() })
            .collect();
        if media.iter().any(|m| m.owner.is_some()) {
            let rows = self.pipeline_plan(pending);
            for m in &mut media {
                m.stage |= rows.rows(m.id) > 0;
            }
        }
        media
    }

    pub(crate) fn pipeline_snapshot(&self, pending: &PendingRows, next: &PendingRows) -> Result<Vec<PipelineSpan>, BoxError> {
        next.0.iter().map(|&(id, rows)| {
            let s = self.sessions.iter().find(|s| s.id == id && !s.retired).ok_or("pipeline request missing")?;
            let (start, input, emits) = PipelineInput::project(&s.generation, pending.rows(id), rows)?;
            Ok(PipelineSpan { owner: s.owner, pages: s.pages.clone(), start, input,
                banned: emits.then(|| s.banned.clone()) })
        }).collect()
    }

    /// Commit the oldest step. `next` names rows already queued, so their owners cannot yet be reused.
    pub fn pipeline_finish(&mut self, rows: &PendingRows, picks: Vec<Option<usize>>, next: &PendingRows) -> Vec<(Events, StreamEvent)> {
        assert_eq!(rows.0.len(), picks.len());
        self.leased.clear();
        self.leased.extend(next.0.iter().map(|&(id, _)| id));
        let mut results: std::collections::HashMap<_, _> =
            rows.0.iter().zip(picks).map(|(&(id, n), pick)| (id, (n, pick))).collect();
        let mut aligned = Vec::with_capacity(self.sessions.len());
        let mut ordered = Vec::with_capacity(rows.0.len());
        for s in &self.sessions {
            if let Some((n, pick)) = results.remove(&s.id) {
                aligned.push(n);
                ordered.push(pick);
            } else {
                aligned.push(0);
            }
        }
        debug_assert!(results.is_empty(), "step result lost its request lease");
        self.finish_step_deferred(&aligned, ordered)
    }
}

fn selections_after(g: &Generation, rows: usize) -> usize {
    let prompt = g.remaining_prompt();
    if prompt == 0 { rows } else if rows < prompt { 0 } else { rows - prompt + 1 }
}
