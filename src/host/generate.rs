use crate::host::audio::AudioFrames;
use crate::host::image::ImagePatches;
use crate::host::sampling::SamplingConfig;
use crate::host::tokenizer::Tokenizer;
use llm_tokenizer::DecodeStream;
use std::collections::HashSet;

pub const EOS_TOKEN: usize = 1;
pub const END_OF_TURN_TOKEN: usize = 106;
pub const TOOL_RESPONSE_TOKEN: usize = 50;

pub const EOS_TOKENS: [usize; 3] = [EOS_TOKEN, END_OF_TURN_TOKEN, TOOL_RESPONSE_TOKEN];

pub const CHANNEL_OPEN_TOKEN: usize = 100;
pub const CHANNEL_CLOSE_TOKEN: usize = 101;

const CHANNEL_OPEN_TEXT: &str = "<|channel>";
const CHANNEL_CLOSE_TEXT: &str = "<channel|>";

pub const DEFAULT_MAX_NEW_TOKENS: usize = 300;
pub const BOI_TOKEN: usize = 255999;
pub const IMAGE_TOKEN: usize = 258880;
pub const EOI_TOKEN: usize = 258882;
pub const BOA_TOKEN: usize = 256000;
pub const AUDIO_TOKEN: usize = 258881;
pub const EOA_TOKEN: usize = 258883;

const SUPPRESSED_TOKENS: [usize; 2] = [EOI_TOKEN, EOA_TOKEN];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Channel {
    Answer,
    Name,
    Reasoning,
}

#[derive(Clone, Copy, Debug)]
pub enum Delta<'a> {
    Content(&'a str),
    Reasoning(&'a str),
}

#[derive(Clone, Copy)]
pub enum SoftToken {
    Image(usize),
    Audio(usize),
}

pub fn splice_multimodal_tokens(
    ids: Vec<usize>,
    image: Option<&ImagePatches>,
    audio: Option<&AudioFrames>,
) -> Result<(Vec<usize>, Vec<Option<SoftToken>>), String> {
    let image_markers = ids.iter().filter(|&&id| id == IMAGE_TOKEN).count();
    let audio_markers = ids.iter().filter(|&&id| id == AUDIO_TOKEN).count();
    if image.is_some() && image_markers != 1 {
        return Err(format!("expected one image placeholder, found {image_markers}"));
    }
    if audio.is_some() && audio_markers != 1 {
        return Err(format!("expected one audio placeholder, found {audio_markers}"));
    }

    let image_len = image.map_or(0, |value| value.pixels.len());
    let audio_len = audio.map_or(0, AudioFrames::len);
    let mut out_ids = Vec::with_capacity(ids.len() + image_len + audio_len + 4);
    let mut soft_at = Vec::with_capacity(out_ids.capacity());
    for id in ids {
        match id {
            IMAGE_TOKEN if image.is_some() => {
                out_ids.push(BOI_TOKEN);
                soft_at.push(None);
                for index in 0..image_len {
                    out_ids.push(IMAGE_TOKEN);
                    soft_at.push(Some(SoftToken::Image(index)));
                }
                out_ids.push(EOI_TOKEN);
                soft_at.push(None);
            }
            AUDIO_TOKEN if audio.is_some() => {
                out_ids.push(BOA_TOKEN);
                soft_at.push(None);
                for index in 0..audio_len {
                    out_ids.push(AUDIO_TOKEN);
                    soft_at.push(Some(SoftToken::Audio(index)));
                }
                out_ids.push(EOA_TOKEN);
                soft_at.push(None);
            }
            _ => {
                out_ids.push(id);
                soft_at.push(None);
            }
        }
    }
    Ok((out_ids, soft_at))
}

fn pending_stop_overlap(text: &str, stop_strings: &[&str]) -> usize {
    let mut hold = 0usize;
    for &stop in stop_strings {
        let max_len = (stop.len() - 1).min(text.len());
        for len in (1..=max_len).rev() {
            if stop.is_char_boundary(len) && text.ends_with(&stop[..len]) {
                hold = hold.max(len);
                break;
            }
        }
    }
    hold
}

fn step_over_marker(
    stream: &mut DecodeStream,
    id: usize,
    marker: &str,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    let Some(delta) = stream.step(id as u32)? else {
        return Ok(None);
    };
    let held = delta.strip_suffix(marker).unwrap_or(&delta);
    Ok((!held.is_empty()).then(|| held.to_owned()))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}

pub struct GenerationRequest {
    pub prompt_ids: Vec<usize>,
    pub soft_at: Vec<Option<SoftToken>>,
    pub image: Option<ImagePatches>,
    pub audio: Option<AudioFrames>,
    pub max_new_tokens: usize,
    pub sampling: SamplingConfig,
    pub stop_strings: Vec<String>,
    pub separate_reasoning: bool,
    /// Never stop at an end-of-sequence token: generate exactly `max_new_tokens` tokens.
    /// The startup warm-up uses this for fixed-length decoding.
    pub ignore_eos: bool,
}

pub struct GenerationOutput {
    pub text: String,
    pub reasoning: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: FinishReason,
}

/// One conversation's progress, independent of the order in which requests run.
pub struct Generation {
    request: std::rc::Rc<GenerationRequest>,
    phase: Phase,
    /// The text side (detokenization, channels, stop strings), shared with a deferred text job (`TextState`).
    text: std::rc::Rc<std::cell::RefCell<TextState>>,
    /// Counts and reason; its text and reasoning live in `text` until `finish`.
    output: GenerationOutput,
    /// `detok`: selected tokens wait in `TextState::deferred` instead of being detokenized at once, when the request
    /// has no stop string (its end is then decided by tokens alone) and its first content has gone out (the first
    /// content delta, which TTFT measures, is never deferred).
    defer: bool,
}

/// Incremental detokenization, channel boundaries and stop strings. After the first content
/// delta, requests without stop strings can defer text until the next device step. Finishing
/// catches up before returning text.
pub struct TextState {
    stream: DecodeStream,
    channel: Channel,
    text: String,
    reasoning: String,
    emitted: usize,
    stop_strings: Vec<String>,
    separate_reasoning: bool,
    deferred: Vec<usize>,
    /// A deferred token's detokenization error, reported by the generation's next `advance`.
    failed: Option<String>,
}

#[derive(Clone, Copy)]
enum Phase {
    Prompt(usize),
    Decode(usize),
    Complete,
}

impl Generation {
    pub fn new(
        tokenizer: &Tokenizer,
        request: GenerationRequest,
        capacity: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if request.prompt_ids.is_empty() {
            return Err("prompt produced no tokens".into());
        }
        if request.prompt_ids.len().saturating_add(request.max_new_tokens) > capacity {
            return Err(format!("prompt plus generation exceeds {capacity} positions").into());
        }
        if request.prompt_ids.len() != request.soft_at.len() {
            return Err(format!(
                "prompt_ids ({}) and soft_at ({}) must be parallel",
                request.prompt_ids.len(),
                request.soft_at.len()
            )
            .into());
        }
        let text = TextState {
            stream: tokenizer.decode_stream(&request.prompt_ids.iter().map(|&id| id as u32).collect::<Vec<_>>()),
            channel: Channel::Answer,
            text: String::new(),
            reasoning: String::new(),
            emitted: 0,
            stop_strings: request.stop_strings.iter().filter(|s| !s.is_empty()).cloned().collect(),
            separate_reasoning: request.separate_reasoning,
            deferred: Vec::new(),
            failed: None,
        };
        Ok(Self {
            text: std::rc::Rc::new(std::cell::RefCell::new(text)),
            output: GenerationOutput {
                text: String::new(),
                reasoning: String::new(),
                prompt_tokens: request.prompt_ids.len(),
                completion_tokens: 0,
                finish_reason: FinishReason::Length,
            },
            request: std::rc::Rc::new(request),
            phase: Phase::Prompt(0),
            defer: false,
        })
    }

    /// `detok`: whether the next selected tokens may wait for `TextState::catch_up` (only without stop strings and after
    /// the first content delta).
    pub fn set_defer(&mut self, on: bool) {
        self.defer = on;
    }

    /// The text side, for a deferred text job, when it holds tokens still to detokenize.
    pub fn deferred_text(&self) -> Option<std::rc::Rc<std::cell::RefCell<TextState>>> {
        (!self.text.borrow().deferred.is_empty()).then(|| self.text.clone())
    }

    /// The next token's ID and position.
    pub(crate) fn input(&self) -> (usize, usize) {
        match self.phase {
            Phase::Prompt(pos) => (self.request.prompt_ids[pos], pos),
            Phase::Decode(token) => (token, self.request.prompt_ids.len() + self.output.completion_tokens - 1),
            Phase::Complete => unreachable!("completed requests have no input"),
        }
    }

    fn wants_logits(&self) -> bool {
        self.request.max_new_tokens > 0 && self.remaining_prompt() <= 1
    }

    pub fn advance(
        &mut self,
        token: Option<usize>,
        mut on_delta: impl FnMut(Delta) -> bool,
    ) -> Result<Option<GenerationOutput>, Box<dyn std::error::Error>> {
        let phase = std::mem::replace(&mut self.phase, Phase::Complete);
        if matches!(phase, Phase::Complete) {
            return Err("generation is already complete".into());
        }
        if let Some(error) = self.text.borrow_mut().failed.take() {
            return Err(error.into());
        }
        if self.request.max_new_tokens == 0 {
            return Ok(Some(self.finish(FinishReason::Length, &mut on_delta)?));
        }
        if let Phase::Prompt(pos) = phase
            && pos + 1 < self.request.prompt_ids.len()
        {
            self.phase = Phase::Prompt(pos + 1);
            return Ok(None);
        }
        let next = token.ok_or("missing selection for an emitting row")?;
        if EOS_TOKENS.contains(&next) {
            return Ok(Some(self.finish(FinishReason::Stop, &mut on_delta)?));
        }
        let mut text = self.text.borrow_mut();
        if self.defer && text.stop_strings.is_empty() && text.emitted > 0 {
            text.deferred.push(next);
            drop(text);
        } else {
            text.catch_up(&mut on_delta)?;
            let finished = text.emit(next, &mut on_delta)?;
            drop(text);
            if let Some(reason) = finished {
                return Ok(Some(self.finish(reason, &mut on_delta)?));
            }
        }
        self.output.completion_tokens += 1;
        if self.output.completion_tokens == self.request.max_new_tokens {
            return Ok(Some(self.finish(FinishReason::Length, &mut on_delta)?));
        }
        // The next turn computes the token just emitted, so every turn has one forward.
        self.phase = Phase::Decode(next);
        Ok(None)
    }

    /// Token ids selection must skip for this request.
    pub fn banned(&self) -> HashSet<usize> {
        let mut banned: HashSet<usize> = SUPPRESSED_TOKENS.into_iter().collect();
        if self.request.ignore_eos {
            // Channel tokens are banned too: text in the reasoning channel is not part of the
            // answer, so it would shorten the visible output.
            banned.extend(EOS_TOKENS);
            banned.extend([CHANNEL_OPEN_TOKEN, CHANNEL_CLOSE_TOKEN]);
        }
        banned
    }

    pub(crate) fn shared_request(&self) -> std::rc::Rc<GenerationRequest> {
        self.request.clone()
    }

    pub(crate) fn media_inputs(&self) -> (Option<&ImagePatches>, Option<&AudioFrames>) {
        (self.request.image.as_ref(), self.request.audio.as_ref())
    }

    /// Whether the request carries an image and audio (its rows' precision class, `engine::precision`).
    pub fn media(&self) -> (bool, bool) {
        (self.request.image.is_some(), self.request.audio.is_some())
    }

    /// The next prompt position, or the prompt length once decoding.
    pub fn prompt_position(&self) -> usize {
        match self.phase {
            Phase::Prompt(pos) => pos,
            _ => self.request.prompt_ids.len(),
        }
    }

    /// Prompt tokens not yet computed (0 once decoding).
    pub fn remaining_prompt(&self) -> usize {
        match self.phase {
            Phase::Prompt(pos) => self.request.prompt_ids.len() - pos,
            _ => 0,
        }
    }

    /// Remaining selections, independent of their values (the pipeline's scheduling budget).
    pub fn remaining_picks(&self) -> usize { self.request.max_new_tokens.saturating_sub(self.output.completion_tokens) }

    /// The prompt's length.
    pub fn prompt_len(&self) -> usize {
        self.request.prompt_ids.len()
    }

    /// Positions the request can occupy: its prompt plus its generation budget.
    pub fn max_positions(&self) -> usize {
        self.request.prompt_ids.len() + self.request.max_new_tokens
    }

    /// Consumes `rows` pending inputs one step computed. Only the last can want logits; `last`
    /// carries its selection input (ignored when it wants none).
    pub fn advance_rows(
        &mut self,
        rows: usize,
        last: Option<usize>,
        mut on_delta: impl FnMut(Delta) -> bool,
    ) -> Result<Option<GenerationOutput>, Box<dyn std::error::Error>> {
        if rows == 0 {
            return Err("a step consumes at least one input".into());
        }
        for _ in 1..rows {
            if self.wants_logits() {
                return Err("only the last input of a step can want logits".into());
            }
            if let Some(output) = self.advance(None, &mut on_delta)? {
                return Ok(Some(output));
            }
        }
        if last.is_none() && self.wants_logits() {
            return Err("missing selection for an emitting row".into());
        }
        self.advance(last, on_delta)
    }

    fn finish(&mut self, reason: FinishReason, on_delta: &mut impl FnMut(Delta) -> bool) -> Result<GenerationOutput, Box<dyn std::error::Error>> {
        let mut text = self.text.borrow_mut();
        text.catch_up(on_delta)?;
        if text.emitted < text.text.len() {
            let _ = on_delta(Delta::Content(&text.text[text.emitted..]));
        }
        self.phase = Phase::Complete;
        Ok(GenerationOutput {
            text: std::mem::take(&mut text.text),
            reasoning: std::mem::take(&mut text.reasoning),
            prompt_tokens: self.output.prompt_tokens,
            completion_tokens: self.output.completion_tokens,
            finish_reason: reason,
        })
    }
}

impl TextState {
    /// Detokenizes and streams the tokens still waiting (`Generation::set_defer`), in order. They come from requests
    /// without stop strings, so none of them ends the request.
    pub fn catch_up(&mut self, on_delta: &mut impl FnMut(Delta) -> bool) -> Result<(), Box<dyn std::error::Error>> {
        if self.deferred.is_empty() {
            return Ok(());
        }
        for token in std::mem::take(&mut self.deferred) {
            self.emit(token, on_delta)?;
        }
        Ok(())
    }

    /// `catch_up` for a deferred text job: an error waits for the generation's next `advance`.
    pub fn catch_up_or_hold(&mut self, mut on_delta: impl FnMut(Delta) -> bool) {
        if let Err(error) = self.catch_up(&mut on_delta) {
            self.failed = Some(error.to_string());
        }
    }

    fn emit(
        &mut self,
        token: usize,
        on_delta: &mut impl FnMut(Delta) -> bool,
    ) -> Result<Option<FinishReason>, Box<dyn std::error::Error>> {
        let (delta, switch_to) = match token {
            CHANNEL_OPEN_TOKEN if self.separate_reasoning => (
                step_over_marker(&mut self.stream, token, CHANNEL_OPEN_TEXT)?,
                Some(Channel::Name),
            ),
            CHANNEL_CLOSE_TOKEN if self.separate_reasoning => (
                step_over_marker(&mut self.stream, token, CHANNEL_CLOSE_TEXT)?,
                Some(Channel::Answer),
            ),
            _ => (self.stream.step(token as u32)?, None),
        };
        if let Some(delta) = delta {
            let keep = match self.channel {
                Channel::Name => {
                    if let Some((_, rest)) = delta.split_once('\n') {
                        self.channel = Channel::Reasoning;
                        self.reasoning.push_str(rest);
                        rest.is_empty() || on_delta(Delta::Reasoning(rest))
                    } else {
                        true
                    }
                }
                Channel::Reasoning => {
                    self.reasoning.push_str(&delta);
                    on_delta(Delta::Reasoning(&delta))
                }
                Channel::Answer => {
                    self.text.push_str(&delta);
                    let stops: Vec<_> = self.stop_strings.iter().map(String::as_str).collect();
                    if let Some(cut) = stops.iter().filter_map(|s| self.text.find(s)).min() {
                        if cut > self.emitted {
                            let _ = on_delta(Delta::Content(&self.text[self.emitted..cut]));
                        }
                        self.text.truncate(cut);
                        self.emitted = cut;
                        return Ok(Some(FinishReason::Stop));
                    }
                    let safe = self.text.len() - pending_stop_overlap(&self.text, &stops);
                    if safe > self.emitted {
                        let keep = on_delta(Delta::Content(&self.text[self.emitted..safe]));
                        self.emitted = safe;
                        keep
                    } else {
                        true
                    }
                }
            };
            if !keep {
                return Ok(Some(FinishReason::Length));
            }
        }
        if let Some(channel) = switch_to {
            self.channel = channel;
        }
        Ok(None)
    }
}
