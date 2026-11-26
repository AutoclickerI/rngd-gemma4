//! A small LZMA decoder for the `.lzma` ("alone") streams our carried payloads use
//! (`tools/native/prebuilt/pack.py`, Python's `lzma` module, `FORMAT_ALONE`, preset 9e). The crate cannot
//! add a dependency (the grader builds `src/` with its own manifest), so the payloads' decoder lives
//! here. It follows Igor Pavlov's public-domain reference decoder (`LzmaSpec.cpp`, LZMA SDK) and
//! decodes into one output vector, which is also the dictionary.
//!
//! Stream: 1 property byte (lc, lp, pb), the dictionary size (u32 LE), the uncompressed size (u64
//! LE, all ones when unknown, as liblzma's alone encoder writes it), then the range-coded data,
//! ended by the end-of-payload marker or by the known size.

const PROB_BITS: u32 = 11;
const PROB_INIT: u16 = 1 << (PROB_BITS - 1);
const MOVE_BITS: u32 = 5;
const TOP: u32 = 1 << 24;
const STATES: usize = 12;
const POS_BITS_MAX: usize = 4;
const LEN_TO_POS_STATES: usize = 4;
const ALIGN_BITS: u32 = 4;
const END_POS_MODEL_INDEX: u32 = 14;
const FULL_DISTANCES: usize = 1 << (END_POS_MODEL_INDEX >> 1);
const MATCH_MIN_LEN: usize = 2;

struct RangeDecoder<'a> {
    input: &'a [u8],
    at: usize,
    range: u32,
    code: u32,
}

impl<'a> RangeDecoder<'a> {
    fn new(input: &'a [u8]) -> Result<Self, String> {
        if input.len() < 5 || input[0] != 0 {
            return Err("range coder must start with a zero byte".into());
        }
        let code = u32::from_be_bytes([input[1], input[2], input[3], input[4]]);
        if code == u32::MAX {
            return Err("corrupted range coder start".into());
        }
        Ok(Self { input, at: 5, range: u32::MAX, code })
    }

    fn byte(&mut self) -> Result<u8, String> {
        let b = *self.input.get(self.at).ok_or("stream ends early")?;
        self.at += 1;
        Ok(b)
    }

    fn normalize(&mut self) -> Result<(), String> {
        if self.range < TOP {
            self.range <<= 8;
            self.code = (self.code << 8) | self.byte()? as u32;
        }
        Ok(())
    }

    fn bit(&mut self, prob: &mut u16) -> Result<u32, String> {
        let v = *prob as u32;
        let bound = (self.range >> PROB_BITS) * v;
        let bit = if self.code < bound {
            *prob = (v + (((1 << PROB_BITS) - v) >> MOVE_BITS)) as u16;
            self.range = bound;
            0
        } else {
            *prob = (v - (v >> MOVE_BITS)) as u16;
            self.code -= bound;
            self.range -= bound;
            1
        };
        self.normalize()?;
        Ok(bit)
    }

    fn direct(&mut self, bits: u32) -> Result<u32, String> {
        let mut res = 0u32;
        for _ in 0..bits {
            self.range >>= 1;
            self.code = self.code.wrapping_sub(self.range);
            let t = 0u32.wrapping_sub(self.code >> 31);
            self.code = self.code.wrapping_add(self.range & t);
            self.normalize()?;
            res = (res << 1).wrapping_add(t.wrapping_add(1));
        }
        Ok(res)
    }

    /// `bits` bits through the tree `probs` (index 1 is the root), most significant first.
    fn tree(&mut self, probs: &mut [u16], bits: u32) -> Result<u32, String> {
        let mut m = 1usize;
        for _ in 0..bits {
            m = (m << 1) + self.bit(&mut probs[m])? as usize;
        }
        Ok(m as u32 - (1 << bits))
    }

    /// `bits` bits through the tree `probs`, least significant first.
    fn reverse(&mut self, probs: &mut [u16], bits: u32) -> Result<u32, String> {
        let mut m = 1usize;
        let mut symbol = 0u32;
        for i in 0..bits {
            let bit = self.bit(&mut probs[m])?;
            m = (m << 1) + bit as usize;
            symbol |= bit << i;
        }
        Ok(symbol)
    }
}

struct LenDecoder {
    choice: u16,
    choice2: u16,
    low: [[u16; 8]; 1 << POS_BITS_MAX],
    mid: [[u16; 8]; 1 << POS_BITS_MAX],
    high: [u16; 256],
}

impl LenDecoder {
    fn new() -> Self {
        Self {
            choice: PROB_INIT,
            choice2: PROB_INIT,
            low: [[PROB_INIT; 8]; 1 << POS_BITS_MAX],
            mid: [[PROB_INIT; 8]; 1 << POS_BITS_MAX],
            high: [PROB_INIT; 256],
        }
    }

    fn decode(&mut self, rc: &mut RangeDecoder, pos_state: usize) -> Result<usize, String> {
        if rc.bit(&mut self.choice)? == 0 {
            return Ok(rc.tree(&mut self.low[pos_state], 3)? as usize);
        }
        if rc.bit(&mut self.choice2)? == 0 {
            return Ok(8 + rc.tree(&mut self.mid[pos_state], 3)? as usize);
        }
        Ok(16 + rc.tree(&mut self.high, 8)? as usize)
    }
}

/// Decodes a `.lzma` stream whose output is `expected` bytes long.
pub fn decode_alone(stream: &[u8], expected: usize) -> Result<Vec<u8>, String> {
    if stream.len() < 13 {
        return Err("stream shorter than its header".into());
    }
    let mut d = stream[0] as usize;
    if d >= 9 * 5 * 5 {
        return Err("invalid LZMA properties".into());
    }
    let lc = d % 9;
    d /= 9;
    let lp = d % 5;
    let pb = d / 5;
    let dict = u32::from_le_bytes(stream[1..5].try_into().unwrap()).max(1 << 12) as usize;
    let declared = u64::from_le_bytes(stream[5..13].try_into().unwrap());
    if declared != u64::MAX && declared != expected as u64 {
        return Err(format!("stream declares {declared} bytes, expected {expected}"));
    }
    let mut rc = RangeDecoder::new(&stream[13..])?;
    let mut out: Vec<u8> = Vec::with_capacity(expected);

    let mut literal = vec![PROB_INIT; 0x300 << (lc + lp)];
    let mut pos_slot = [[PROB_INIT; 64]; LEN_TO_POS_STATES];
    let mut pos_decoders = [PROB_INIT; 1 + FULL_DISTANCES - END_POS_MODEL_INDEX as usize];
    let mut align = [PROB_INIT; 1 << ALIGN_BITS];
    let mut is_match = [PROB_INIT; STATES << POS_BITS_MAX];
    let mut is_rep = [PROB_INIT; STATES];
    let mut is_rep_g0 = [PROB_INIT; STATES];
    let mut is_rep_g1 = [PROB_INIT; STATES];
    let mut is_rep_g2 = [PROB_INIT; STATES];
    let mut is_rep0_long = [PROB_INIT; STATES << POS_BITS_MAX];
    let mut len_decoder = LenDecoder::new();
    let mut rep_len_decoder = LenDecoder::new();
    let (mut rep0, mut rep1, mut rep2, mut rep3) = (0usize, 0usize, 0usize, 0usize);
    let mut state = 0usize;
    let pb_mask = (1usize << pb) - 1;
    let lp_mask = (1usize << lp) - 1;

    loop {
        if out.len() == expected && declared != u64::MAX {
            break;
        }
        let pos_state = out.len() & pb_mask;
        if rc.bit(&mut is_match[(state << POS_BITS_MAX) + pos_state])? == 0 {
            if out.len() == expected {
                return Err("literal past the expected size".into());
            }
            let prev = out.last().copied().unwrap_or(0) as usize;
            let lit_state = ((out.len() & lp_mask) << lc) + (prev >> (8 - lc));
            let probs = &mut literal[0x300 * lit_state..0x300 * (lit_state + 1)];
            let mut symbol = 1usize;
            if state >= 7 {
                let mut match_byte = out[out.len() - rep0 - 1] as usize;
                while symbol < 0x100 {
                    let match_bit = (match_byte >> 7) & 1;
                    match_byte <<= 1;
                    let bit = rc.bit(&mut probs[((1 + match_bit) << 8) + symbol])? as usize;
                    symbol = (symbol << 1) | bit;
                    if match_bit != bit {
                        break;
                    }
                }
            }
            while symbol < 0x100 {
                symbol = (symbol << 1) | rc.bit(&mut probs[symbol])? as usize;
            }
            out.push((symbol - 0x100) as u8);
            state = if state < 4 {
                0
            } else if state < 10 {
                state - 3
            } else {
                state - 6
            };
            continue;
        }

        let len;
        if rc.bit(&mut is_rep[state])? != 0 {
            if out.is_empty() {
                return Err("repeat match before any output".into());
            }
            if rc.bit(&mut is_rep_g0[state])? == 0 {
                if rc.bit(&mut is_rep0_long[(state << POS_BITS_MAX) + pos_state])? == 0 {
                    state = if state < 7 { 9 } else { 11 };
                    let b = out[out.len() - rep0 - 1];
                    out.push(b);
                    continue;
                }
            } else {
                let dist;
                if rc.bit(&mut is_rep_g1[state])? == 0 {
                    dist = rep1;
                } else {
                    if rc.bit(&mut is_rep_g2[state])? == 0 {
                        dist = rep2;
                    } else {
                        dist = rep3;
                        rep3 = rep2;
                    }
                    rep2 = rep1;
                }
                rep1 = rep0;
                rep0 = dist;
            }
            len = rep_len_decoder.decode(&mut rc, pos_state)?;
            state = if state < 7 { 8 } else { 11 };
        } else {
            rep3 = rep2;
            rep2 = rep1;
            rep1 = rep0;
            len = len_decoder.decode(&mut rc, pos_state)?;
            state = if state < 7 { 7 } else { 10 };
            // Distance.
            let len_state = len.min(LEN_TO_POS_STATES - 1);
            let slot = rc.tree(&mut pos_slot[len_state], 6)?;
            let dist = if slot < 4 {
                slot
            } else {
                let direct = (slot >> 1) - 1;
                let base = (2 | (slot & 1)) << direct;
                if slot < END_POS_MODEL_INDEX {
                    base + rc.reverse(&mut pos_decoders[(base - slot) as usize..], direct)?
                } else {
                    base.wrapping_add(rc.direct(direct - ALIGN_BITS)? << ALIGN_BITS).wrapping_add(rc.reverse(&mut align, ALIGN_BITS)?)
                }
            };
            if dist == u32::MAX {
                // End-of-payload marker.
                break;
            }
            rep0 = dist as usize;
            if rep0 >= dict || rep0 >= out.len() {
                return Err(format!("match distance {rep0} reaches before the output start"));
            }
        }
        let len = len + MATCH_MIN_LEN;
        if out.len() + len > expected {
            return Err("match past the expected size".into());
        }
        let from = out.len() - rep0 - 1;
        for i in 0..len {
            let b = out[from + i];
            out.push(b);
        }
    }
    if out.len() != expected {
        return Err(format!("decoded {} bytes, expected {expected}", out.len()));
    }
    Ok(out)
}
