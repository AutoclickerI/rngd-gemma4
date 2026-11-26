//! FP32 scaled embedding and RMSNorm, decomposed into three BF16 terms.

/// BF16 round-to-nearest-even of `value`, bit-identical to `half` 2.7.1 `f32_to_bf16` for every input
/// (NaN keeps its high payload and sets the quiet bit; overflow rounds to infinity).
#[inline(always)]
pub fn f32_to_bf16_bits(value: f32) -> u16 {
    let x = value.to_bits();
    let rounded = (x.wrapping_add(0x7FFF + ((x >> 16) & 1)) >> 16) as u16;
    let nan = ((x >> 16) | 0x0040) as u16;
    if x & 0x7FFF_FFFF > 0x7F80_0000 { nan } else { rounded }
}

#[inline(always)]
fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

const H: usize = 3840;
const ROWS: usize = 1024;
/// Rows processed side by side: their sums of squares are independent chains (one row per lane).
const LANES: usize = 8;

/// One 1024-row chunk starting at token `first`, written into `out` = [hi, mid, lo]: each 1024 * 3840 BF16
/// values as little-endian bytes in element order (the packed image `HostTensor::from_vec` would build).
/// `embedding` is the BF16 little-endian [262144, 3840] tensor; `gamma` the input RMSNorm weight as f32.
pub fn chunk_into(embedding: &[u8], gamma: &[f32], first: usize, scale: f32, eps: f32,
    out: [&mut [u8]; 3]) -> Result<(), String> {
    let gamma: &[f32; H] = gamma.try_into().map_err(|_| "layer0 input norm geometry mismatch".to_string())?;
    if embedding.len() < (first + ROWS) * H * 2 { return Err("layer0 embedding chunk out of range".into()); }
    let [hi, mid, lo] = out;
    if [hi.len(), mid.len(), lo.len()] != [ROWS * H * 2; 3] { return Err("layer0 chunk buffer geometry".into()); }
    rows(embedding, gamma, first, scale, eps, hi, mid, lo)
}

fn row_bytes(s: &mut [u8], at: usize) -> Result<&mut [u8; 2 * H], String> {
    s.get_mut(at..at + 2 * H).and_then(|r| r.try_into().ok()).ok_or_else(|| "layer0 output row geometry".to_string())
}

#[allow(clippy::too_many_arguments)]
fn rows(embedding: &[u8], gamma: &[f32; H], first: usize, scale: f32, eps: f32,
    h: &mut [u8], m: &mut [u8], l: &mut [u8]) -> Result<(), String> {
    let mut x: Box<[[f32; H]; LANES]> = vec![[0f32; H]; LANES].into_boxed_slice().try_into()
        .map_err(|_| "layer0 lane buffer".to_string())?;
    for row in (0..ROWS).step_by(LANES) {
        for (lane, xr) in x.iter_mut().enumerate() {
            let id = first + row + lane;
            let src: &[u8; 2 * H] = embedding.get(id * H * 2..(id + 1) * H * 2)
                .and_then(|s| s.try_into().ok())
                .ok_or_else(|| "layer0 embedding chunk out of range".to_string())?;
            for j in 0..H {
                xr[j] = bf16_bits_to_f32(u16::from_le_bytes([src[2 * j], src[2 * j + 1]])) * scale;
            }
        }
        // Keep the FP32 sum in channel order while interleaving independent rows.
        let mut sum = [0f32; LANES];
        for j in 0..H {
            for lane in 0..LANES {
                let v = x[lane][j];
                sum[lane] += v * v;
            }
        }
        for lane in 0..LANES {
            let rms = (sum[lane] / 3840.0 + eps).sqrt();
            let at = (row + lane) * H * 2;
            let (hr, mr, lr) = (row_bytes(h, at)?, row_bytes(m, at)?, row_bytes(l, at)?);
            let xr = &x[lane];
            let mut bad = false;
            for j in 0..H {
                let v = xr[j] / rms * gamma[j];
                let hb = f32_to_bf16_bits(v);
                let rem = v - bf16_bits_to_f32(hb);
                let mb = f32_to_bf16_bits(rem);
                let rem = rem - bf16_bits_to_f32(mb);
                let lb = f32_to_bf16_bits(rem);
                bad |= (rem - bf16_bits_to_f32(lb) != 0.0) | !v.is_finite();
                [hr[2 * j], hr[2 * j + 1]] = hb.to_le_bytes();
                [mr[2 * j], mr[2 * j + 1]] = mb.to_le_bytes();
                [lr[2 * j], lr[2 * j + 1]] = lb.to_le_bytes();
            }
            if bad {
                let j = (0..H).find(|&j| {
                    let v = xr[j] / rms * gamma[j];
                    let rem = v - bf16_bits_to_f32(f32_to_bf16_bits(v));
                    let rem = rem - bf16_bits_to_f32(f32_to_bf16_bits(rem));
                    rem - bf16_bits_to_f32(f32_to_bf16_bits(rem)) != 0.0 || !v.is_finite()
                }).unwrap_or(0);
                return Err(format!("layer0 split3 loses input at token {} channel {j}", first + row + lane));
            }
        }
    }
    Ok(())
}
