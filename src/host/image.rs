//! Image preprocessing: decode, aspect-preserving resize to at most 280 patches of
//! 48 x 48 pixels, and the patch layout the vision kernels consume. The vectorized
//! Catmull-Rom sampler matches `image::imageops::resize`.
use furiosa_opt_std::prelude::*;

use crate::axes::Rv;

const PATCH_SIZE: u32 = 16;
const POOLING_KERNEL_SIZE: u32 = 3;
const MODEL_PATCH_SIZE: u32 = PATCH_SIZE * POOLING_KERNEL_SIZE;
pub(crate) const MAX_SOFT_TOKENS: usize = 280;

#[derive(Clone)]
pub struct ImagePatches {
    pub pixels: Vec<Vec<bf16>>,
    pub positions: Vec<(usize, usize)>,
    /// Exact RGB8 patch bytes for the resident vision encoders.
    pub(crate) rgb8: Vec<i8>,
}

const MAX_SOURCE_SIDE: u32 = 16384;
const MAX_DECODE_BYTES: u64 = 256 * 1024 * 1024;

pub fn load_from_bytes(bytes: &[u8]) -> Result<ImagePatches, Box<dyn std::error::Error>> {
    let _permit = super::preprocess::enter();
    let decoded = decode_rgb(bytes)?;
    let resized = resize_for_model(decoded)?;
    Ok(patchify_resized(&resized))
}

/// Decode JPEG or PNG to 8-bit RGB with bounded dimensions and allocation.
pub(crate) fn decode_rgb(bytes: &[u8]) -> Result<image::RgbImage, Box<dyn std::error::Error>> {
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_SOURCE_SIDE);
    limits.max_image_height = Some(MAX_SOURCE_SIDE);
    limits.max_alloc = Some(MAX_DECODE_BYTES);

    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format()?;
    reader.limits(limits);
    Ok(reader.decode()?.to_rgb8())
}

fn aspect_preserving_size(height: u32, width: u32) -> Result<(u32, u32), String> {
    let max_raw_patches = MAX_SOFT_TOKENS as f64 * (POOLING_KERNEL_SIZE as f64).powi(2);
    let target_px = max_raw_patches * f64::from(PATCH_SIZE).powi(2);
    let factor = (target_px / (f64::from(height) * f64::from(width))).sqrt();

    let block = f64::from(MODEL_PATCH_SIZE);
    let mut target_height = (f64::from(height) * factor / block).floor() * block;
    let mut target_width = (f64::from(width) * factor / block).floor() * block;

    if target_height == 0.0 && target_width == 0.0 {
        return Err(format!(
            "cannot resize {width}x{height} to a multiple of {MODEL_PATCH_SIZE}"
        ));
    }
    let max_side = MAX_SOFT_TOKENS as f64 * block;
    if target_height == 0.0 {
        target_height = block;
        target_width = ((f64::from(width) / f64::from(height)).floor() * block).min(max_side);
    } else if target_width == 0.0 {
        target_width = block;
        target_height = ((f64::from(height) / f64::from(width)).floor() * block).min(max_side);
    }
    if target_height * target_width > target_px {
        return Err(format!(
            "resizing {width}x{height} to {target_width}x{target_height} exceeds {MAX_SOFT_TOKENS} patches"
        ));
    }
    Ok((target_height as u32, target_width as u32))
}

/// Stage 2: the aspect-preserving Catmull-Rom resize to a multiple of 48 pixels per side.
pub(crate) fn resize_for_model(image: image::RgbImage) -> Result<image::RgbImage, Box<dyn std::error::Error>> {
    let (width, height) = image.dimensions();
    let (target_height, target_width) = aspect_preserving_size(height, width)?;
    Ok(if (target_width, target_height) == (width, height) {
        image
    } else {
        catmull_rom::resize(&image, target_width, target_height)
    })
}

/// `image::imageops::resize(.., FilterType::CatmullRom)` for RGB8, bit for bit, laid out for
/// speed. It follows image 0.25.10's separable sampler (`imageops/sample.rs`): a vertical pass
/// into f32 rows, then a horizontal pass that clamps to [0, 255] and rounds half away from zero.
/// Every output value is the same f32 expression evaluated in the same order (starting from 0,
/// taps ascending, multiply then add, no fusion); only the loop order around it changes, so the
/// compiler can vectorize: the vertical pass accumulates whole source rows tap by tap, and the
/// horizontal pass transposes eight f32 rows so eight output rows advance together. One thread:
/// the largest model input (280 patches, 0.65 MP) takes a few ms. The tests compare it
/// with the crate on the public image and on up- and down-scaled synthetic sizes. Only the
/// three colour channels are computed (the crate's fourth, constant channel is dropped by the
/// RGB output).
mod catmull_rom {
    /// Mitchell-Netravali cubic with B = 0, C = 0.5 (the crate's `catmullrom_kernel`).
    fn kernel(x: f32) -> f32 {
        let (b, c) = (0.0f32, 0.5f32);
        let a = x.abs();
        let k = if a < 1.0 {
            (12.0 - 9.0 * b - 6.0 * c) * a.powi(3) + (-18.0 + 12.0 * b + 6.0 * c) * a.powi(2) + (6.0 - 2.0 * b)
        } else if a < 2.0 {
            (-b - 6.0 * c) * a.powi(3) + (6.0 * b + 30.0 * c) * a.powi(2) + (-12.0 * b - 48.0 * c) * a + (8.0 * b + 24.0 * c)
        } else {
            0.0
        };
        k / 6.0
    }

    fn clamp<N: PartialOrd>(a: N, min: N, max: N) -> N {
        if a < min { min } else if a > max { max } else { a }
    }

    /// The first source index and the normalized weights of every output position.
    fn taps(source: u32, target: u32) -> Vec<(usize, Vec<f32>)> {
        const SUPPORT: f32 = 2.0;
        let ratio = source as f32 / target as f32;
        let sratio = if ratio < 1.0 { 1.0 } else { ratio };
        let src_support = SUPPORT * sratio;
        (0..target)
            .map(|out| {
                let input = (out as f32 + 0.5) * ratio;
                let left = (input - src_support).floor() as i64;
                let left = clamp(left, 0, i64::from(source) - 1) as u32;
                let right = (input + src_support).ceil() as i64;
                let right = clamp(right, i64::from(left) + 1, i64::from(source)) as u32;
                let input = input - 0.5;
                let mut ws = Vec::with_capacity((right - left) as usize);
                let mut sum = 0.0;
                for i in left..right {
                    let w = kernel((i as f32 - input) / sratio);
                    ws.push(w);
                    sum += w;
                }
                for w in ws.iter_mut() {
                    *w /= sum;
                }
                (left as usize, ws)
            })
            .collect()
    }

    /// The crate's `FloatNearest(clamp(v, 0, 255))` (round half away from zero; NaN gives 0)
    /// without the `f32::round` call, which is not inlined on x86-64: on
    /// [0, 255] the fraction `v - trunc(v)` is exact, so `trunc + (fraction >= 0.5)` is the same
    /// integer. The comparisons send NaN to 0 first, so the unchecked conversion only sees
    /// values in [0, 255]; loops over it vectorize.
    #[inline(always)]
    pub(super) fn quantize(v: f32) -> u8 {
        let v = if v >= 0.0 { if v <= 255.0 { v } else { 255.0 } } else { 0.0 };
        // SAFETY: `v` is finite and in [0, 255].
        let whole: i32 = unsafe { v.to_int_unchecked() };
        (whole + i32::from(v - whole as f32 >= 0.5)) as u8
    }

    /// Output rows the horizontal pass advances together.
    const LANES: usize = 8;

    pub(super) fn resize(image: &image::RgbImage, nwidth: u32, nheight: u32) -> image::RgbImage {
        let (width, height) = image.dimensions();
        if width == 0 || height == 0 || (nwidth, nheight) == (width, height) {
            return image::imageops::resize(image, nwidth, nheight, image::imageops::FilterType::CatmullRom);
        }
        let (w3, nw, nh) = (width as usize * 3, nwidth as usize, nheight as usize);
        let raw = image.as_raw();
        // Vertical pass: `nheight` rows of `width` f32 RGB. Each output value starts at 0 and
        // adds `f32(p) * w` over the taps in order, the crate's per-pixel loop.
        let vertical = taps(height, nheight);
        let mut tmp = vec![0f32; nh * w3];
        for (row, (left, ws)) in tmp.chunks_exact_mut(w3).zip(&vertical) {
            for (i, &wt) in ws.iter().enumerate() {
                let src = &raw[(left + i) * w3..(left + i + 1) * w3];
                for (t, &p) in row.iter_mut().zip(src) {
                    *t += f32::from(p) * wt;
                }
            }
        }
        // Horizontal pass: `nwidth x nheight` RGB8, eight rows at a time from a transposed copy
        // of their f32 values (`column[k][lane]` = row `lane`'s value `k`).
        let horizontal = taps(width, nwidth);
        let nw3 = nw * 3;
        let mut out = vec![0u8; nh * nw3];
        let mut column = vec![[0f32; LANES]; w3];
        for (b, rows) in out.chunks_mut(LANES * nw3).enumerate() {
            let r0 = b * LANES;
            let live = rows.len() / nw3;
            for lane in 0..live {
                for (k, &v) in tmp[(r0 + lane) * w3..(r0 + lane + 1) * w3].iter().enumerate() {
                    column[k][lane] = v;
                }
            }
            for (x, (left, ws)) in horizontal.iter().enumerate() {
                let (mut r, mut g, mut b) = ([0f32; LANES], [0f32; LANES], [0f32; LANES]);
                for (i, &wt) in ws.iter().enumerate() {
                    let base = (left + i) * 3;
                    let (cr, cg, cb) = (&column[base], &column[base + 1], &column[base + 2]);
                    for lane in 0..LANES {
                        r[lane] += cr[lane] * wt;
                        g[lane] += cg[lane] * wt;
                        b[lane] += cb[lane] * wt;
                    }
                }
                for lane in 0..live {
                    let o = lane * nw3 + x * 3;
                    rows[o] = quantize(r[lane]);
                    rows[o + 1] = quantize(g[lane]);
                    rows[o + 2] = quantize(b[lane]);
                }
            }
        }
        image::RgbImage::from_raw(nwidth, nheight, out).expect("resize output size")
    }
}

/// Stage 3: row-major 48 x 48 x RGB patches scaled to [0, 1] in bf16, with (col, row) positions.
pub(crate) fn patchify_resized(resized: &image::RgbImage) -> ImagePatches {
    let (target_width, target_height) = resized.dimensions();
    let patch_cols = (target_width / MODEL_PATCH_SIZE) as usize;
    let patch_rows = (target_height / MODEL_PATCH_SIZE) as usize;
    let block_px = MODEL_PATCH_SIZE as usize;
    let stride = target_width as usize * 3;
    let raw = resized.as_raw();
    // `bf16(f32(c) / 255)` for every byte value: the same rounding, looked up.
    let unit: [bf16; 256] = std::array::from_fn(|c| bf16::from_f32(f32::from(c as u8) / 255.0));

    let mut rgb8 = Vec::with_capacity(patch_rows * patch_cols * Rv::SIZE);
    let mut pixels = Vec::with_capacity(patch_rows * patch_cols);
    let mut positions = Vec::with_capacity(patch_rows * patch_cols);
    for row in 0..patch_rows {
        for col in 0..patch_cols {
            let mut patch = Vec::with_capacity(Rv::SIZE);
            for local_row in 0..block_px {
                let y = row * block_px + local_row;
                let row_start = y * stride + col * block_px * 3;
                let row_bytes = &raw[row_start..row_start + block_px * 3];
                rgb8.extend(row_bytes.iter().map(|&c| c as i8));
                patch.extend(row_bytes.iter().map(|&channel| unit[usize::from(channel)]));
            }
            assert_eq!(patch.len(), Rv::SIZE);
            pixels.push(patch);
            positions.push((col, row));
        }
    }

    ImagePatches { pixels, positions, rgb8 }
}
