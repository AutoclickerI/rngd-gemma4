//! Rotate and pack each row directly in the serving layout.
use super::*;

pub(super) fn direct(tensors: &safetensors::SafeTensors<'_>, layer: usize, h60: &[f32], lut: &[f32; 256]) -> Result<Host, String> {
    const WIDTH: usize = H::SIZE;
    const ROWS: usize = Lu::SIZE;
    const PACKED: usize = Hb::SIZE;
    let sites = [
        q4cal::Projection::load(tensors, layer, "up_proj")?,
        q4cal::Projection::load(tensors, layer, "gate_proj")?,
    ];
    let smooth = q4cal::smooth(layer)?;
    let signs = signs(layer, Site::Mlp, WIDTH);
    let mut codes = vec![0i8; ROWS * PACKED];
    let mut scales = vec![0f32; 2 * ROWS];
    let workers = std::thread::available_parallelism().map_or(1, |n| n.get()).min(16);
    let rows_per_worker = ROWS.div_ceil(workers);
    let (weights, corrections) = scales.split_at_mut(ROWS);
    std::thread::scope(|scope| -> Result<(), String> {
        let mut jobs = Vec::new();
        for (chunk, ((targets, weights), corrections)) in codes.chunks_mut(rows_per_worker * PACKED)
            .zip(weights.chunks_mut(rows_per_worker)).zip(corrections.chunks_mut(rows_per_worker)).enumerate()
        {
            let (sites, smooth, signs) = (&sites, &smooth, &signs);
            jobs.push(scope.spawn(move || -> Result<(), String> {
                let (mut x, mut y) = ([0f32; WIDTH], [0f32; WIDTH]);
                for (local, target) in targets.chunks_exact_mut(PACKED).enumerate() {
                    let lu = chunk * rows_per_worker + local;
                    let site = &sites[lu / 8 % 2];
                    // Invert the Lu row order to select the checkpoint row.
                    let row = 512 * (lu / 1024) + 256 * (lu / 512 % 2) + 128 * (lu / 32 % 2)
                        + 16 * (lu / 64 % 8) + 8 * (lu / 16 % 2) + lu % 8;
                    site.expand(row, lut, smooth, signs, &mut x);
                    q4cal::rotate(&x, h60, &mut y);
                    let amax = y.iter().fold(1e-30f32, |a, v| a.max(v.abs()));
                    let step = (amax / 15.) * site.ratios[row];
                    let weight = step * site.global;
                    if !weight.is_finite() {
                        return Err(format!("nonfinite Q4 scale at layer {layer}, row {lu}"));
                    }
                    weights[local] = weight;
                    let mut sum = 0;
                    // RotK order [Ob/32, Ra, Ob%32]. The first eight
                    // codes in each 16-code block occupy the low nibbles.
                    for (block, dst) in target.chunks_exact_mut(8).enumerate() {
                        let src = block / 120 * 32 + block / 2 % 60 * 64 + block % 2 * 16;
                        for (j, byte) in dst.iter_mut().enumerate() {
                            let lo = q4cal::code(y[src + j], step);
                            let hi = q4cal::code(y[src + 8 + j], step);
                            sum += i32::from(lo) + i32::from(hi);
                            *byte = (((lo - 1) / 2) & 15) | (((hi - 1) / 2) << 4);
                        }
                    }
                    corrections[local] = (sum - 16 * WIDTH as i32) as f32;
                }
                Ok(())
            }));
        }
        for job in jobs {
            job.join().map_err(|_| "Q4 packing worker panicked".to_string())??;
        }
        Ok(())
    })?;
    Ok(Host { codes, scales })
}
