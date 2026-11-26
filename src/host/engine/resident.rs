//! Resident Q4 segments fuse the next layer's Sliding attention.
//! Tok128 covers decode-only steps. Tok1024 and Tok1536 fuse six decode
//! pairs and the prompt rounds at `PROMPT_ROW`.
//! Earlier decode groups in a larger step run through separate launches.

use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;
use crate::host::engine::attention::{AttentionState, StepPlan};
use crate::host::engine::attention_geometry::Family;
use crate::host::engine::buckets::Dense;
use crate::host::engine::dsl::{self, DECODE_ROWS, Kernel};
use crate::host::engine::int8::{MidFsInt8, MidSsInt8, SlidingKv};
use crate::ops_core::resident::PROMPT_ROW;
use crate::ops_dsl_attention::prompt_blocks as pb;

use crate::host::engine::attention_geometry::SINK_ROW;
use crate::ops_core::resident::self as rs;

type BoxError = Box<dyn std::error::Error>;

pub const WINDOW: usize = crate::ops_dsl_attention::window::GROUP_WINDOW;
const BAND: u8 = pb::TILES as u8;
const MIDR: u8 = pb::MID_TILES as u8;
const P941: [u8; 3] = [BAND, MIDR, 1];
pub(super) const P999: [u8; 3] = [BAND, BAND, BAND];
const STEP_GROUPS: usize = crate::host::engine::tables::OWNERS.div_ceil(DECODE_ROWS);

struct Program {
    groups: usize,
    tiles: Option<[u8; 3]>,
    rss: &'static str,
    rfs: &'static str,
}

macro_rules! programs {
    ($($g:literal, $tiles:expr, $rss:literal, $rfs:literal;)+) => {
        &[$(Program {
            groups: $g,
            tiles: $tiles,
            rss: concat!("furiosa_opt_gemma4::ops_core::resident::", $rss),
            rfs: concat!("furiosa_opt_gemma4::ops_core::resident_fs::", $rfs),
        }),+]
    };
}

const PROGRAMS: &[Program] = programs! {
    1, None, "rss_q4_class512_t128_g1", "rfs_q4_class512_t128_g1";
    2, None, "rss_q4_class512_t128_g2", "rfs_q4_class512_t128_g2";
    4, None, "rss_q4_class512_t128_g4", "rfs_q4_class512_t128_g4";
    8, None, "rss_q4_class512_t128_g8", "rfs_q4_class512_t128_g8";
    16, None, "rss_q4_class512_t128_g16", "rfs_q4_class512_t128_g16";
    12, Some(P941), "q4::rss_int8_fp8kv_t1024_g12_p941_pair", "q4::rfs_int8_fp8kv_t1024_g12_p941_pair";
    12, Some(P999), "q4::rss_int8_fp8kv_t1536_g12_p999_pair", "q4::rfs_int8_fp8kv_t1536_g12_p999_pair";
};

pub(super) const PAIRED_GROUPS: usize = 12;

pub(super) fn paired_prompt(bucket: usize, tiles: [u8; 3]) -> bool {
    matches!((bucket, tiles), (1024, P941) | (1536, P999))
}

struct Empty {
    off: rs::wtop::Off,
    cap: rs::wtop::Cap,
    rows: rs::wtop::GroupRows,
}

pub(crate) struct Resident {
    packed_rows: Option<HbmTensor<i32, Chip, m![Rq]>>,
    empty: Option<Empty>,
    functions: std::cell::RefCell<
        std::collections::HashMap<
            &'static str,
            std::sync::Arc<furiosa_opt_std::backend::npu::Function>,
        >,
    >,
}

pub(crate) struct Step {
    program: &'static Program,
    /// Bytes for the decode groups of a Tok128 program are drawn from these
    /// pairs of word offsets. Paired programs bind the plan's paired tables.
    tables: Vec<[usize; 2]>,
    prompt: bool,
}

impl Step {
    /// Skip the fused prompt launch at the beginning and fused decode groups
    /// at the end of the attention plan.
    pub fn skips(&self) -> (usize, usize) {
        (usize::from(self.prompt), self.tables.len())
    }

    pub fn paths(&self) -> (&'static str, Option<&'static str>) {
        (self.program.rss, Some(self.program.rfs))
    }
}

impl Resident {
    pub async fn new(_device: &mut Device) -> Result<Self, BoxError> {
        let (packed_rows, empty) = {
            let row_bytes = Hqs::SIZE * 2;
            let mut packed = vec![0i32; Rq::SIZE];
            for (j, x) in packed
                .iter_mut()
                .take(STEP_GROUPS * 2 * DECODE_ROWS)
                .enumerate()
            {
                *x = (j * row_bytes) as i32;
            }
            let packed = HostTensor::<i32, m![Rq]>::from_vec(packed)
                .to_hbm(&mut _device.pdma)
                .await?;
            let n = DECODE_ROWS * WINDOW;
            // Spread masked gathers over the pool to avoid contention on one address.
            let kv_row_bytes = Ns::SIZE * Ds::SIZE * std::mem::size_of::<SlidingKv>();
            let spread = (1..=n)
                .map(|i| ((i * 7919) % Rs::SIZE * kv_row_bytes) as i32)
                .collect();
            let (off, cap) = rs::wtop::upload_tables(_device, spread, vec![-1e30; n]).await?;
            let rows = rs::wtop::upload_rows(
                _device,
                vec![(SINK_ROW * row_bytes) as i32; 2 * DECODE_ROWS],
            )
            .await?;
            (Some(packed), Some(Empty { off, cap, rows }))
        };
        Ok(Self {
            packed_rows,
            empty,
            functions: Default::default(),
        })
    }

    /// Select resident geometry for the step before attention planning.
    #[allow(clippy::type_complexity)]
    pub fn peek(
        &self,
        bucket: usize,
        prompt_rows: usize,
        decode_rows: usize,
    ) -> Option<(usize, Option<&'static [[u8; 3]]>)> {
        if decode_rows == 0 {
            return None;
        }
        let groups = decode_rows.div_ceil(DECODE_ROWS);
        if groups > STEP_GROUPS {
            return None;
        }
        let images: Option<&'static [[u8; 3]]> = match (bucket, prompt_rows) {
            (128, 0) if decode_rows <= 128 => None,
            (1024, 1..) => Some(&[P941]),
            (1536, 1..) => Some(&[P999]),
            _ => return None,
        };
        Some((groups, images))
    }

    pub fn step(&self, plan: &StepPlan, bucket: usize, groups: usize) -> Option<Step> {
        fit(plan, bucket, groups)
    }

    fn bind_attention(
        &self,
        step: &Step,
        plan: &dsl::Plan,
        official: &AttentionState,
    ) -> Result<furiosa_opt_std::runtime::Buffers, BoxError> {
        use furiosa_opt_std::runtime::{Buffers, DeviceSend};
        let (words, base) = official.words(Family::Sliding).tensor()?;
        let mut bound = Buffers::new();
        words.bind(&mut bound)?;
        let table = match bound.len() {
            1 => bound.pop().unwrap(),
            n => return Err(format!("the Sliding words bind {n} buffers").into()),
        };
        let mut inputs = Buffers::new();
        if step.prompt {
            bind_pair_tables(plan, &table, base, &mut inputs)?;
            return Ok(inputs);
        }
        let empty = self.empty.as_ref().ok_or("resident: empty group missing")?;
        let mut packed = Buffers::new();
        self.packed_rows
            .as_ref()
            .ok_or("resident: packed rows missing")?
            .bind(&mut packed)?;
        let packed = packed.pop().ok_or("resident: packed rows have no buffer")?;
        let span = DECODE_ROWS * WINDOW * 4;
        for g in 0..step.program.groups {
            if let Some(offsets) = step.tables.get(g) {
                for at in offsets {
                    let lo = base + at * 4;
                    let hi = lo + span;
                    if hi > table.size() {
                        return Err(format!(
                            "resident group {g}: words {lo}..{hi} exceed the upload's {} B",
                            table.size()
                        )
                        .into());
                    }
                    inputs
                        .try_push(table.slice(lo..hi))
                        .map_err(|_| "resident launch: too many arguments")?;
                }
                let at = g * 2 * DECODE_ROWS * 4;
                inputs
                    .try_push(packed.slice(at..at + 2 * DECODE_ROWS * 4))
                    .map_err(|_| "resident: row frame overflow")?;
            } else {
                (&empty.off, &empty.cap, &empty.rows).bind(&mut inputs)?;
            }
        }
        Ok(inputs)
    }

    /// Run the Q4 Sliding segment and the next layer's fused decode and prompt launches.
    #[allow(clippy::too_many_arguments)]
    pub async fn mid_ss<T: AxisName>(
        &self,
        device: &mut Device,
        step: &Step,
        plan: &StepPlan,
        dense: &mut Dense<T>,
        layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
        args: MidSsInt8<'_>,
        official: &mut AttentionState,
        k_l: &mut HbmTensor<SlidingKv, Chip, m![Rs, Ns, Ds]>,
        v_l: &mut HbmTensor<SlidingKv, Chip, m![Rs, Ns, Ds]>,
        k_n: &mut HbmTensor<SlidingKv, Chip, m![Rs, Ns, Ds]>,
        v_n: &mut HbmTensor<SlidingKv, Chip, m![Rs, Ns, Ds]>,
    ) -> Result<(), BoxError> {
        {
            use furiosa_opt_std::runtime::{Buffers, DeviceSend};
            let path = step.program.rss;
            let function = self.program_function(&*device, path).await?;
            let mut inputs = self.bind_attention(step, plan, official)?;
            let MidSsInt8 { consts, o, next, q4 } = args;
            let space = &official.sliding;
            (
                &space.out,
                &dense.rows_s,
                &dense.stage,
                &dense.publish_s,
                &dense.rope_s,
                &dense.x,
                consts,
                o,
            )
                .bind(&mut inputs)?;
            q4.bind(&mut inputs)?;
            (layer_scalar, next, &*k_l, &*v_l, &space.qin, &*k_n, &*v_n).bind(&mut inputs)?;
            if step.prompt {
                dsl::bind_prompt_tables(
                    &official.dsl,
                    &plan.launches[0],
                    official.words(Family::Sliding),
                    &mut inputs,
                )?;
                space.out.bind(&mut inputs)?;
            }
            let result = super::queue::run(function, inputs, Buffers::new()).await;
            result?;
            Ok(())
        }
    }

    /// Run the Q4 Full segment and the next layer's fused Sliding attention.
    #[allow(clippy::too_many_arguments)]
    pub async fn mid_fs<T: AxisName>(
        &self,
        device: &mut Device,
        step: &Step,
        plan: &StepPlan,
        dense: &mut Dense<T>,
        layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
        args: MidFsInt8<'_>,
        official: &mut AttentionState,
        k_n: &mut HbmTensor<SlidingKv, Chip, m![Rs, Ns, Ds]>,
        v_n: &mut HbmTensor<SlidingKv, Chip, m![Rs, Ns, Ds]>,
    ) -> Result<(), BoxError> {
        {
            use furiosa_opt_std::runtime::{Buffers, DeviceSend};
            let path = step.program.rfs;
            let function = self.program_function(&*device, path).await?;
            let mut inputs = self.bind_attention(step, plan, official)?;
            let MidFsInt8 {
                consts,
                o,
                next,
                q4,
                ..
            } = args;
            (
                &official.sliding.out,
                &official.full.out,
                &dense.rows_f,
                &dense.rope_s,
                &dense.rows_s,
                &dense.stage,
                &dense.x,
                consts,
                o,
            )
                .bind(&mut inputs)?;
            q4.bind(&mut inputs)?;
            (layer_scalar, next, &official.sliding.qin, &*k_n, &*v_n).bind(&mut inputs)?;
            if step.prompt {
                dsl::bind_prompt_tables(
                    &official.dsl,
                    &plan.launches[0],
                    official.words(Family::Sliding),
                    &mut inputs,
                )?;
                official.sliding.out.bind(&mut inputs)?;
            }
            let result = super::queue::run(function, inputs, Buffers::new()).await;
            result?;
            Ok(())
        }
    }
}

fn fit(plan: &dsl::Plan, bucket: usize, groups: usize) -> Option<Step> {
    if groups == 0 || groups > STEP_GROUPS {
        return None;
    }
    let prefill = plan.launches.len().checked_sub(groups)?;
    let (program, first, prompt) = match (bucket, prefill) {
        (128, 0) => (
            PROGRAMS
                .iter()
                .find(|p| p.tiles.is_none() && p.groups >= groups)?,
            0,
            false,
        ),
        (1024, 1) | (1536, 1..) if WINDOW == 384 => {
            if plan.launches[1..prefill].iter().any(|launch| matches!(launch.kernel, Kernel::Prompt { .. })) {
                return None;
            }
            let launch = &plan.launches[0];
            let Kernel::Prompt { tiles } = launch.kernel else {
                return None;
            };
            if launch.row != PROMPT_ROW {
                return None;
            }
            if !paired_prompt(bucket, tiles) || plan.paired_decode.len() != PAIRED_GROUPS / 2 {
                return None;
            }
            let first = groups.saturating_sub(PAIRED_GROUPS);
            let program = PROGRAMS.iter().find(|p| p.tiles == Some(tiles))?;
            (program, first, true)
        }
        _ => return None,
    };
    let mut tables = Vec::with_capacity(groups - first);
    for (g, launch) in plan.launches[prefill..].iter().enumerate() {
        if launch.kernel
            != (Kernel::Decode {
                family: Family::Sliding,
                w: WINDOW,
            })
            || launch.row != 2 * DECODE_ROWS * g
            || launch.tables.len() != 2
        {
            return None;
        }
        if g >= first {
            tables.push([launch.tables[0], launch.tables[1]]);
        }
    }
    Some(Step {
        program,
        tables,
        prompt,
    })
}

fn bind_pair_tables(
    plan: &dsl::Plan,
    table: &<furiosa_opt_std::runtime::Buffers as IntoIterator>::Item,
    base: usize,
    inputs: &mut furiosa_opt_std::runtime::Buffers,
) -> Result<(), BoxError> {
    if plan.paired_decode.len() != PAIRED_GROUPS / 2 {
        return Err("invalid paired decode table count".into());
    }
    let rows = 2 * DECODE_ROWS;
    for offsets in &plan.paired_decode {
        for (&at, size) in offsets
            .iter()
            .zip([rows * WINDOW * 4, rows * WINDOW * 4, rows * 2 * 4])
        {
            let lo = base + at * 4;
            let hi = lo + size;
            if hi > table.size() {
                return Err("pair tables exceed uploaded words".into());
            }
            inputs
                .try_push(table.slice(lo..hi))
                .map_err(|_| "paired launch argument overflow")?;
        }
    }
    Ok(())
}


impl Resident {
    async fn program_function(
        &self,
        device: &Device,
        path: &'static str,
    ) -> Result<std::sync::Arc<furiosa_opt_std::backend::npu::Function>, BoxError> {
        if let Some(function) = self.functions.borrow().get(path) {
            return Ok(function.clone());
        }
        let row = crate::host::prebuilt::launch_path(path);
        let function =
            std::sync::Arc::new(furiosa_opt_std::backend::npu::function(device, row, &[]).await?);
        super::step_image::register(path, &function);
        self.functions.borrow_mut().insert(path, function.clone());
        Ok(function)
    }

}
