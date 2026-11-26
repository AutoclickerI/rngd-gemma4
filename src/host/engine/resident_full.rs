//! A Sliding → Full segment runs the following layer's covered decode blocks.
//! Each block takes the smallest free slot that covers its window. Empty slots
//! read masked tables and write only the sink row.

use std::{cell::RefCell, sync::Arc};

use furiosa_opt_std::backend::npu::{Function, function};
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;
use crate::host::engine::attention::{AttentionState, StepPlan};
use crate::host::engine::buckets::Dense;
use crate::host::engine::dsl::{DECODE_ROWS, Kernel};
use crate::host::engine::int8::{MidSfInt8, SlidingKv};
use crate::host::engine::attention_geometry::Family;
use crate::host::engine::attention_geometry::SINK_ROW;
use crate::ops_core::full_resident::w512;
use crate::ops_core::full_resident::{w1024, w1536, w2048, w2560, w3072};

type BoxError = Box<dyn std::error::Error>;

struct Program {
    path: &'static str,
    groups: [usize; 8],
}

const PROGRAMS: [Program; 2] = [
    Program {
        path: "furiosa_opt_gemma4::ops_core::full_resident::rsf_int8_fp8kv_t1024_s8",
        groups: [512, 512, 1024, 1024, 1024, 1536, 2560, 2560],
    },
    Program {
        path: "furiosa_opt_gemma4::ops_core::full_resident::rsf_int8_fp8kv_t1536_s8",
        groups: [512, 512, 1024, 1024, 1024, 2048, 3072, 3072],
    },
];

pub(super) fn program(bucket: usize) -> Option<usize> {
    match bucket {
        1024 => Some(0),
        1536 => Some(1),
        _ => None,
    }
}

/// For each decode block (longest first), return its slot's window if it fits.
pub(super) fn place(program: usize, windows: impl Iterator<Item = usize>) -> Vec<Option<usize>> {
    let groups = &PROGRAMS[program].groups;
    let mut used = [false; 8];
    windows.map(|w| {
        let g = groups.iter().enumerate().position(|(g, &size)| !used[g] && size >= w)?;
        used[g] = true;
        Some(groups[g])
    }).collect()
}

/// `v` uploaded as a tensor of the mapping its use names.
async fn hbm<D: MaterializableScalar, E: M>(device: &mut Device, v: Vec<D>) -> Result<HbmTensor<D, Chip, E>, Error> {
    HostTensor::<D, E>::from_vec(v).to_hbm(&mut device.pdma).await
}

/// A window's constant all-masked tables (an empty group).
struct Empty {
    off: Vec<i32>,
    cap: Vec<f32>,
}

impl Empty {
    fn new(w: usize) -> Self {
        let n = DECODE_ROWS * w;
        // Spread pool reads across addresses to avoid gather stalls.
        let row = Df::SIZE;
        Self { off: (1..=n).map(|i| ((i * 7919) % Rf::SIZE * row) as i32).collect(), cap: vec![-1e30; n] }
    }
}

/// The device copies of every window's empty tables and the sink rows (raw route).
struct EmptyHbm {
    w512: (w512::Off, w512::Cap),
    w1024: (w1024::Off, w1024::Cap),
    w1536: (w1536::Off, w1536::Cap),
    w2560: (w2560::Off, w2560::Cap),
    w2048: (w2048::Off, w2048::Cap),
    w3072: (w3072::Off, w3072::Cap),
    rows: w512::Rows,
}

/// The engine's Full-side resident state.
pub(crate) struct ResidentFull {
    /// Per decode block `b`, its 16 row-space rows (`16 b..`), byte offsets (raw route; grown on demand).
    rows: Vec<w512::Rows>,
    empty: EmptyHbm,
    functions: [RefCell<Option<Arc<Function>>>; PROGRAMS.len()],
}

/// A step the program serves.
pub(crate) struct Step {
    program: usize,
    /// Per program group (argument order): the served launch's `[kv_off, cap]` word offsets in the Full upload and its
    /// block, or `None` (an empty group).
    slots: [Option<[usize; 3]>; 8],
    /// The plan's decode launches the program runs (its last ones).
    served: usize,
}

impl Step {
    /// The definition path of the step's program (`step_image`'s key).
    pub fn path(&self) -> &'static str {
        PROGRAMS[self.program].path
    }

    /// The plan's launches the program runs (the last ones).
    pub fn served(&self) -> usize {
        self.served
    }
}

impl ResidentFull {
    pub async fn new(device: &mut Device) -> Result<Self, BoxError> {
        // The raw 1,536-row lookup needs a compiled image in the reference build.
        #[cfg(feature = "compile-carried")]
        if std::hint::black_box(false) {
            let _ = launch(
                crate::ops_core::full_resident::rsf_int8_fp8kv_t1536_s8,
                super::dsl::embedding_args(),
            ).await;
        }
        macro_rules! up {
            ($n:literal) => {{
                let e = Empty::new($n);
                (hbm(device, e.off).await?, hbm(device, e.cap).await?)
            }};
        }
        let empty = EmptyHbm {
            w512: up!(512),
            w1024: up!(1024),
            w1536: up!(1536),
            w2560: up!(2560),
            w2048: up!(2048),
            w3072: up!(3072),
            rows: hbm(device, vec![(SINK_ROW * Hqf::SIZE * 2) as i32; 2 * DECODE_ROWS]).await?,
        };
        Ok(Self {
            rows: Vec::new(),
            empty,
            functions: Default::default(),
        })
    }

    pub async fn step(&mut self, device: &mut Device, plan: &StepPlan) -> Result<Option<Step>, BoxError> {
        let fit = self.fit(plan);
        let Some(step) = fit else {
            return Ok(None);
        };
        let need = step.slots.iter().flatten().map(|s| s[2] + 1).max().unwrap_or(0);
        while self.rows.len() < need {
            let b = self.rows.len();
            let r: Vec<i32> = (0..2 * DECODE_ROWS).map(|j| ((2 * DECODE_ROWS * b + j) * Hqf::SIZE * 2) as i32).collect();
            self.rows.push(hbm(device, r).await?);
        }
        Ok(Some(step))
    }

    /// The program's step from the plan: its last `served` launches are Full decode launches at the program's group
    /// windows, one group each.
    fn fit(&self, plan: &crate::host::engine::dsl::Plan) -> Option<Step> {
        let (program, served) = plan.full_groups?;
        if served == 0 || served > plan.launches.len() {
            return None;
        }
        let groups = &PROGRAMS[program].groups;
        let mut slots = [None; 8];
        for launch in &plan.launches[plan.launches.len() - served..] {
            let Kernel::Decode { family: Family::Full, w } = launch.kernel else { return None };
            if launch.tables.len() != 2 || launch.row % (2 * DECODE_ROWS) != 0 {
                return None;
            }
            let g = (0..groups.len()).find(|&g| slots[g].is_none() && groups[g] == w)?;
            slots[g] = Some([launch.tables[0], launch.tables[1], launch.row / (2 * DECODE_ROWS)]);
        }
        Some(Step { program, slots, served })
    }

    /// Layer `l`'s INT8-Full `mid_sf` and Full layer `l + 1`'s decode launches that the program runs, in one launch, in
    /// place of `mid_sf_int8_fp8kv_tN` and those launches.
    #[allow(clippy::too_many_arguments)]
    pub async fn mid_sf<T: AxisName>(
        &self,
        device: &mut Device,
        step: &Step,
        dense: &mut Dense<T>,
        layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
        args: MidSfInt8<'_>,
        official: &mut AttentionState,
        k_l: &mut HbmTensor<SlidingKv, Chip, m![Rs, Ns, Ds]>,
        v_l: &mut HbmTensor<SlidingKv, Chip, m![Rs, Ns, Ds]>,
        k_n: &mut HbmTensor<f8e4m3, Chip, m![Rf, Df]>,
        v_n: &mut HbmTensor<f8e4m3, Chip, m![Rf, Df]>,
    ) -> Result<(), BoxError> {
        let p = &PROGRAMS[step.program];
        {
            use furiosa_opt_std::runtime::{Buffers, DeviceSend};
            let empty = &self.empty;
            let function = self.program_function(device, step.program).await?;
            let (words, base) = official.words(Family::Full).tensor()?;
            let mut bound = Buffers::new();
            words.bind(&mut bound)?;
            let table = match bound.len() {
                1 => bound.pop().unwrap(),
                n => return Err(format!("the Full words bind {n} buffers").into()),
            };
            let MidSfInt8 { consts, o_f8: o, ffn_f8: ffn, next_f8: next } = args;
            let mut inputs = Buffers::new();
            (&official.sliding.out, &dense.rows_s, &dense.stage, &dense.publish_s, &dense.rope_f).bind(&mut inputs)?;
            (&dense.rows_f, &dense.publish_f, &dense.x).bind(&mut inputs)?;
            (consts, o).bind(&mut inputs)?;
            ffn.bind(&mut inputs)?;
            (layer_scalar, next, &*k_l, &*v_l, &official.full.qin, &*k_n, &*v_n, &official.full.out).bind(&mut inputs)?;
            for (g, &w) in p.groups.iter().enumerate() {
                match step.slots[g] {
                    Some([off_at, cap_at, block]) => {
                        let span = DECODE_ROWS * w * 4;
                        for at in [off_at, cap_at] {
                            let (lo, hi) = (base + at * 4, base + at * 4 + span);
                            if hi > table.size() {
                                return Err(format!("resident_f group {g}: words {lo}..{hi} exceed the upload's {} B", table.size()).into());
                            }
                            inputs.try_push(table.slice(lo..hi)).map_err(|_| "resident_f launch: too many arguments")?;
                        }
                        self.rows[block].bind(&mut inputs)?;
                    }
                    None => {
                        match w {
                            512 => (&empty.w512.0, &empty.w512.1).bind(&mut inputs)?,
                            1024 => (&empty.w1024.0, &empty.w1024.1).bind(&mut inputs)?,
                            1536 => (&empty.w1536.0, &empty.w1536.1).bind(&mut inputs)?,
                            2560 => (&empty.w2560.0, &empty.w2560.1).bind(&mut inputs)?,
                            2048 => (&empty.w2048.0, &empty.w2048.1).bind(&mut inputs)?,
                            3072 => (&empty.w3072.0, &empty.w3072.1).bind(&mut inputs)?,
                            _ => return Err(format!("resident_f: no group window {w}").into()),
                        }
                        empty.rows.bind(&mut inputs)?;
                    }
                }
            }
            let result = super::queue::run(function, inputs, Buffers::new()).await;
            result?;
            Ok(())
        }
    }

    async fn program_function(&self, device: &Device, program: usize) -> Result<Arc<Function>, BoxError> {
        let cell = &self.functions[program];
        if let Some(f) = cell.borrow().as_ref() {
            return Ok(f.clone());
        }
        let path = PROGRAMS[program].path;
        let f = Arc::new(function(device, crate::host::prebuilt::launch_path(path), &[]).await?);
        super::step_image::register(path, &f);
        *cell.borrow_mut() = Some(f.clone());
        Ok(f)
    }
}
