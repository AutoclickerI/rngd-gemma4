//! Per-bucket step buffers and the bucket's kernel launches.
//!
//! A step's token-major tensors are typed by its bucket axis; the official
//! attention moves them as head halves (`[TokN, Hh,..]`), so `retype` moves a buffer between
//! typings without a copy. One `Bucket` implementation per step width
//! names the bucket's concrete kernels (embedding, the core's and the INT8 carrier's segments,
//! widening); the engine's step is generic over it.
use furiosa_opt_std::prelude::*;
use crate::axes::MediaPool;

use crate::Chip;
use crate::axes::*;
use crate::host::engine::attention::AttentionState;
use crate::host::engine::int8::{
    MidFsInt8, MidSfInt8, MidSsInt8, SlidingKv, int8_full_kernels,
};
use crate::host::engine::queue::queued;
use crate::host::engine::tables::{self, AttnRow};
use crate::host::load::Weights;
use crate::ops_step;
use crate::ops_media::PutRows;

/// Moves a tensor to another typing of the same bytes (an axis regrouping).
pub(crate) fn retype<D: Scalar, E: M, E2: M>(tensor: &mut HbmTensor<D, Chip, E>) -> HbmTensor<D, Chip, E2> {
    unsafe { std::mem::replace(tensor, HbmTensor::new()).reshape() }
}

/// Puts a retyped tensor back.
pub(crate) fn restore<D: Scalar, E: M, E2: M>(tensor: &mut HbmTensor<D, Chip, E>, retyped: HbmTensor<D, Chip, E2>) {
    *tensor = unsafe { retyped.reshape() };
}

async fn zeros<D: MaterializableScalar + num_traits::Zero, E: M>(device: &mut Device) -> Result<HbmTensor<D, Chip, E>, Error> {
    HostTensor::<D, E>::zero().to_hbm(&mut device.pdma).await
}

async fn upload<D: MaterializableScalar, E: M>(device: &mut Device, values: Vec<D>, into: &mut HbmTensor<D, Chip, E>) -> Result<(), Error> {
    HostTensor::<D, E>::from_vec(values).to_hbm(&mut device.pdma).output(into).await
}

/// A per-step `upload`, booked in the profile's transfer ledger under `site`. The tensor binds itself again
/// (`uploads::unalias`).
async fn upload_as<D: MaterializableScalar, E: M>(device: &mut Device, values: Vec<D>, into: &mut HbmTensor<D, Chip, E>) -> Result<(), Error> {
    crate::host::engine::uploads::unalias(&*into)?;
    upload(device, values, into).await
}

/// The token-major step tensors of one bucket (`T` rows), its per-row
/// metadata and its attention tables.
pub(crate) struct Dense<T: AxisName> {
    /// The residual stream.
    pub x: HbmTensor<bf16, Chip, m![T, H]>,
    /// Per row: the embedding-table row (byte offset) and the RoPE offsets of each family.
    tok: HbmTensor<i32, Chip, m![T]>,
    pub(crate) rope_s: HbmTensor<i32, Chip, m![T]>,
    pub(crate) rope_f: HbmTensor<i32, Chip, m![T]>,
    pub(crate) stage: HbmTensor<i32, Chip, m![T]>,
    /// Row `i`'s byte offset in the Tok1024 residual (`ops_step::widen_*`; constant).
    widen_x: HbmTensor<i32, Chip, m![T]>,
    pub(crate) publish_s: HbmTensor<i32, Chip, m![T]>,
    pub(crate) publish_f: HbmTensor<i32, Chip, m![T]>,
    /// Attention: each row's two head halves in the family's row space (byte offsets).
    pub rows_s: HbmTensor<i32, Chip, m![T, Hh]>,
    pub rows_f: HbmTensor<i32, Chip, m![T, Hh]>,
    /// The second set of the host-written tables above (`StepTables`), allocated only for next-step preparation (`double`,
    /// `engine::ahead`): the next step's tables are posted into it while the running step's queued launches still read
    /// the first set, and the next step swaps it in.
    alt: Option<StepTables<T>>,
    feedback: Option<super::feedback::Metadata<T>>,
}

/// The tables the host writes into a bucket's `Dense` every step: `tok`, `rope_s`, `rope_f`, `publish_s`, `publish_f`,
/// `rows_s` and `rows_f` (the constants `stage` and `widen_x` are written once).
struct StepTables<T: AxisName> {
    tok: HbmTensor<i32, Chip, m![T]>,
    rope_s: HbmTensor<i32, Chip, m![T]>,
    rope_f: HbmTensor<i32, Chip, m![T]>,
    publish_s: HbmTensor<i32, Chip, m![T]>,
    publish_f: HbmTensor<i32, Chip, m![T]>,
    rows_s: HbmTensor<i32, Chip, m![T, Hh]>,
    rows_f: HbmTensor<i32, Chip, m![T, Hh]>,
}

impl<T: AxisName> Dense<T> {
    /// Allocates the second table set (`alt`), zeroed like the first.
    pub async fn double(&mut self, device: &mut Device) -> Result<(), Error> {
        if self.alt.is_none() {
            self.alt = Some(StepTables {
                tok: zeros(device).await?,
                rope_s: zeros(device).await?,
                rope_f: zeros(device).await?,
                publish_s: zeros(device).await?,
                publish_f: zeros(device).await?,
                rows_s: zeros(device).await?,
                rows_f: zeros(device).await?,
            });
        }
        Ok(())
    }

    /// Swaps the second table set in (no-op without one): uploads then go into the set the previous step of this
    /// bucket did not use, and the launches the previous step queued keep the buffers they bound.
    pub fn swap_tables(&mut self) {
        if let Some(alt) = self.alt.as_mut() {
            std::mem::swap(&mut self.tok, &mut alt.tok);
            std::mem::swap(&mut self.rope_s, &mut alt.rope_s);
            std::mem::swap(&mut self.rope_f, &mut alt.rope_f);
            std::mem::swap(&mut self.publish_s, &mut alt.publish_s);
            std::mem::swap(&mut self.publish_f, &mut alt.publish_f);
            std::mem::swap(&mut self.rows_s, &mut alt.rows_s);
            std::mem::swap(&mut self.rows_f, &mut alt.rows_f);
        }
    }

    pub async fn new(device: &mut Device) -> Result<Self, Error> {
        let n = T::SIZE;
        let mut dense = Self {
            x: zeros(device).await?,
            tok: zeros(device).await?,
            rope_s: zeros(device).await?,
            rope_f: zeros(device).await?,
            stage: zeros(device).await?,
            widen_x: zeros(device).await?,
            publish_s: zeros(device).await?,
            publish_f: zeros(device).await?,
            rows_s: zeros(device).await?,
            rows_f: zeros(device).await?,
            alt: None,
            feedback: None,
        };
        upload(device, tables::sliding_stage(n), &mut dense.stage).await?;
        upload(device, (0..n).map(|i| (i * H::SIZE * 2) as i32).collect(), &mut dense.widen_x).await?;
        Ok(dense)
    }

    /// Next-step preparation (`ahead`): the step's `tok` alone, its other tables posted during the previous step (rows past the
    /// given ones use token 0).
    pub async fn upload_tok(&mut self, device: &mut Device, tokens: &[usize]) -> Result<(), Error> {
        let mut tok: Vec<i32> = tokens.iter().map(|&t| (t * H::SIZE * 2) as i32).collect();
        tok.resize(T::SIZE, 0);
        upload_as(device, tok, &mut self.tok).await
    }

    /// The step's small tables in `table_tensors` order.
    pub fn table_images(rows: &[AttnRow<'_>], rows_s: Vec<i32>, rows_f: Vec<i32>, tokens: &[usize], positions: &[usize]) -> [Vec<i32>; 7] {
        assert_eq!(rows.len(), T::SIZE, "tables cover the whole bucket");
        assert_eq!(rows_s.len(), T::SIZE * Hh::SIZE);
        assert_eq!(rows_f.len(), T::SIZE * Hh::SIZE);
        let pad = |mut v: Vec<i32>| {
            v.resize(T::SIZE, 0);
            v
        };
        [
            tables::sliding_publish(rows),
            tables::full_publish_e4m3(rows),
            rows_s,
            rows_f,
            pad(tokens.iter().map(|&t| (t * H::SIZE * 2) as i32).collect()),
            pad(positions.iter().map(|&p| (p * Ds::SIZE * 2) as i32).collect()),
            pad(positions.iter().map(|&p| (p * Df::SIZE * 2) as i32).collect()),
        ]
    }

    /// The small tables' tensors in `table_images` order.
    pub fn table_tensors(&self) -> [&dyn furiosa_opt_std::runtime::DeviceSend; 7] {
        [&self.publish_s, &self.publish_f, &self.rows_s, &self.rows_f, &self.tok, &self.rope_s, &self.rope_f]
    }
}

/// The step buffers of every bucket.
pub(crate) struct DenseSet {
    pub b128: Dense<Tok128>,
    pub b1024: Dense<Tok1024>,
    pub b1536: Dense<Tok1536>,
    pub compact_rows: [HbmTensor<i32, Chip, m![Out128]>; 2],
}

impl DenseSet {
    /// Every bucket's second table set (`Dense::double`).
    pub async fn double(&mut self, device: &mut Device) -> Result<(), Error> {
        self.b128.double(device).await?;
        self.b1024.double(device).await?;
        self.b1536.double(device).await
    }

    pub async fn new(device: &mut Device) -> Result<Self, Error> {
        Ok(Self {
            b128: Dense::new(device).await?,
            b1024: Dense::new(device).await?,
            b1536: Dense::new(device).await?,
            compact_rows: [zeros(device).await?, zeros(device).await?],
        })
    }
}

/// One token bucket's kernels.
pub(crate) trait Bucket {
    type T: AxisName;

    fn dense(set: &mut DenseSet) -> &mut Dense<Self::T>;

    /// The step's embedded input `x`: every row's token embedding (`ops_step::embed_rows_*`).
    async fn resolve_tokens(device: &mut Device, dense: &mut Dense<Self::T>, previous: &super::pipeline::TokenTable, inputs: &[super::RowInput<'_>]) -> Result<(), Error>;
    async fn embed(device: &mut Device, dense: &mut Dense<Self::T>, weights: &Weights) -> Result<(), Error>;

    async fn put_media(
        device: &mut Device, staged: &HbmTensor<bf16, Chip, m![Tok1024, H]>,
        source: &HbmTensor<i32, Chip, m![PutRows]>, destination: &HbmTensor<i32, Chip, m![PutRows]>,
        dense: &mut Dense<Self::T>,
    ) -> Result<(), Error>;

    async fn put_resident_tail(
        device: &mut Device, pool: &HbmTensor<bf16, Chip, m![MediaPool, H]>,
        source: &HbmTensor<i32, Chip, m![PutRows]>, destination: &HbmTensor<i32, Chip, m![PutRows]>,
        dense: &mut Dense<Self::T>,
    ) -> Result<(), Error>;

    async fn embed_media(device: &mut Device, dense: &mut Dense<Self::T>, weights: &Weights,
        pool: &HbmTensor<bf16, Chip, m![MediaPool, H]>,
        source: &HbmTensor<i32, Chip, m![PutRows]>,
        destination: &HbmTensor<i32, Chip, m![PutRows]>) -> Result<(), Error>;

    fn table_media_source(dense: &Dense<Self::T>) -> super::l0table::MediaSource<'_>;

    async fn table_pre(
        device: &mut Device, dense: &Dense<Self::T>, tables: crate::ops_l0table::serve::Tables<'_>,
        rope: &super::segments::Rope, qin: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
        k: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>, v: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
    ) -> Result<(), Error>;

    /// `mid_ss` on the INT8 carrier (`ops_core::mid_ss_int8_tN`; contract 3.2): the core
    /// `mid_ss`'s tables, pools and moves with the rotated INT8 stages and layer `l`'s prepared
    /// weights (`args`). Buckets with the kernel only (`carriers::int8_kernel`).
    #[allow(clippy::too_many_arguments)]
    async fn int8_mid_ss(
        device: &mut Device,
        dense: &mut Dense<Self::T>,
        layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
        args: MidSsInt8<'_>,
        official: &mut AttentionState,
        k_l: &mut HbmTensor<SlidingKv, Chip, m![Rs, Ns, Ds]>,
        v_l: &mut HbmTensor<SlidingKv, Chip, m![Rs, Ns, Ds]>,
        k_n: &mut HbmTensor<SlidingKv, Chip, m![Rs, Ns, Ds]>,
        v_n: &mut HbmTensor<SlidingKv, Chip, m![Rs, Ns, Ds]>,
    ) -> Result<(), Error>;

    /// `mid_sf` on the INT8 carrier (`ops_core::full_i8::mid_sf_int8_fp8kv_tN`, the INT8 Full transition):
    /// the hybrid `mid_sf`'s tables, E4M3 Sliding pools and moves with the INT8 Full QKV.
    #[allow(clippy::too_many_arguments)]
    async fn int8_mid_sf(
        device: &mut Device,
        dense: &mut Dense<Self::T>,
        layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
        args: MidSfInt8<'_>,
        official: &mut AttentionState,
        k_l: &mut HbmTensor<SlidingKv, Chip, m![Rs, Ns, Ds]>,
        v_l: &mut HbmTensor<SlidingKv, Chip, m![Rs, Ns, Ds]>,
        k_n: &mut HbmTensor<f8e4m3, Chip, m![Rf, Df]>,
        v_n: &mut HbmTensor<f8e4m3, Chip, m![Rf, Df]>,
    ) -> Result<(), Error>;

    /// `mid_fs` on the INT8 carrier (`ops_core::full_i8::mid_fs_int8_fp8kv_tN`): the hybrid `mid_fs`'s
    /// with the INT8 Full O.
    #[allow(clippy::too_many_arguments)]
    async fn int8_mid_fs(
        device: &mut Device,
        dense: &mut Dense<Self::T>,
        layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
        args: MidFsInt8<'_>,
        official: &mut AttentionState,
        k_n: &mut HbmTensor<SlidingKv, Chip, m![Rs, Ns, Ds]>,
        v_n: &mut HbmTensor<SlidingKv, Chip, m![Rs, Ns, Ds]>,
    ) -> Result<(), Error>;

    /// The step's residual into the Tok1024 tensor `last` reads.
    async fn widen(device: &mut Device, set: &mut DenseSet) -> Result<(), Error>;
}

macro_rules! bucket {
    ($bucket:ident, $field:ident, $tok:ident, $embed:ident, $resolve:ident,
     $widen:tt, $lookup:ident, [ $int8:tt, $int8sf:tt, $int8fs:tt]) => {
        pub(crate) struct $bucket;

        impl Bucket for $bucket {
            type T = $tok;

            fn dense(set: &mut DenseSet) -> &mut Dense<$tok> {
                &mut set.$field
            }

            async fn resolve_tokens(device: &mut Device, d: &mut Dense<$tok>, previous: &super::pipeline::TokenTable, inputs: &[super::RowInput<'_>]) -> Result<(), Error> {
                // Packed buffer regions are substituted by the launch queue.
                let region_binding = super::queue::queueing();
                if region_binding && super::feedback::Metadata::<$tok>::fits(inputs) {
                    if d.feedback.is_none() { d.feedback = Some(super::feedback::Metadata::new(device).await?); }
                    let bank = d.feedback.as_mut().unwrap().prepare(device, inputs).await?;
                    super::uploads::unalias(&d.tok)?;
                    let result = crate::host::prebuilt::queued_kernel!(device, ops_sampler::feedback::$resolve, (previous, &bank.sources, &bank.literals, &mut d.tok));
                    bank.submitted();
                    return result;
                }
                let mut sources = Vec::new();
                let mut literals = Vec::new();
                for i in inputs {
                    let (source, literal) = match i { super::RowInput::DeviceToken(owner) => ((*owner * Dummy8::SIZE * 4) as i32, -1), _ => (0, i.token() as i32) };
                    sources.push(source); literals.extend([literal; Dummy8::SIZE]);
                }
                sources.resize($tok::SIZE, 0); literals.resize($tok::SIZE*Dummy8::SIZE, 0);
                let sources: HbmTensor<i32, Chip, m![$tok]> = HostTensor::<i32, m![$tok]>::from_vec(sources).to_hbm(&mut device.pdma).await?;
                let literals: HbmTensor<i32, Chip, m![$tok, Dummy8]> = HostTensor::<i32, m![$tok, Dummy8]>::from_vec(literals).to_hbm(&mut device.pdma).await?;
                super::uploads::unalias(&d.tok)?;
                crate::host::prebuilt::queued_kernel!(device, ops_sampler::feedback::$resolve, (previous, &sources, &literals, &mut d.tok))
            }
            async fn embed(device: &mut Device, d: &mut Dense<$tok>, weights: &Weights) -> Result<(), Error> {
                crate::host::prebuilt::queued_kernel!(device, ops_step::$embed, (&weights.embedding_table, &d.tok, &mut d.x))
            }

            async fn put_media(
                device: &mut Device, staged: &HbmTensor<bf16, Chip, m![Tok1024, H]>,
                source: &HbmTensor<i32, Chip, m![PutRows]>, destination: &HbmTensor<i32, Chip, m![PutRows]>,
                dense: &mut Dense<$tok>,
            ) -> Result<(), Error> {
                let x = &mut dense.x;
                bucket!(@put_media $tok, device, staged, source, destination, x)
            }

            async fn put_resident_tail(
                device: &mut Device, pool: &HbmTensor<bf16, Chip, m![MediaPool, H]>,
                source: &HbmTensor<i32, Chip, m![PutRows]>, destination: &HbmTensor<i32, Chip, m![PutRows]>,
                dense: &mut Dense<$tok>,
            ) -> Result<(), Error> {
                let x = &mut dense.x;
                bucket!(@put_resident_tail $tok, device, pool, source, destination, x)
            }

            async fn embed_media(device: &mut Device, d: &mut Dense<$tok>, weights: &Weights,
                pool: &HbmTensor<bf16, Chip, m![MediaPool, H]>,
                source: &HbmTensor<i32, Chip, m![PutRows]>,
                destination: &HbmTensor<i32, Chip, m![PutRows]>) -> Result<(), Error> {
                bucket!(@media $embed, device, d, weights, pool, source, destination)
            }

            async fn int8_mid_ss(
                device: &mut Device,
                d: &mut Dense<$tok>,
                layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
                args: MidSsInt8<'_>,
                official: &mut AttentionState,
                k_l: &mut HbmTensor<SlidingKv, Chip, m![Rs, Ns, Ds]>,
                v_l: &mut HbmTensor<SlidingKv, Chip, m![Rs, Ns, Ds]>,
                k_n: &mut HbmTensor<SlidingKv, Chip, m![Rs, Ns, Ds]>,
                v_n: &mut HbmTensor<SlidingKv, Chip, m![Rs, Ns, Ds]>,
            ) -> Result<(), Error> {
                bucket!(@int8_ss $int8, device, d, layer_scalar, args, official, k_l, v_l, k_n, v_n)
            }

            fn table_media_source(d: &Dense<$tok>) -> super::l0table::MediaSource<'_> {
                bucket!(@table_media $tok, d)
            }

            async fn table_pre(
                device: &mut Device, d: &Dense<$tok>, tables: crate::ops_l0table::serve::Tables<'_>,
                rope: &super::segments::Rope, qin: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
                k: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>, v: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
            ) -> Result<(), Error> {
                bucket!(@table_pre $lookup, device, d, tables, rope, qin, k, v)
            }

            async fn int8_mid_sf(
                device: &mut Device,
                d: &mut Dense<$tok>,
                layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
                args: MidSfInt8<'_>,
                official: &mut AttentionState,
                k_l: &mut HbmTensor<SlidingKv, Chip, m![Rs, Ns, Ds]>,
                v_l: &mut HbmTensor<SlidingKv, Chip, m![Rs, Ns, Ds]>,
                k_n: &mut HbmTensor<f8e4m3, Chip, m![Rf, Df]>,
                v_n: &mut HbmTensor<f8e4m3, Chip, m![Rf, Df]>,
            ) -> Result<(), Error> {
                bucket!(@int8_sf $int8sf, device, d, layer_scalar, args, official, k_l, v_l, k_n, v_n)
            }

            async fn int8_mid_fs(
                device: &mut Device,
                d: &mut Dense<$tok>,
                layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
                args: MidFsInt8<'_>,
                official: &mut AttentionState,
                k_n: &mut HbmTensor<SlidingKv, Chip, m![Rs, Ns, Ds]>,
                v_n: &mut HbmTensor<SlidingKv, Chip, m![Rs, Ns, Ds]>,
            ) -> Result<(), Error> {
                bucket!(@int8_fs $int8fs, device, d, layer_scalar, args, official, k_n, v_n)
            }

            async fn widen(device: &mut Device, set: &mut DenseSet) -> Result<(), Error> {
                bucket!(@widen $widen, device, set, $field)
            }
        }
    };
    (@table_media Tok1024, $d:ident) => {
        super::l0table::MediaSource::Tok1024(&$d.x)
    };
    (@table_media Tok1536, $d:ident) => {
        super::l0table::MediaSource::Tok1536(&$d.x)
    };
    (@table_media $tok:ident, $d:ident) => {{
        let _ = $d;
        unreachable!("media rows require a prefill bucket")
    }};
    (@table_pre none, $($args:ident),+) => {{
        let _ = ($($args),+);
        unreachable!("the production step floor excludes rotated buckets")
    }};
    (@table_pre $lookup:ident, $device:ident, $d:ident, $tables:ident, $rope:ident, $qin:ident, $k:ident, $v:ident) => {{
        queued!($device, crate::ops_l0table::serve::$lookup,
            (&$d.tok, &$d.rope_s, &$d.rows_s, &$d.stage, $tables, &$rope.cos_s, &$rope.sin_s, $qin, $k, $v))
    }};
    (@int8_sf ($module:ident::$kernel:ident), $device:ident, $d:ident, $scalar:ident, $args:ident, $official:ident, $k_l:ident, $v_l:ident, $k_n:ident, $v_n:ident) => {{
        int8_full_kernels!(
            {
                let MidSfInt8 { consts, o_f8: o, ffn_f8: ffn, next_f8: next } = $args;



                crate::host::prebuilt::queued_kernel!(
                    $device,
                    ops_core::$module::$kernel,
                    (
                        &$official.sliding.out,
                        &$d.rows_s,
                        &$d.stage,
                        &$d.publish_s,
                        &$d.rope_f,
                        &$d.rows_f,
                        &$d.publish_f,
                        &mut $d.x,
                        consts,
                        o,
                        ffn,
                        $scalar,
                        next,
                        $k_l,
                        $v_l,
                        &mut $official.full.qin,
                        $k_n,
                        $v_n,
                    ),
                )
            }
        )
    }};
    (@int8_ss none, $device:ident, $d:ident, $scalar:ident, $args:ident, $official:ident, $k_l:ident, $v_l:ident, $k_n:ident, $v_n:ident) => {{
        let _ = (&$device, &$d, &$scalar, &$args, &$official, &$k_l, &$v_l, &$k_n, &$v_n);
        Err(Error::Uncompiled { path: "ops_core: no mid_ss_int8 at this bucket (carriers::int8_kernel)", key: vec![] })
    }};
    (@int8_fs none, $device:ident, $d:ident, $scalar:ident, $args:ident, $official:ident, $k_n:ident, $v_n:ident) => {{
        let _ = (&$device, &$d, &$scalar, &$args, &$official, &$k_n, &$v_n);
        Err(Error::Uncompiled { path: "ops_core: no Q4 Full-to-Sliding image for this bucket", key: vec![] })
    }};
    (@int8_ss ($first:ident $(::$kernel:ident)+), $device:ident, $d:ident, $scalar:ident, $args:ident, $official:ident, $k_l:ident, $v_l:ident, $k_n:ident, $v_n:ident) => {{
        let MidSsInt8 { consts, o, next, q4: ffn } = $args;

        crate::host::prebuilt::queued_kernel!($device, ops_core::$first$(::$kernel)+, (&$official.sliding.out, &$d.rows_s, &$d.stage, &$d.publish_s, &$d.rope_s, &mut $d.x, consts, o, ffn, $scalar, next, $k_l, $v_l, &mut $official.sliding.qin, $k_n, $v_n,))
    }};
    (@int8_fs ($first:ident $(::$kernel:ident)+), $device:ident, $d:ident, $scalar:ident, $args:ident, $official:ident, $k_n:ident, $v_n:ident) => {{
        let MidFsInt8 { consts, o, next, q4: ffn } = $args;

        crate::host::prebuilt::queued_kernel!($device, ops_core::$first$(::$kernel)+, (&$official.full.out, &$d.rows_f, &$d.rope_s, &$d.rows_s, &$d.stage, &mut $d.x, consts, o, ffn, $scalar, next, &mut $official.sliding.qin, $k_n, $v_n,))
    }};
    (@put_media Tok1536, $device:ident, $staged:ident, $source:ident, $destination:ident, $x:ident) => {{
        launch(crate::ops_media::put_rows_at, (&mut *$device, $staged, $source, $destination, $x)).await
    }};
    (@put_media Tok128, $($args:ident),+) => {{
        let _ = ($($args),+);
        unreachable!("128-row decode steps contain no media input")
    }};
    (@put_media $tok:ident, $device:ident, $staged:ident, $source:ident, $destination:ident, $x:ident) => {{
        let _ = $source;
        launch(crate::ops_media::put_rows, (&mut *$device, $staged, $destination, $x)).await
    }};
    (@put_resident_tail Tok1536, $device:ident, $pool:ident, $source:ident, $destination:ident, $x:ident) => {{
        queued!($device, crate::ops_media::put_pool_rows, [Tok1536::SIZE], ($pool, $source, $destination, $x))
    }};
    (@put_resident_tail $tok:ident, $($args:ident),+) => {{
        let _ = ($($args),+);
        unreachable!("resident media batches fit in a narrow step")
    }};
    (@media embed_rows_t128, $device:ident, $d:ident, $weights:ident, $pool:ident, $source:ident, $destination:ident) => {{
        let _ = ($device, $d, $weights, $pool, $source, $destination);
        unreachable!("128-row decode steps contain no media input")
    }};
    (@media embed_rows_t1024, $device:ident, $d:ident, $weights:ident, $pool:ident, $source:ident, $destination:ident) => {{
        queued!($device, ops_step::embed_media_t1024, (&$weights.embedding_table, &$d.tok, $pool, $source, $destination, &mut $d.x))
    }};
    (@media embed_rows_t1536, $device:ident, $d:ident, $weights:ident, $pool:ident, $source:ident, $destination:ident) => {{
        crate::host::prebuilt::queued_kernel!($device, ops_step::embed_media_t1536, (&$weights.embedding_table, &$d.tok, $pool, $source, $destination, &mut $d.x))
    }};
    (@widen none, $device:ident, $set:ident, $field:ident) => {{
        let _ = (&$device, &$set);
        Ok(())
    }};
    (@widen $kernel:ident, $device:ident, $set:ident, $field:ident) => {{
        let DenseSet { $field: small, b1024, .. } = $set;
        queued!($device, ops_step::$kernel, (&small.x, &small.widen_x, &mut b1024.x))
    }};

}

bucket!(B128, b128, Tok128, embed_rows_t128, resolve_t128,
    widen_t128, lookup_t128, [none, (smallbatch::mid_sf_class512_t128), none]);
bucket!(B1024, b1024, Tok1024, embed_rows_t1024, resolve_t1024,
    none, lookup_t1024, [(cohort_q4::q4::mid_ss_q4_t1024), (full_i8::mid_sf_int8_fp8kv_t1024), (cohort_q4::q4::mid_fs_q4_t1024)]);

bucket!(B1536, b1536, Tok1536, embed_rows_t1536, resolve_t1536_three512,
    none, lookup_t1536, [(q4serve::mid_ss_t1536), (full_i8::mid_sf_int8_fp8kv_t1536), (q4serve::mid_fs_t1536)]);
