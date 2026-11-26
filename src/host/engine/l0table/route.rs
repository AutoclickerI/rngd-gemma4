use super::super::{RowInput, queue, tables};
use super::super::{int8::Int8Weights, segments::Rope};
use super::*;
use crate::host::load::SlidingLayer;

async fn upload<T: M>(
    device: &mut Device,
    words: Vec<i32>,
) -> Result<HbmTensor<i32, Chip, T>, Error> {
    HostTensor::<i32, T>::from_vec(words)
        .to_hbm(&mut device.pdma)
        .await
}

pub(crate) enum MediaSource<'a> {
    Tok1024(&'a HbmTensor<bf16, Chip, m![Tok1024, H]>),
    Tok1536(&'a HbmTensor<bf16, Chip, m![Tok1536, H]>),
}

pub(in crate::host::engine) struct MediaScratch {
    t128: HbmTensor<bf16, Chip, m![Tok128, H]>,
    t512: HbmTensor<bf16, Chip, m![Tok512, H]>,
    t1024: HbmTensor<bf16, Chip, m![Tok1024, H]>,
    t1536: HbmTensor<bf16, Chip, m![Tok1536, H]>,
}
impl MediaScratch {
    pub(super) async fn new(device: &mut Device) -> Result<Self, Error> {
        async fn zeros<T: AxisName>(
            device: &mut Device,
        ) -> Result<HbmTensor<bf16, Chip, m![T, H]>, Error> {
            HostTensor::<bf16, m![T, H]>::zero()
                .to_hbm(&mut device.pdma)
                .await
        }
        Ok(Self {
            t128: zeros(device).await?,
            t512: zeros(device).await?,
            t1024: zeros(device).await?,
            t1536: zeros(device).await?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::host::engine) async fn run(
        &mut self,
        device: &mut Device,
        source: MediaSource<'_>,
        inputs: &[RowInput<'_>],
        positions: &[usize],
        row_offsets: &[i32],
        weights: &Int8Weights,
        layer0: &SlidingLayer,
        rope: &Rope,
        qin: &mut HbmTensor<bf16, Chip, m![Rq, Hqs]>,
        k0: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
        v0: &mut HbmTensor<f8e4m3, Chip, m![Rs, Ns, Ds]>,
    ) -> Result<(), BoxError> {
        let mut active: Vec<_> = inputs
            .iter()
            .enumerate()
            .filter_map(|(i, row)| {
                matches!(row, RowInput::Image(..) | RowInput::Audio(..)).then_some(i)
            })
            .collect();
        let count = active.len();
        if count == 0 {
            return Ok(());
        }
        let size = match source {
            MediaSource::Tok1024(_) => Tok1024::SIZE,
            MediaSource::Tok1536(_) => Tok1536::SIZE,
        };
        let stage = tables::sliding_stage(size);
        let args = weights.pre(layer0, rope)?;
        macro_rules! run {
            ($x:ident, $bucket:ident, $t:ident, $field:ident, $pack:ident, [$($module:ident)::+] $kernel:ident) => {{
                active.resize($t::SIZE, active[count - 1]);
                let source = upload::<m![$t]>(device, active.iter().map(|&i| (i * H::SIZE * 2) as i32).collect()).await?;
                let slots = upload::<m![$t]>(device, active.iter().map(|&i| stage[i]).collect()).await?;
                let pos = upload::<m![$t]>(device, active.iter().map(|&i| (positions[i] * Ds::SIZE * 2) as i32).collect()).await?;
                let rows = upload::<m![$t, Hh]>(device, active.iter().flat_map(|&i| [row_offsets[2*i], row_offsets[2*i+1]]).collect()).await?;
                queue::queued!(device, crate::ops_l0table::serve::$pack, [$bucket::SIZE], ($x, &source, &mut self.$field))?;
                crate::host::prebuilt::queued_kernel!(device, $($module)::+::$kernel, (&self.$field, &pos, &rows, &slots, args.consts, args.next, qin, k0, v0))?;
            }};
            ($x:ident, $bucket:ident, compact128) => {
                run!($x, $bucket, Tok128, t128, pack_t128, [ops_core::smallbatch] pre_class512_t128)
            };
            ($x:ident, $bucket:ident, compact512) => {
                run!($x, $bucket, Tok512, t512, pack_t512, [ops_core] pre_int8_fp8kv_t512)
            };
        }
        match source {
            MediaSource::Tok1536(x) => match count {
                1..=128 => run!(x, Tok1536, compact128),
                129..=512 => run!(x, Tok1536, compact512),
                513..=1024 => run!(x, Tok1536, Tok1024, t1024, pack_t1024, [ops_core] pre_int8_fp8kv_t1024),
                _ => run!(x, Tok1536, Tok1536, t1536, pack_t1536, [ops_core] pre_int8_fp8kv_t1536),
            },
            MediaSource::Tok1024(x) => match count {
                1..=128 => run!(x, Tok1024, compact128),
                129..=512 => run!(x, Tok1024, compact512),
                _ => run!(x, Tok1024, Tok1024, t1024, pack_t1024, [ops_core] pre_int8_fp8kv_t1024),
            },
        }
        Ok(())
    }
}
