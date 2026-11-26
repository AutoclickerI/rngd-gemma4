// Shared arithmetic; AudioRows is the geometry alias supplied by the parent module.
const GROUPS: usize = AudioRows::SIZE / 32;
const AA_F32: f32 = Aa::SIZE as f32;
const _: () = assert!(AudioRows::SIZE % 32 == 0);

// ------------------------------------------------------------------ row groups

/// 32 rows x 8 shards; each slice holds `KL` consecutive elements of one frame.
type RowShards<K, const KL: usize> = m![AudioRows % 32, K / {KL}];
type RowReduced = m![AudioRows % 32, Dummy8];
type Group<K, const KL: usize> = DmTensor<bf16, Chip, Cluster, RowShards<K, KL>, m![K % {KL}]>;
type GroupF32<K, const KL: usize> = DmTensor<f32, Chip, Cluster, RowShards<K, KL>, m![K % {KL}]>;
type Stat<K, const KL: usize> = DmTensor<f32, Chip, Cluster, RowShards<K, KL>, m![1 # 8]>;
type StatReduced = DmTensor<f32, Chip, Cluster, RowReduced, m![1 # 8]>;
type StatVrf<K, const KL: usize> = VrfTensor<f32, Chip, Cluster, RowShards<K, KL>, m![1 # 8]>;

/// Per-shard sum of squares of an f32 frame.
fn partial_sum_squares<K: AxisName, const KL: usize>(device: &mut Device, x: &GroupF32<K, KL>) -> Stat<K, KL> {
    device
        .main
        .begin(x.view())
        .fetch::<m![K % {KL} / 8], m![K % {KL} % 8]>()
        .collect::<m![K % {KL} / 8], m![K % {KL} % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![K % {KL} / 4], m![K % {KL} % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<K, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

/// Sum of the eight shard partials of each frame.
fn reduce_rows<K: AxisName, const KL: usize>(device: &mut Device, partial: &Stat<K, KL>) -> StatReduced {
    device
        .main
        .begin(partial.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<RowReduced, m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

/// `sqrt(x / size + EPS)`, the baseline's RMS: its `mean_square` divides by `Aa` and
/// adds `EPS` before the square root (`device/vision/projection.rs`'s two-stage
/// add-then-sqrt idiom — the clip runs on the wide state, the sqrt in its own stage).
fn rms(device: &mut Device, x: &StatReduced, size: f32) -> StatReduced {
    let shifted: StatReduced = device
        .main
        .begin(x.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_div(size)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    device
        .main
        .begin(shifted.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

fn stat_vrf<K: AxisName, const KL: usize>(device: &mut Device, x: StatReduced) -> StatVrf<K, KL> {
    let x: Stat<K, KL> = unsafe { x.reshape() };
    device
        .sub
        .begin(x.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}

/// Weight-free RMS normalization of every frame: `x / sqrt(mean(x^2) + EPS)`.
fn rms_norm<K: AxisName, const KL: usize>(device: &mut Device, x: &Group<K, KL>, size: f32) -> Group<K, KL> {
    let x32: GroupF32<K, KL> = device
        .main
        .begin(x.view())
        .fetch::<m![K % {KL} / 16], m![K % {KL} % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![K % {KL} / 8], m![K % {KL} % 8]>()
        .commit_trim::<m![K % {KL} % 8]>()
        .commit();
    let squares = partial_sum_squares::<K, KL>(device, &x32);
    let squares = reduce_rows::<K, KL>(device, &squares);
    let rms = rms(device, &squares, size);
    let rms = stat_vrf::<K, KL>(device, rms);
    device
        .main
        .begin(x32.view())
        .fetch::<m![K % {KL} / 8], m![K % {KL} % 8]>()
        .collect::<m![K % {KL} / 8], m![K % {KL} % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![K % {KL} / 4], m![K % {KL} % 4]>()
        .vector_fp_div(&rms)
        .vector_widen_concat::<m![K % {KL} / 8], m![K % {KL} % 8]>()
        .vector_final()
        .cast::<bf16, m![K % {KL} % 8 # 16]>()
        .commit_trim::<m![K % {KL} % 8]>()
        .commit()
}

fn load_group<K: AxisName, const KL: usize>(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![AudioRows, K]>,
    _group: usize,
) -> Group<K, KL> {
    x.view()
        .tile::<m![AudioRows / 32], 1, m![AudioRows / 32 = 1 # {GROUPS}, AudioRows % 32, K]>(self::row_group_index!(_group))
        .to_dm(&mut device.tdma)
}

fn store_group<K: AxisName, const KL: usize>(
    device: &mut Device,
    x: &Group<K, KL>,
    out: &mut HbmTensor<bf16, Chip, m![AudioRows, K]>,
    _group: usize,
) {
    x.view().to_hbm_view(
        &mut device.tdma,
        out.view_mut().tile::<m![AudioRows / 32], 1, m![AudioRows / 32 = 1 #{!} {GROUPS}, AudioRows % 32, K]>(self::row_group_index!(_group)),
    );
}

// ------------------------------------------------------------- linear tiles

type LinCluster<const B2: usize, const B: usize> = m![AudioRows % {B2} / {B}];
/// Owner `o` holds output rows `[o*S, o*S + S)`, shard `k` columns `[k*KL, k*KL + KL)`.
/// The shard axis pads `Aa / KL = 4` live shards to the device's 8-slice group so the DM
/// allocation spans the whole 256-slice cluster (the pad copies read DM zeros and vanish
/// in the acc's inter-slice reduce — the same 4-live-of-8-copy reduce the CPU gate verifies).
type LinRC<N, K, const S: usize, const OP: usize, const KL: usize> = m![N / {S} # {OP}, K / {KL} # 8];
type LinRR<N, const S: usize, const OP: usize, const KS: usize> = m![N / {S} # {OP}, 1 # {KS}];
type LinW<N, K, const S: usize, const OP: usize, const KL: usize, const B2: usize, const B: usize> =
    TrfTensor<bf16, Chip, LinCluster<B2, B>, LinRC<N, K, S, OP, KL>, m![N % {S} = 32 % 8], m![N % {S} = 32 / 8, K % {KL}]>;
type LinIn<N, K, const S: usize, const OP: usize, const KL: usize, const B2: usize, const B: usize> =
    DmTensor<bf16, Chip, LinCluster<B2, B>, LinRC<N, K, S, OP, KL>, m![AudioRows % {B2} % {B}, K % {KL}]>;
type LinAcc<N, const S: usize, const OP: usize, const KS: usize, const B2: usize, const B: usize> =
    DmTensor<bf16, Chip, LinCluster<B2, B>, LinRR<N, S, OP, KS>, m![AudioRows % {B2} % {B}, N % {S}]>;

/// 32 weight rows of panel `panel` per owner, every K-shard, into TRF.
fn weight32<N: AxisName, K: AxisName, const S: usize, const OP: usize, const KL: usize, const B2: usize, const B: usize>(
    device: &mut Device,
    w: &HbmTensor<bf16, Chip, m![N, K]>,
    panel: usize,
) -> LinW<N, K, S, OP, KL, B2, B> {
    let dm: DmTensor<bf16, Chip, LinCluster<B2, B>, LinRC<N, K, S, OP, KL>, m![N % {S} = 32, K % {KL}]> = w
        .view()
        .tile::<m![N % {S}], 32, m![N / {S}, N % {S} = 32 # {S}, K]>(panel * 32)
        .to_dm(&mut device.tdma);
    device
        .sub
        .begin(dm.view())
        .fetch::<m![N % {S} = 32 % 8, N % {S} = 32 / 8, K % {KL} / 16], m![K % {KL} % 16]>()
        .collect::<m![N % {S} = 32 % 8, N % {S} = 32 / 8, K % {KL} / 16], m![K % {KL} % 16]>()
        .to_trf()
}

/// Streams the activation rows against one panel, reduces the K-shards in f32 and
/// commits the panel's 32 columns into the owner's accumulator.
fn dot_into<N: AxisName, K: AxisName, const S: usize, const OP: usize, const KL: usize, const KS: usize, const B2: usize, const B: usize>(
    device: &mut Device,
    x: &LinIn<N, K, S, OP, KL, B2, B>,
    w: &LinW<N, K, S, OP, KL, B2, B>,
    acc: &mut LinAcc<N, S, OP, KS, B2, B>,
    panel: usize,
) {
    device
        .main
        .begin(x.view())
        .fetch::<m![AudioRows % {B2} % {B}, K % {KL} / 16], m![K % {KL} % 16]>()
        .collect::<m![AudioRows % {B2} % {B}, K % {KL} / 16], m![K % {KL} % 16]>()
        .contract_outer::<m![AudioRows % {B2} % {B}, K % {KL} / 32, N % {S} = 32 / 8], m![K % {KL} % 32], _, _, _>(w)
        .contract_packet::<m![1]>()
        .contract_time::<m![AudioRows % {B2} % {B}, N % {S} = 32 / 8]>()
        .contract_lane::<m![AudioRows % {B2} % {B}, N % {S} = 32 / 8], m![N % {S} = 32 % 8 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<LinRR<N, S, OP, KS>, m![AudioRows % {B2} % {B}, N % {S} = 32 / 8]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![N % {S} = 32 % 8 # 16]>()
        .commit_trim::<m![N % {S} = 32 % 8]>()
        .commit_view(acc.view_mut().tile::<m![N % {S}], 32, m![AudioRows % {B2} % {B}, N % {S} = 32 #{!} {S}]>(panel * 32));
}

/// `out[tile rows] = x[tile rows] * w^T` for one B2-row tile; S = 128 owner rows (four
/// 32-row panels per owner), the panel list spelled out so the scheduler overlaps the
/// next panel's weight load with the current contraction.
fn linear_tile<N: AxisName, K: AxisName, const KL: usize, const B2: usize, const B: usize, const NT: usize>(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![AudioRows, K]>,
    w: &HbmTensor<bf16, Chip, m![N, K]>,
    out: &mut HbmTensor<bf16, Chip, m![AudioRows, N]>,
    tile: usize,
) {
    const S: usize = 128;
    const OP: usize = 32;
    const KS: usize = 8;
    let xd: LinIn<N, K, S, OP, KL, B2, B> = x
        .view()
        .tile::<m![AudioRows / {B2}], 1, m![AudioRows / {B2} = 1 # {NT}, AudioRows % {B2}, K]>(tile)
        .to_dm(&mut device.tdma);
    let mut acc: LinAcc<N, S, OP, KS, B2, B> = DmTensor::new();
    let w0 = weight32::<N, K, S, OP, KL, B2, B>(device, w, 0);
    dot_into::<N, K, S, OP, KL, KS, B2, B>(device, &xd, &w0, &mut acc, 0);
    let w1 = weight32::<N, K, S, OP, KL, B2, B>(device, w, 1);
    dot_into::<N, K, S, OP, KL, KS, B2, B>(device, &xd, &w1, &mut acc, 1);
    let w2 = weight32::<N, K, S, OP, KL, B2, B>(device, w, 2);
    dot_into::<N, K, S, OP, KL, KS, B2, B>(device, &xd, &w2, &mut acc, 2);
    let w3 = weight32::<N, K, S, OP, KL, B2, B>(device, w, 3);
    dot_into::<N, K, S, OP, KL, KS, B2, B>(device, &xd, &w3, &mut acc, 3);
    acc.view().to_hbm_view(
        &mut device.tdma,
        out.view_mut().tile::<m![AudioRows / {B2}], 1, m![AudioRows / {B2} = 1 #{!} {NT}, AudioRows % {B2}, N]>(tile),
    );
}

// ------------------------------------------------------------------- the kernel

/// Every frame of an `AudioRows`-row tensor in one layout: frame `r` in slice `r % 32`
/// of cluster 0, the frames of a slice in its element axis. The scatter uses it (a whole
/// `[H]` row per element run).
type AllRows<K> = DmTensor<bf16, Chip, Cluster, m![AudioRows % 32, 1 # 8], m![AudioRows / 32, K]>;

const LIN_B: usize = LIN_B2 / 2;
const LIN_TILES: usize = AudioRows::SIZE / LIN_B2;
const _: () = assert!(LIN_B2 == 2 * LIN_B && AudioRows::SIZE % LIN_B2 == 0);

fn audio_rows_impl<R: AxisName>(
    device: &mut Device,
    frames: &HbmTensor<bf16, Chip, m![AudioRows, Aa]>,
    weight: &HbmTensor<bf16, Chip, m![H, Aa]>,
    destination: &HbmTensor<i32, Chip, m![AudioRows]>,
    out: &mut HbmTensor<bf16, Chip, m![R, H]>,
) {
    // Stage A: the weight-free RMS over every 640-wide frame.
    let mut normalized: HbmTensor<bf16, Chip, m![AudioRows, Aa]> = HbmTensor::new();
    for group in 0..GROUPS {
        let x = load_group::<Aa, 80>(device, frames, group);
        let y = rms_norm::<Aa, 80>(device, &x, AA_F32);
        store_group::<Aa, 80>(device, &y, &mut normalized, group);
    }
    // Stage B: the projection into the text hidden size. The weight streams once per tile.
    let mut projected: HbmTensor<bf16, Chip, m![AudioRows, H]> = HbmTensor::new();
    for tile in 0..LIN_TILES {
        linear_tile::<H, Aa, 160, LIN_B2, LIN_B, LIN_TILES>(device, &normalized, weight, &mut projected, tile);
    }
    // Stage C: every frame to its prompt position in one indirect DMA.
    let rows: AllRows<H> = projected.to_dm(&mut device.tdma);
    out.view_mut().scatter::<m![R], m![H]>().by_byte_offsets(destination.view()).from_dm_view(&mut device.tdma, rows.view());
}
