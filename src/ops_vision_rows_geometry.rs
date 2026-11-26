// Exact row arithmetic, in two contraction tiles; VisionRows is supplied by the parent module.
const GROUPS: usize = VisionRows::SIZE / 32;
const RV_F32: f32 = Rv::SIZE as f32;
const MV_F32: f32 = Mv::SIZE as f32;
const _: () = assert!(VisionRows::SIZE % 32 == 0 && PvRows::SIZE == 2 * Pv::SIZE);

// ------------------------------------------------------------------ row groups

/// 32 rows x 8 shards; each slice holds `KL` consecutive elements of one row.
type RowShards<K, const KL: usize> = m![VisionRows % 32, K / {KL}];
/// After an inter-slice reduction every shard of a row holds the row's value.
type RowReduced = m![VisionRows % 32, Dummy8];
type Group<K, const KL: usize> = DmTensor<bf16, Chip, Cluster, RowShards<K, KL>, m![K % {KL}]>;
type GroupF32<K, const KL: usize> = DmTensor<f32, Chip, Cluster, RowShards<K, KL>, m![K % {KL}]>;
type Stat<K, const KL: usize> = DmTensor<f32, Chip, Cluster, RowShards<K, KL>, m![1 # 8]>;
type StatReduced = DmTensor<f32, Chip, Cluster, RowReduced, m![1 # 8]>;
type StatVrf<K, const KL: usize> = VrfTensor<f32, Chip, Cluster, RowShards<K, KL>, m![1 # 8]>;
type RowVrf<K, const KL: usize> = VrfTensor<f32, Chip, Cluster, RowShards<K, KL>, m![K % {KL}]>;

/// A `[K]` vector replicated into every row's shards (the row axis is absent from the
/// source, so the DMA replicates it).
fn replicated<K: AxisName, const KL: usize>(device: &mut Device, v: &HbmTensor<bf16, Chip, m![K]>) -> Group<K, KL> {
    v.to_dm(&mut device.tdma)
}

fn row_vrf<K: AxisName, const KL: usize>(device: &mut Device, x: &Group<K, KL>) -> RowVrf<K, KL> {
    device
        .sub
        .begin(x.view())
        .fetch::<m![K % {KL} / 16], m![K % {KL} % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![K % {KL} / 8], m![K % {KL} % 8]>()
        .to_vrf()
}

fn stat_vrf<K: AxisName, const KL: usize>(device: &mut Device, x: StatReduced) -> StatVrf<K, KL> {
    // Each shard receives its own row's completed reduction.
    let x: Stat<K, KL> = unsafe { x.reshape() };
    device
        .sub
        .begin(x.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}

/// Per-shard sum of the row's elements.
fn partial_sum<K: AxisName, const KL: usize>(device: &mut Device, x: &Group<K, KL>) -> Stat<K, KL> {
    device
        .main
        .begin(x.view())
        .fetch::<m![K % {KL} / 16], m![K % {KL} % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![K % {KL} / 8], m![K % {KL} % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![K % {KL} / 4], m![K % {KL} % 4]>()
        .vector_intra_slice_reduce::<K, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

/// Per-shard sum of squares of an f32 row (the centred row, or the raw row for RMS).
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

/// Sum of the eight shard partials of each row.
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

/// `sum / size` (the mean, or the mean square).
fn scale_stat(device: &mut Device, x: &StatReduced, size: f32) -> StatReduced {
    device
        .main
        .begin(x.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_div(size)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

/// `sqrt(x + EPS)`: the standard deviation of a centred row or the RMS of a raw row.
fn sqrt_eps(device: &mut Device, x: &StatReduced) -> StatReduced {
    let shifted: StatReduced = device
        .main
        .begin(x.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
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

/// `x - mean` in f32.
fn centre<K: AxisName, const KL: usize>(device: &mut Device, x: &Group<K, KL>, mean: &StatVrf<K, KL>) -> GroupF32<K, KL> {
    device
        .main
        .begin(x.view())
        .fetch::<m![K % {KL} / 16], m![K % {KL} % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![K % {KL} / 8], m![K % {KL} % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![K % {KL} / 4], m![K % {KL} % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, mean)
        .vector_widen_concat::<m![K % {KL} / 8], m![K % {KL} % 8]>()
        .vector_final()
        .commit_trim::<m![K % {KL} % 8]>()
        .commit()
}

/// LayerNorm of every row of a group: `(x - mean) * weight / std + bias`, same op order
/// as `vision::layernorm::normalize`.
fn layer_norm<K: AxisName, const KL: usize>(
    device: &mut Device,
    x: &Group<K, KL>,
    weight: &Group<K, KL>,
    bias: &Group<K, KL>,
    size: f32,
) -> Group<K, KL> {
    let sum = partial_sum::<K, KL>(device, x);
    let sum = reduce_rows::<K, KL>(device, &sum);
    let mean = scale_stat(device, &sum, size);
    let mean = stat_vrf::<K, KL>(device, mean);
    let centred = centre::<K, KL>(device, x, &mean);
    let squares = partial_sum_squares::<K, KL>(device, &centred);
    let squares = reduce_rows::<K, KL>(device, &squares);
    let variance = scale_stat(device, &squares, size);
    let std = sqrt_eps(device, &variance);
    let std = stat_vrf::<K, KL>(device, std);
    let weight = row_vrf::<K, KL>(device, weight);
    let bias = row_vrf::<K, KL>(device, bias);
    device
        .main
        .begin(centred.view())
        .fetch::<m![K % {KL} / 8], m![K % {KL} % 8]>()
        .collect::<m![K % {KL} / 8], m![K % {KL} % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![K % {KL} / 4], m![K % {KL} % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight)
        .vector_fp_div(&std)
        .vector_widen_concat::<m![K % {KL} / 8], m![K % {KL} % 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &bias)
        .vector_final()
        .cast::<bf16, m![K % {KL} % 8 # 16]>()
        .commit_trim::<m![K % {KL} % 8]>()
        .commit()
}

/// Weight-free RMS normalization of every row: `x / sqrt(mean(x^2) + EPS)`.
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
    let mean_square = scale_stat(device, &squares, size);
    let rms = sqrt_eps(device, &mean_square);
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

/// `bf16(x + rhs)` per row (the dense bias, or the position embedding).
fn add_rows<K: AxisName, const KL: usize>(device: &mut Device, x: &Group<K, KL>, rhs: &Group<K, KL>) -> Group<K, KL> {
    let rhs = row_vrf::<K, KL>(device, rhs);
    device
        .main
        .begin(x.view())
        .fetch::<m![K % {KL} / 16], m![K % {KL} % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![K % {KL} / 8], m![K % {KL} % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &rhs)
        .vector_final()
        .cast::<bf16, m![K % {KL} % 8 # 16]>()
        .commit_trim::<m![K % {KL} % 8]>()
        .commit()
}

fn load_group<K: AxisName, const KL: usize>(device: &mut Device, x: &HbmTensor<bf16, Chip, m![VisionRows, K]>, _group: usize) -> Group<K, KL> {
    x.view()
        .tile::<m![VisionRows / 32], 1, m![VisionRows / 32 = 1 # {GROUPS}, VisionRows % 32, K]>(self::row_group_index!(_group))
        .to_dm(&mut device.tdma)
}

fn store_group<K: AxisName, const KL: usize>(device: &mut Device, x: &Group<K, KL>, out: &mut HbmTensor<bf16, Chip, m![VisionRows, K]>, _group: usize) {
    x.view().to_hbm_view(
        &mut device.tdma,
        out.view_mut()
            .tile::<m![VisionRows / 32], 1, m![VisionRows / 32 = 1 #{!} {GROUPS}, VisionRows % 32, K]>(self::row_group_index!(_group)),
    );
}

// ------------------------------------------------------------- linear stages

/// Cluster `c` holds rows `[t*B2 + c*B, t*B2 + c*B + B)` of tile `t` (`B2 = 2*B`).
type LinCluster<const B2: usize, const B: usize> = m![VisionRows % {B2} / {B}];
/// Owner x K-shard slices: owner `o` holds output rows `[o*S, o*S + S)`, shard `k`
/// columns `[k*KL, k*KL + KL)`.
type LinRC<N, K, const S: usize, const OP: usize, const KL: usize> = m![N / {S} # {OP}, K / {KL}];
type LinRR<N, const S: usize, const OP: usize, const KS: usize> = m![N / {S} # {OP}, 1 # {KS}];
type LinW<N, K, const S: usize, const OP: usize, const KL: usize, const B2: usize, const B: usize> =
    TrfTensor<bf16, Chip, LinCluster<B2, B>, LinRC<N, K, S, OP, KL>, m![N % {S} = 32 % 8], m![N % {S} = 32 / 8, K % {KL}]>;
type LinIn<N, K, const S: usize, const OP: usize, const KL: usize, const B2: usize, const B: usize> =
    DmTensor<bf16, Chip, LinCluster<B2, B>, LinRC<N, K, S, OP, KL>, m![VisionRows % {B2} % {B}, K % {KL}]>;
type LinAcc<N, const S: usize, const OP: usize, const KS: usize, const B2: usize, const B: usize> =
    DmTensor<bf16, Chip, LinCluster<B2, B>, LinRR<N, S, OP, KS>, m![VisionRows % {B2} % {B}, N % {S}]>;

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
        .fetch::<m![VisionRows % {B2} % {B}, K % {KL} / 16], m![K % {KL} % 16]>()
        .collect::<m![VisionRows % {B2} % {B}, K % {KL} / 16], m![K % {KL} % 16]>()
        .contract_outer::<m![VisionRows % {B2} % {B}, K % {KL} / 32, N % {S} = 32 / 8], m![K % {KL} % 32], _, _, _>(w)
        .contract_packet::<m![1]>()
        .contract_time::<m![VisionRows % {B2} % {B}, N % {S} = 32 / 8]>()
        .contract_lane::<m![VisionRows % {B2} % {B}, N % {S} = 32 / 8], m![N % {S} = 32 % 8 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<LinRR<N, S, OP, KS>, m![VisionRows % {B2} % {B}, N % {S} = 32 / 8]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![N % {S} = 32 % 8 # 16]>()
        .commit_trim::<m![N % {S} = 32 % 8]>()
        .commit_view(acc.view_mut().tile::<m![N % {S}], 32, m![VisionRows % {B2} % {B}, N % {S} = 32 #{!} {S}]>(panel * 32));
}

/// `out[tile rows] = x[tile rows] * w^T` for one B2-row tile of NT; S = 128 owner rows
/// (four 32-row panels per owner), the panel list spelled out so the scheduler overlaps
/// the next panel's weight load with the current contraction. The tile views split
/// `VisionRows` as `/ B2` and `% B2` (a `VisionRows = B2 # VisionRows` window is not
/// accepted by the SDK's DMA layout check on a cluster split).
fn linear_tile<N: AxisName, K: AxisName, const KL: usize, const B2: usize, const B: usize, const NT: usize>(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![VisionRows, K]>,
    w: &HbmTensor<bf16, Chip, m![N, K]>,
    out: &mut HbmTensor<bf16, Chip, m![VisionRows, N]>,
    tile: usize,
) {
    const S: usize = 128;
    const OP: usize = 32;
    const KS: usize = 8;
    let xd: LinIn<N, K, S, OP, KL, B2, B> = x
        .view()
        .tile::<m![VisionRows / {B2}], 1, m![VisionRows / {B2} = 1 # {NT}, VisionRows % {B2}, K]>(tile)
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
        out.view_mut().tile::<m![VisionRows / {B2}], 1, m![VisionRows / {B2} = 1 #{!} {NT}, VisionRows % {B2}, N]>(tile),
    );
}

// ------------------------------------------------------------------- the kernel

/// Every row of a `VisionRows`-row tensor in one layout: row `r` in slice `r % 32` of
/// cluster 0, the nine rows of a slice in its element axis. Indirect gathers and the
/// scatter use it (a whole `[Mv]`/`[H]` row per element run).
type AllRows<K> = DmTensor<bf16, Chip, Cluster, m![VisionRows % 32, 1 # 8], m![VisionRows / 32, K]>;

const LIN_B: usize = LIN_B2 / 2;
const LIN_TILES: usize = VisionRows::SIZE / LIN_B2;
const _: () = assert!(LIN_B2 == 2 * LIN_B && VisionRows::SIZE % LIN_B2 == 0);

#[allow(clippy::too_many_arguments)]
fn vision_rows_impl<R: AxisName>(
    device: &mut Device,
    pixels: &HbmTensor<Pixel, Chip, m![VisionRows, Rv]>,
    pos_table: &HbmTensor<bf16, Chip, m![PvRows, Mv]>,
    col_offsets: &HbmTensor<i32, Chip, m![VisionRows]>,
    row_offsets: &HbmTensor<i32, Chip, m![VisionRows]>,
    ln1_weight: &HbmTensor<bf16, Chip, m![Rv]>,
    ln1_bias: &HbmTensor<bf16, Chip, m![Rv]>,
    dense_weight: &HbmTensor<bf16, Chip, m![Mv, Rv]>,
    dense_bias: &HbmTensor<bf16, Chip, m![Mv]>,
    ln2_weight: &HbmTensor<bf16, Chip, m![Mv]>,
    ln2_bias: &HbmTensor<bf16, Chip, m![Mv]>,
    ln3_weight: &HbmTensor<bf16, Chip, m![Mv]>,
    ln3_bias: &HbmTensor<bf16, Chip, m![Mv]>,
    projection: &HbmTensor<bf16, Chip, m![H, Ov]>,
    destination: &HbmTensor<i32, Chip, m![VisionRows]>,
    out: &mut HbmTensor<bf16, Chip, m![R, H]>,
) {
    // Stage 0: the two position-embedding rows of every patch, gathered by byte offset.
    let col: AllRows<Mv> = pos_table.gather::<m![PvRows], m![Mv]>().by_byte_offsets(col_offsets).to_dm(&mut device.tdma);
    let col_rows: HbmTensor<bf16, Chip, m![VisionRows, Mv]> = col.to_hbm(&mut device.tdma);
    let row: AllRows<Mv> = pos_table.gather::<m![PvRows], m![Mv]>().by_byte_offsets(row_offsets).to_dm(&mut device.tdma);
    let row_rows: HbmTensor<bf16, Chip, m![VisionRows, Mv]> = row.to_hbm(&mut device.tdma);
    // Stage A: LN1 over every 6912-wide patch row.
    let mut normed1: HbmTensor<bf16, Chip, m![VisionRows, Rv]> = HbmTensor::new();
    {
        let w = replicated::<Rv, 864>(device, ln1_weight);
        let b = replicated::<Rv, 864>(device, ln1_bias);
        for group in 0..GROUPS {
            let x = load_pixels(device, pixels, group);
            let y = layer_norm::<Rv, 864>(device, &x, &w, &b, RV_F32);
            store_group::<Rv, 864>(device, &y, &mut normed1, group);
        }
    }
    // Stage B: the dense patch projection.
    let mut dense: HbmTensor<bf16, Chip, m![VisionRows, Mv]> = HbmTensor::new();
    for tile in 0..LIN_TILES {
        linear_tile::<Mv, Rv, 864, LIN_B2, LIN_B, LIN_TILES>(device, &normed1, dense_weight, &mut dense, tile);
    }
    // Stage C: bias, LN2, position embedding, LN3 and the weight-free RMS per row.
    let mut normed3: HbmTensor<bf16, Chip, m![VisionRows, Mv]> = HbmTensor::new();
    {
        let bias = replicated::<Mv, 480>(device, dense_bias);
        let w2 = replicated::<Mv, 480>(device, ln2_weight);
        let b2 = replicated::<Mv, 480>(device, ln2_bias);
        let w3 = replicated::<Mv, 480>(device, ln3_weight);
        let b3 = replicated::<Mv, 480>(device, ln3_bias);
        for group in 0..GROUPS {
            let x = load_group::<Mv, 480>(device, &dense, group);
            let x = add_rows::<Mv, 480>(device, &x, &bias);
            let x = layer_norm::<Mv, 480>(device, &x, &w2, &b2, MV_F32);
            let c = load_group::<Mv, 480>(device, &col_rows, group);
            let r = load_group::<Mv, 480>(device, &row_rows, group);
            let pos = add_rows::<Mv, 480>(device, &c, &r);
            let x = add_rows::<Mv, 480>(device, &x, &pos);
            let x = layer_norm::<Mv, 480>(device, &x, &w3, &b3, MV_F32);
            let x = rms_norm::<Mv, 480>(device, &x, MV_F32);
            store_group::<Mv, 480>(device, &x, &mut normed3, group);
        }
    }
    // Stage D: the projection into the text hidden size. Mv and Ov are the same width;
    // the reshape renames the axis of the same BF16 words.
    let normed3: HbmTensor<bf16, Chip, m![VisionRows, Ov]> = unsafe { normed3.reshape() };
    let mut projected: HbmTensor<bf16, Chip, m![VisionRows, H]> = HbmTensor::new();
    for tile in 0..LIN_TILES {
        linear_tile::<H, Ov, 480, LIN_B2, LIN_B, LIN_TILES>(device, &normed3, projection, &mut projected, tile);
    }
    // Stage E: every row to its prompt position in one indirect DMA.
    let rows: AllRows<H> = projected.to_dm(&mut device.tdma);
    out.view_mut().scatter::<m![R], m![H]>().by_byte_offsets(destination.view()).from_dm_view(&mut device.tdma, rows.view());
}
