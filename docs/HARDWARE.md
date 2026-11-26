# RNGD measurements

Gemma-4-12B-it on Opt 0.9.0, firmware 2026.3.0, 2026.4.0 and 2026.4.1.

## Kernel resources and timing

- Two clusters, four PEs/cluster, 64 slices/PE, eight lanes/slice. Each PE has two DMNs (32 slices each), one DMA engine and Main/Sub contexts. DM spans the configured device. Main-issued `.to_trf()` occupies both contexts. Contractions compete with Main vector work. `furiosa-opt-abi-0.9.0/src/lib.rs:43-46`, `furiosa-opt-std-0.9.0/src/constraints.rs:21-35`.
- Each slice has 512 KiB DM (16 × 32 KiB pages), 65,536 B logical TRF (80 KiB physical), 8,192 B VRF. TRF is 8 lanes × 2 banks × 128 rows × 32 B. SPM is 4 KiB/PE. A 16 × 3,840 INT8 tile uses 61,440 B TRF in one pass. A 7,680 B VRF operand failed with a 1 KiB stash. Load rolled norm TRF tensors inside each pass to limit their lifetime. `furiosa-opt-abi-0.9.0/src/dm.rs:9-13`, `furiosa-opt-std-0.9.0/src/engine/collect.rs:144-146`, `furiosa-opt-lower-0.9.0/src/verify/mod.rs:101-105`, `src/device/tokenwise_int8/gemm.rs#WTrf`, `src/device/proj_fast/rotq.rs#norm_rotate_quantize_pass`.
- At the 1 GHz reference, GEMM reached 30.7/32 BF16 and 56.5/64 E4M3 MACs/lane/cycle. INT8 and one-flit i4 ran in the same range as E4M3. Two-flit i4 × i4 reached 102–110 MACs/lane/cycle (1.85–1.89× INT8 across four operand families, reduction shards of 1, 4 and 15 packets). FP8 saves passes. Integer accumulators are i32, floating-point f32. `furiosa-opt-std-0.9.0/src/cast.rs:343-350`.
- In 512-token kernels, measured/static time was ~2× with a second full-length vector operand, 0.9–1.0× with scalar/broadcast, 1.00–1.02× for GEMM. An immediate stash read cost 2.2×. Separate stash writes and reads by two vector stages.
- Two 1 MiB code slots per cluster allow chunk prefetch. Boundaries cost 17.8 µs. Single-chunk images compete for slot 0. The device caches one image by staging address, length and token. `furiosa-opt-abi-0.9.0/src/image.rs:6,18`.
- An alternating-function round trip costs 7.9 µs + 153 µs/staged MB after subtracting two relaunches. Restaging the same bytes cost 28.2 µs. Same-function medians were 4.5/7.5 µs at 512/1,024 rows. Queued switches between byte-identical 16,584 B chunks in distinct `Function` handles added 8.99 µs/switch at 2.0 GHz. Reuse the loaded handle.
- Internal RSS Task seams cost 1.38, 1.51 and 1.21 µs/boundary across three sets at 2.0 GHz, excluding time outside Task markers. Trace deltas are divided by two for normalized cycles. The Opt profile uses the 1 GHz reference.
- Each runtime link allows 16 in-flight requests, blocking on completion at capacity. Initial polls spin up to 200 µs. Launches take up to 119 arguments. Writes ≤ 2 KiB use BAR (15–31 µs/call). Larger writes and all reads use DMA (~12.5 µs/call for reads ≤ 4 KB). `control_tu_clock(NULL,0)` still queues a firmware boundary. `furiosa-opt-rt-0.9.0/src/device/engine/link.rs:52-54`, `furiosa-opt-rt-0.9.0/src/device/engine/mod.rs:41-48`, `furiosa-opt-ipc-0.9.0/src/frame.rs:19`.
- Permuting FP8 operands across packets changed results, ruling out the tested nearest-even and truncation models. SFU `Exp` was +1 ULP on 2,421 of 8,320 inputs, exact elsewhere against correctly rounded f32 with flush-to-zero. `DivF` uses `FpFpu`, `vector_fp_div` uses `ReduceFpDiv`. Both IEEE and all-RTZ CPU models missed the 4-bit FFN Down producer. Use stage-specific oracles for bitwise checks. `furiosa-opt-std-0.9.0/src/engine/vector/op/mod.rs:236,361`.

- With DMA pending, 64 consecutive accesses to one DM bank can exceed the 4,096-cycle NoC acknowledgement window and reset the cluster.

## Host scheduling

- The run-delay gate permits early preparation between launches. It divides the worker's runnable wait by wall time over steps without cgroup CPU throttling. Quiet-host completion waves reached 1.0–1.5% over 32 steps, and the worker's own contention with preparation off read 0.55–0.8%. Shared hosts read 2–14%. The gate turns preparation off at 2% over 32 steps and back on below 1% over 16 steps for earlier recovery. It extends the window to collect eight unthrottled steps. At 128 steps with fewer than eight it treats the run as quota-bound and enables preparation. The base window, eight-step minimum and four-window limit were set by hand and not tuned. `src/host/engine/queue.rs#gate_decide`
- The worker spins for 50 µs on a pending device wait and then yields while remaining runnable. Parking behind the HTTP threads on two CPUs gave 0.9–15 ms for the same token readback. The launch helper uses 1 µs timer slack to limit added wake-up delay and sleeps 10 µs on passes with no new submission and at least four launches pending. A filled queue can make progress during the sleep. The short tail, or a failed timer-slack request, uses yields. The exact spin, slack, sleep and pending-count values were set by hand without a recorded tuning sweep. `src/host/worker.rs#spin_block_on`, `src/host/engine/launch_thread.rs#worker`
- Request preprocessing uses CPUs − 3 permits clamped to 1–4, reserving CPU time for the worker, runtime DMA and SQPOLL. The clamp was set by hand and not tuned. Layer-0 input prep uses CPUs − 1 lanes, clamped to 1–7, with one chunk under preparation per lane. Q4 startup uses up to 16 workers to reconstruct and pack rows. `src/host/preprocess.rs#slots`, `src/host/engine/l0table.rs#MAX_LANES`, `src/host/engine/int8/q4serve/packed.rs#direct`

## Power and clock

November 21–23, 2026. Sysfs `power1_max` was 300 W. Cards A (rev 01, firmware 2026.4.0) and B/C (rev 04, 2026.4.1) ran on separate hosts by revision. Board, firmware and host effects are confounded. Loaded samples are ≥150 W at 10 Hz.

- Back-to-back 1,024-token kernels at ~64% tensor duty ran at 2.0 GHz and 293–298 W on one card. Another ran at 1.7 GHz and 253–256 W with `0x4` (15% longer/launch). Hot serving sampled 1.0–2.0 GHz.
- At 2.0 GHz, A rose from 285 W at 55 °C to 295 W at 66 °C. On the wide step, A first flagged at 50/76 s and 283/288 W, with one unflagged run reaching 293 W. B/C flagged within 2 s at 150–213 W in five of six runs, otherwise at 3.4 s and 261 W. `power1_average` (`RMS_TOTAL`) did not predict the flag.
- With the peer idle, B/C gave 1,016–1,020 / 983–991 tok/s at 1.92 / 1.85–1.86 GHz from 35–43 °C. With both loaded they gave 1,016–1,025 / 969–994. C's lowest scores came on the second consecutive loaded leg.
- `0x4` (`THROTTLE_APP_POWER_CAP`) drops clocks to 1.9 then 1.7 GHz. Scale cap-only runs by 2.0 GHz / mean clock. A at 36/60/66 °C start gave 1,041/1,013/1,010 tok/s with 14/74/80% cap share, within 0.4% after scaling.
- Above ~80 °C, `0x2` (`THROTTLE_THERMAL_SLOWDOWN`) brings some clusters to 1.7 then 1.4 GHz. At 82–83 °C, runs gave 865–902 tok/s with the thermal flag in 56–65% of loaded samples. Sustained unequal cluster clocks invalidate mean-clock scaling; cap-only legs had at most five unequal samples in about 1,000, all at transitions. Flag names follow driver 2026.4.0 (`npu_mgmt_ncdev.c`).

## Step width under the power cap

November 23, 2026, on the prior serving route, with B and C loaded together. Both widths returned the same 1,500 texts. Energy/row is the sum of loaded power samples × mean sample interval / 1,575,360 rows (1,600 requests including warm-ups). [Stream definition](../README.md#results)

| Card | Rows | Start °C | tok/s | GHz | Cap share (0x4) | Median W | mJ/row | tok/s at 2.0 GHz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 1,536 | 35 | 1,048 | 1.986 | 15% | 282 | 17.2 | 1,055 |
| A | 1,024 | 36 | 1,036 | 1.973 | 27% | 289 | 17.7 | 1,050 |
| B | 1,536 | 39 | 1,025 | 1.936 | 64% | 286 | 18.0 | 1,059 |
| B | 1,536 | 43 | 1,019 | 1.922 | 77% | 287 | 18.1 | 1,061 |
| B | 1,024 | 43 | 982 | 1.860 | 90% | 284 | 18.6 | 1,056 |
| C | 1,536 | 37 | 994 | 1.874 | 86% | 285 | 18.3 | 1,061 |
| C | 1,536 | 43 | 969 | 1.830 | 92% | 279 | 18.5 | 1,060 |
| C | 1,024 | 43 | 937 | 1.780 | 89% | 271 | 18.8 | 1,054 |

At matched or near-matched starts, the wide step gained 1.2/3.8/3.4% throughput on A/B/C (0.5–0.6% clock-scaled), drew 2.9/2.9/1.8% less energy per row and sustained 0.7/3.3/2.8% higher clocks.
