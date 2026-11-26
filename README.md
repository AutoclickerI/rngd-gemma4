# Gemma-4-12B-it on one RNGD at 1,000 tok/s

Rust serving engine for one FuriosaAI RNGD, built with Opt 0.9.0. The contest grader measured 988.69 tok/s on its own inputs.
`src/api/`, `scripts/stage2/`, `public-test/`, `Cargo.lock`, `rust-toolchain.toml` and `Cargo.toml` come from the
MOA 2026 Stage 2 baseline [`micro2026-moa/furiosa-opt-gemma4-12B` at `73828c722d0b`](https://github.com/micro2026-moa/furiosa-opt-gemma4-12B/tree/73828c722d0b).
`Cargo.toml` adds the `compile-carried` feature.

## Results

The public stream cycles through 256, 1,024 and 2,048 text tokens, 512 with image, and 768 with audio. Concurrency 100, 64 generated tokens/request, 100 warm-ups, 1,500 measured requests. Gates are p90 TTFT ≤ 4 s and TPOT ≤ 100 ms, as `scripts/stage2/performance.py` computes them.

| Card | Firmware | tok/s | TTFT p90 | TPOT p90 | Mean GHz | Cap share | tok/s at 2.0 GHz |
|---|---|---:|---:|---:|---:|---:|---:|
| A | 2026.4.0 | 1,053 | 2.03 s | 65.5 ms | 1.993 | 6% | 1,057 |
| B | 2026.4.1 | 1,009 | 2.02 s | 70.2 ms | 1.917 | 85% | 1,053 |
| C | 2026.4.1 | 963 | 2.02 s | 75.6 ms | 1.825 | 95% | 1,055 |

B and C ran this tree on November 24, 2026, one card loaded at a time. A ran the November 23 build with additional carried variants. The day's ten B and nine C runs of builds within 0.7 % of each other per clock ranged 1,008–1,023 and 963–1,007 tok/s at mean clocks of 1.92–1.95 and 1.83–1.90 GHz, and 1,051–1,058 tok/s at 2.0 GHz on both cards. All texts equal `tools/reference_outputs.json`: one per stream input type over 1,500 requests and one per public accuracy case over 342 requests (three runs of 19 rounds), all six cases passing. [Clock scaling and throttle samples](docs/HARDWARE.md#power-and-clock)

## Build and run

Run from the repo root with an RNGD and driver, Rust `nightly-2026-05-01`, `cargo-binstall`, Python ≥ 3.10 and Hugging Face `hf`.

```sh
cargo binstall -y cargo-furiosa-opt@0.9.0
hf download unsloth/gemma-4-12b-it-NVFP4 --revision b1f649734b34aa5575b03d186abd1b9be3d0d5c4 --local-dir MODEL
export CARGO_TARGET_DIR=$PWD/target FURIOSA_OPT_OUT_DIR=$PWD/target/opt
tools/native/shard_build.py --shards 8 -- --release --locked --bin server
export RNGD_MODEL_DIR=MODEL
tools/run_stream.py --accuracy --rounds 19 -c 100 target/release/server
tools/run_stream.py target/release/server
```

The default build compiles 99 kernels in 4.2 minutes cold on 16 CPUs. The time columns in `tools/native/shards/*.tsv` estimate compile seconds for shard balancing. For a serial build use `cargo furiosa-opt build --release --locked --bin server`. `tools/run_stream.py --chip N` selects the card through `FURIOSA_VISIBLE_CHIPS`.

Run `tools/compare_outputs.py RUN_DIR` on the reported run directory to check every measured text against the reference. Generate it with `tools/compare_outputs.py --write tools/reference_outputs.json ACCURACY_RUN PERFORMANCE_RUN [PERFORMANCE_RUN ...]`. Runs must agree within each replay mode. The file records each run's plan and server SHA-256.

The OpenAI chat-completions server uses model ID `gemma-4-12b-it` and defaults to `GEMMA4_API_ADDR=0.0.0.0:8000`.

```sh
RNGD_MODEL_DIR=$PWD/MODEL GEMMA4_API_ADDR=127.0.0.1:8000 target/release/server &
curl -fsS --retry 300 --retry-connrefused --retry-delay 1 http://127.0.0.1:8000/v1/models
```

## Differences from the reference

Sliding attention uses a 384-position window (1,024 in the reference). Context is capped at 4,096 positions. Gate and Up use INT4 on 39 of 48 layers. Most other projections use FP8 (E4M3) or INT8, and KV storage uses E4M3. [Mask and numerical differences](docs/DESIGN.md#differences-from-the-reference-model)

Streaming holds early events until 2 s after engine arrival while device work continues, shifting measured TPOT into TTFT. With `FIRST_CHUNK_HOLD` set to `Duration::ZERO` in `src/host/arrival.rs`, B and C passed both gates (TTFT p90 0.57–0.59 s, TPOT p90 93–98 ms) and returned the same texts.

## Source and tooling

- [Design and source entry points](docs/DESIGN.md) cover scheduling, device kernels, calibration and the embedded images.
- [Calibration](docs/CALIBRATION.md) fetches the pinned data, captures the statistics and regenerates the shipped tables.
- [Hardware measurements](docs/HARDWARE.md) and [Opt 0.9.0 rules](docs/COMPILER.md) cover performance and lowering.
- [Carried image recipes](tools/native/prebuilt/README.md) use `compile-carried` and generate the embedded images, containers and tables byte for byte from source with the pinned Opt driver.
