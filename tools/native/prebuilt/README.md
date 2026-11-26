# Rebuild the carried device images

Run from the repository root on CPU. Use Opt 0.9.0, Rust `nightly-2026-05-01`,
Python 3.10+ with `pyelftools` 0.33, GNU AArch64 GCC 11.4.0 and AArch64 binutils
2.38. Put `cargo-furiosa-opt`, `furiosa-opt-driver` and `aarch64-linux-gnu-gcc`
on `PATH`. The driver SHA-256 is
`1363b347c2360d66d20d34ee3fcf2426d726defcefad311ec3c85f910e55fa24`.
`toolchain.py` extracts the relink headers and `libpe.a` from that driver.

Before going offline, download the x86-64 Linux assets below and export each
variable as the absolute path to its archive. [Opt v0.9.0 release](https://github.com/furiosa-ai/furiosa-opt/releases/tag/v0.9.0)

| Variable | Release asset | SHA-256 |
| --- | --- | --- |
| `FURIOSA_OPT_LOWER_IMPL_LOCAL_PREBUILT` | `libfuriosa_opt_lower_impl-v0.9.0-x86_64-unknown-linux-gnu.a` | `d41bee1980ad899c1400d3010ae04c367c1ba59e17d1437bb5ca1cd8ca88f505` |
| `FURIOSA_MAPPING_IMPL_LOCAL_PREBUILT` | `libfuriosa_mapping_impl-v0.9.0-x86_64-unknown-linux-gnu.a` | `6b5362617c08e7dc0da8754e45d1948f3abe5da25c5fb5acc7e4ef3bf706b3cc` |

```sh
cargo fetch --locked
(cd tools/native/fopt && cargo fetch --locked)
export CARGO_NET_OFFLINE=true
export SHARDS=8 KNOB_JOBS=4 CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=1
mkdir -p .build
OUT=$(mktemp -d "$PWD/.build/reproduction.XXXXXX")
nice -n 10 tools/native/prebuilt/rebuild.sh "$OUT"
```

The route writes two containers, the manifest, task-C digests and five Rust
tables under `$OUT/packed/host/`. The compiler stages use private source copies
with empty payloads and compile-only tables. The reference server is linked
on CPU. Pass a stage as the second argument to resume in the same output
directory.

| Stage | Work |
| --- | --- |
| `reference` | Stock images and task-C captures. |
| `knobs` | Per-kernel compiler builds and captures. |
| `step` | Standalone attention members for the two dispatchers. |
| `derive` | Apply task-C rewrites, link final standalone and STEP images, generate ABI tables and digests. |
| `pack` | Pack the final images and write the manifest. |

On 16 CPUs the route took 30 minutes (stages 12, 12, 2.7, 3.6 and 0.1 minutes)
and `rebuild.sh "$OUT" compare` reported every output identical to the
installed files.

## Compiler settings

Each kernel directory under `src/host/prebuilt/knobs/` can supply
`ordering.txt`, with one `T73 -> T902` edge per line. This asks Opt to list
the producer of `%T73` before the producer of `%T902`. The numbers identify
LIR output tensors in creation order in that kernel's graph dump. Graph edits
can renumber them. The edges guide list scheduling alongside data dependencies
and resource availability. A compiled schedule can violate a hint.

For a knob build, `build.sh` exports the ordering file through
`SCHEDULER_MANUAL_ORDERING_PATH` and the kernel's config or the common config
through `NPU_COMPILER_CONFIG_PATH`, evicts the prior Opt cache entry, and
compiles that kernel with `cargo furiosa-opt compile --exact`.
[Build script](build.sh#L59-L86)

**Q4 G12.** Gather-before-scatter edges preserve the old K/V snapshot and let
history DMA overlap projection and FFN. The 1,024-row RSS and RFS have 42 and
24 edges, respectively. The 1,536-row pair adds five FFN exchange and Down
edges to 36 K/V edges per member.

The common `config.yaml` disables TUC profiling commands and leaves Task
timing. This preserved the tested schedule while cutting image size by 19%
at 1,024 rows. [Common config](../../../src/host/prebuilt/config.yaml)

## Rebuild inputs and outputs

The primary payload stores the final standalone image for each row. STEP
reuses the post-passed member tasks where applicable. [Task-C rewrites](../../../docs/COMPILER.md#build-scheduling-and-launch)

`layout.py`, `rider_abi.py` and `dispatch_build.py` generate and link the two
STEP chunks. They compact frames, alias common operands and reuse aligned
weight pages from `last_bf16_out128` plus a generated 32×32 mask. Constant
pooling and GCC LTO keep each linked task under 1 MiB.

`src/host/prebuilt.rs` declares the carried rows in payload order and their
reference-build modes. The manifest records the resulting images and compiler
settings. `tools/native/step/step_image_table.rs.in` binds carried member names
and standalone device paths to opcodes and records dispatcher slot capacities.
The shard TSVs record each build's registry and partitioning estimates.
`derive.py` reads the member ABI from the rebuilt images.

## Verification

Compare a completed build against a checkout with the shipped files. The
report gives `exact` or `differs` for each file and each container row and
exits nonzero for any difference or missing file.

```sh
tools/native/prebuilt/rebuild.sh "$OUT" compare
python3 -B tools/native/prebuilt/source_record.py check
```

To install generated files and record intentional input or output changes, run
from the checkout root.

```sh
cp -a "$OUT/packed/host/." src/host/
python3 -B tools/native/prebuilt/source_record.py write
```

Build the server with a fresh target and Opt cache.

```sh
mkdir -p .build
SERVER_OUT=$(mktemp -d "$PWD/.build/server.XXXXXX")
export CARGO_TARGET_DIR="$SERVER_OUT/target" FURIOSA_OPT_OUT_DIR="$SERVER_OUT/opt"
nice -n 10 tools/native/shard_build.py --shards 8 -- --release --locked --bin server
sha256sum "$CARGO_TARGET_DIR/release/server"
```

The host ELF embeds absolute Cargo registry paths. Compare the compiled
kernels across hosts by hashing their ELF section. The source record checks
the carried payloads.

```sh
objcopy -O binary --only-section=furiosa_kernels "${CARGO_TARGET_DIR:-target}/release/server" /dev/stdout | sha256sum
```
