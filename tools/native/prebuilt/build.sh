#!/bin/bash
# Compile the reference server, or one named kernel with its recorded knobs.
# CAPTURE_KERNELS lists kernel paths whose task C must be captured.
set -euo pipefail
OUT=$1
KNOBS=${2:-}
FN=${3:-}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$ROOT"
if [ -n "$KNOBS" ]; then
  [ -n "$FN" ] || { echo "a knob build names the one kernel it compiles (<fn path>)"; exit 1; }
  FN=furiosa_opt_gemma4::${FN#furiosa_opt_gemma4::}
  # Besides the kernel's own entry, a warm knob cache holds the provenance record and the axis record that the end of
  # every knob build writes (opt_cache_rekey.py --axes-record below); neither is another kernel's image.
  other=$(find "$OUT" -type f 2>/dev/null | grep -v -e "/furiosa-opt-gemma4/$FN\.\(bin\|hash\|edf\)\$" -e '/\.provenance\.json$' \
    -e '/\.axes-set\.json$' -e '/\.axes-sizes\.json$' || true)
  [ -z "$other" ] || { echo "a knob build needs a cache empty but for $FN's own entry: $OUT holds $(echo "$other" | head -3)"; exit 1; }
fi
mkdir -p "$OUT"
DRIVER=$(command -v furiosa-opt-driver)
echo "driver: $DRIVER sha256 $(sha256sum "$DRIVER" | cut -c1-64)"
# The recipe's gate: this build reuses only images this driver produced (the content key does not
# cover the driver binary); every other entry moves out of the cache and is compiled again.
python3 tools/native/prebuilt/provenance.py gate "$OUT" "$DRIVER"
LOG=$(mktemp "${TMPDIR:-/tmp}/prebuilt-build.XXXXXX.log")
CONFIG=-; CARRIED=()
if [ -z "$KNOBS" ]; then
  if [ -n "${NPU_COMPILER_CONFIG_PATH:-}" ]; then
    echo "the reference build takes its compiler config from src/host/prebuilt/config.yaml (carried kernels only): unset NPU_COMPILER_CONFIG_PATH"; exit 1
  fi
  [ -f src/host/prebuilt/config.yaml ] && CONFIG=$ROOT/src/host/prebuilt/config.yaml
  carried=$(PYTHONPATH=tools/native/prebuilt python3 -B -c 'from pack import declarations; print(*declarations()[0], sep="\n")')
  mapfile -t CARRIED <<< "$carried"
  python3 tools/native/prebuilt/provenance.py knobgate "$OUT" "$CONFIG" "${CARRIED[@]}"
  if [ "$CONFIG" != - ]; then echo "carried compiler config: $CONFIG ($(sha256sum "$CONFIG" | cut -c1-16)): $(grep -v '^#' "$CONFIG" | tr '\n' ' ')"; fi
fi
trap 'python3 tools/native/prebuilt/provenance.py record "$OUT" "$DRIVER" "$LOG"; if [ -z "$KNOBS" ]; then python3 tools/native/prebuilt/provenance.py knobrecord "$OUT" "$CONFIG" "$LOG" "${CARRIED[@]}"; fi' EXIT
if [ -n "${CAPTURE_KERNELS:-}" ]; then
  # capture.py needs this to match DMA-only kernels such as spm_clean. Fail before
  # capture eviction or compilation: this dependency is otherwise used only after the expensive build.
  python3 -c 'from elftools.elf.elffile import ELFFile' || {
    echo "capture requires pyelftools in the Python environment (or PYTHONPATH); no build attempted" >&2
    exit 1
  }
  CAPTURE_ROOT=${CAPTURE_ROOT:-$OUT.captures/$(date +%Y%m%d-%H%M%S)}
  if [ -d "$CAPTURE_ROOT/captures" ] && [ -n "$(ls -A "$CAPTURE_ROOT/captures")" ]; then
    echo "CAPTURE_ROOT $CAPTURE_ROOT holds captures of an earlier build; give a fresh one"; exit 1
  fi
  mkdir -p "$CAPTURE_ROOT/captures" "$OUT.evicted/capture-$(date +%Y%m%d-%H%M%S)"
  for k in $CAPTURE_KERNELS; do
    for f in "$OUT/furiosa-opt-gemma4/$k.bin" "$OUT/furiosa-opt-gemma4/$k.hash" "$OUT/furiosa-opt-gemma4/$k.edf"; do
      if [ -e "$f" ]; then mv "$f" "$(ls -d "$OUT".evicted/capture-* | tail -1)/"; fi
    done
  done
  export DM_CACHE_CAPTURE_ROOT="$CAPTURE_ROOT"
  source "$ROOT/tools/native/fopt/capture/enable.sh"
  touch src/lib.rs src/bin/server.rs   # a warm target skips the up-to-date crate, and the evicted kernels never compile
fi
if [ -n "$KNOBS" ]; then
  KNOBS=$(realpath -m "$KNOBS")
  if [ -f "$KNOBS/ordering.txt" ]; then
    SCHEDULER_MANUAL_ORDERING_PATH=$(python3 - "$ROOT" "$KNOBS/ordering.txt" <<'PY'
from pathlib import Path
import sys
root, path = (Path(p).resolve() for p in sys.argv[1:])
print(path.relative_to(root) if path.is_relative_to(root) else path)
PY
)
    export SCHEDULER_MANUAL_ORDERING_PATH
    echo "ordering key path: $SCHEDULER_MANUAL_ORDERING_PATH"
  else
    unset SCHEDULER_MANUAL_ORDERING_PATH
  fi
  if [ -f "$KNOBS/config.yaml" ]; then export NPU_COMPILER_CONFIG_PATH="$KNOBS/config.yaml"; else export NPU_COMPILER_CONFIG_PATH="$ROOT/src/host/prebuilt/config.yaml"; fi
  # The compiler key includes ordering and omits config and axis sizes. Always
  # compile this one kernel afresh; keep the old entry outside the cache for diagnosis.
  EVICT="$OUT.evicted/knob-$(date +%Y%m%d-%H%M%S)-$$"
  for ext in bin hash edf; do
    f="$OUT/furiosa-opt-gemma4/$FN.$ext"
    if [ -e "$f" ]; then mkdir -p "$EVICT"; mv "$f" "$EVICT/"; fi
  done
  echo "knob build: $FN"
  # Exact compilation has no --features option. Expose carried definitions to its Rust frontend.
  RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }--cfg feature=\"compile-carried\"" \
    CARGO_NET_OFFLINE=true FURIOSA_OPT_OUT_DIR="$OUT" cargo furiosa-opt compile "${FN#furiosa_opt_gemma4::}" --exact 2>&1 | tee "$LOG"
  [ -f "$OUT/furiosa-opt-gemma4/$FN.bin" ] || { echo "FAIL: the knob build left no $FN.bin in $OUT"; exit 1; }
else
  if [ "$CONFIG" != - ]; then export NPU_COMPILER_CONFIG_PATH="$CONFIG"; fi
  # Parallel kernel compiles (tools/native/shard_build.py: shard binaries fill the cache concurrently, then the same
  # opt_build.sh build reuses them; images byte-identical). SHARDS=0 keeps the single serial build.
  if [ "${SHARDS:-}" = 0 ]; then
    FURIOSA_OPT_OUT_DIR="$OUT" tools/native/opt_build.sh --release --offline --locked --features compile-carried --bin server 2>&1 | tee "$LOG"
  else
    rc=0
    FURIOSA_OPT_OUT_DIR="$OUT" python3 tools/native/shard_build.py ${SHARDS:+--shards "$SHARDS"} --table tools/native/shards/reference.tsv \
      --logs "$OUT.shardlogs" -- --release --offline --locked --features compile-carried --bin server 2>&1 | tee "$LOG" || rc=$?
    # The provenance records (EXIT trap) parse the compile lines, also of a failed build's finished images.
    cat "$OUT.shardlogs"/phase*.log >> "$LOG" 2>/dev/null || true
    [ "$rc" = 0 ] || exit "$rc"
  fi
fi
# (set -e and pipefail: a failed build has exited above, its provenance recorded by the trap.)
if [ -n "${CAPTURE_KERNELS:-}" ]; then python3 tools/native/prebuilt/capture.py "$OUT" "$CAPTURE_ROOT" $CAPTURE_KERNELS; fi
# Exact compilation bypasses opt_build.sh. Record the axis geometry of the
# image in case this cache is reused by a subsequent build.
if [ -n "$KNOBS" ]; then
  python3 tools/native/opt_cache_rekey.py --axes-record src "$OUT"
fi
