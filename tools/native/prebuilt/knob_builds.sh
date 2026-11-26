#!/bin/bash
# Build individually captured carried kernels with any recorded ordering/config.
# KNOB_JOBS workers (default four) each own a Cargo target.
# Usage: knob_builds.sh OUT
set -euo pipefail
BASE=$1
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$ROOT"
knob_list=$(PYTHONPATH=tools/native/prebuilt python3 -B - <<'PYKNOBS'
from pack import kernel_specs

for path, folder, _ in kernel_specs():
    if folder != '.':
        print(path, folder)
PYKNOBS
)
mapfile -t kernels <<< "$knob_list"
mkdir -p "$BASE"
TGT=$BASE/target
WORKERS=${KNOB_JOBS:-4}
[[ $WORKERS =~ ^[1-9][0-9]*$ ]] || { echo "KNOB_JOBS must be positive" >&2; exit 2; }
# A fixed worker owns its Cargo target for the whole shard. Compiler image
# caches and capture roots remain per kernel; only unchanged Rust dependencies
# are reused between successive compiles in that worker.
for ((worker=0; worker<WORKERS; worker++)); do
  (
    for ((i=worker; i<${#kernels[@]}; i+=WORKERS)); do
      read -r path folder <<< "${kernels[$i]}"
      n=${folder##*/}
      rc=0
      (
        export CAPTURE_KERNELS="$path" CAPTURE_ROOT="$BASE/captures/$n"
        CARGO_TARGET_DIR="$TGT-knob$worker" tools/native/prebuilt/build.sh "$BASE/$n" "src/host/prebuilt/$folder" "$path"
      ) > "$BASE/$n.log" 2>&1 || rc=$?
      echo "$rc" > "$BASE/$n.rc"
    done
  ) &
done
wait
fail=0
for kernel in "${kernels[@]}"; do
  n=${kernel##*/}
  if [ "$(cat "$BASE/$n.rc" 2>/dev/null)" = 0 ]; then echo "knob build: $n OK"; else echo "FAIL: knob build of $n ($BASE/$n.log)"; fail=1; fi
done
exit $fail
