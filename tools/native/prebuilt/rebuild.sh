#!/bin/bash
# Rebuild carried rows in a private directory from the repository root.
# Usage: rebuild.sh OUT [reference|knobs|step|derive|pack|compare]
set -euo pipefail
OUT=$(realpath -m "${1:?provide an output directory}")
stage=${2:-all}
mkdir -p "$OUT/tmp"
export TMPDIR="$OUT/tmp" PYTHONDONTWRITEBYTECODE=1 CARGO_NET_OFFLINE=true
CPUS=${CARGO_BUILD_JOBS:-8}
TOOLS=tools/native/prebuilt
C=furiosa_opt_gemma4

compile_source() {
    local scratch
    scratch=$(mktemp -d "$OUT/$stage-source.XXXXXX")
    python3 -B "$TOOLS/derive.py" "$scratch" --prepare-source
    cd "$scratch"
}

case "$stage" in
    all)
        for phase in reference knobs step derive pack; do
            "$0" "$OUT" "$phase"
        done
        ;;
    reference)
        export CARGO_TARGET_DIR="$OUT/ref-target"
        export CAPTURE_ROOT="$OUT/reference-captures"
        export CAPTURE_KERNELS
        CAPTURE_KERNELS=$(python3 -B "$TOOLS/derive.py" "$OUT" --reference-filters)
        compile_source
        "$TOOLS/build.sh" "$OUT/reference"
        ;;
    knobs)
        compile_source
        "$TOOLS/knob_builds.sh" "$OUT/knobs"
        ;;
    step)
        export CARGO_TARGET_DIR="$OUT/step-target"
        export FURIOSA_OPT_OUT_DIR="$OUT/step-stock"
        export DM_CACHE_CAPTURE_ROOT="$OUT/step-stock-captures"
        filter_list=$(python3 -B "$TOOLS/derive.py" "$OUT" --step-filters)
        mapfile -t filters <<< "$filter_list"
        compile_source
        source "$PWD/tools/native/fopt/capture/enable.sh"
        RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }--cfg feature=\"compile-carried\"" \
            CARGO_NET_OFFLINE=true cargo furiosa-opt compile --exact "${filters[@]}"
        paths=()
        for filter in "${filters[@]}"; do paths+=("$C::$filter"); done
        python3 -B "$TOOLS/capture.py" "$FURIOSA_OPT_OUT_DIR" "$DM_CACHE_CAPTURE_ROOT" "${paths[@]}"
        ;;
    derive)
        python3 -B "$TOOLS/derive.py" "$OUT" --cpus "$CPUS"
        ;;
    pack)
        python3 -B "$TOOLS/pack.py" "$OUT"
        ;;
    compare)
        python3 -B "$TOOLS/pack.py" "$OUT" --compare
        ;;
    *)
        echo "unknown rebuild stage $stage" >&2
        exit 2
        ;;
esac
