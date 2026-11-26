#!/bin/bash
# Build with Opt 0.9.0 using a private cache. Axis geometry changes
# invalidate opaque keys; the compiler validates ordinary source dependencies.
set -euo pipefail
OUT=${FURIOSA_OPT_OUT_DIR:?set FURIOSA_OPT_OUT_DIR to a private image cache}
TOOLS=$(cd "$(dirname "$0")" && pwd)
VERSION=$(cargo furiosa-opt --version | awk '{print $2}')
[ "$VERSION" = 0.9.0 ] || { echo "Opt 0.9.0 required; found $VERSION" >&2; exit 2; }
python3 "$TOOLS/opt_cache_rekey.py" --conservative-check src "$OUT"

# Worktrees can share a target whose Cargo freshness timestamps are newer than
# this checkout. Touch changed inputs before building so the current source wins.
TGT=${CARGO_TARGET_DIR:-target}
SRC_HASH=$(find src Cargo.toml Cargo.lock -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -c1-16)
STAMP=$TGT/.opt-build-src
if [ "$(cat "$STAMP" 2>/dev/null)" != "$SRC_HASH" ]; then
  find src -type f -exec touch {} +
  touch Cargo.toml Cargo.lock
fi
cargo furiosa-opt build "$@"
mkdir -p "$TGT"
printf '%s\n' "$SRC_HASH" > "$STAMP"
python3 "$TOOLS/opt_cache_rekey.py" --axes-record src "$OUT"
