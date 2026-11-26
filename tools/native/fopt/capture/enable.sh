# Source before invoking Opt with DM_CACHE_CAPTURE_ROOT set.
capture_bin=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if ! gcc_on_path=$(type -P aarch64-linux-gnu-gcc) ||
   ! FOPT_GCC=$(realpath "$gcc_on_path") ||
   [ "$FOPT_GCC" = "$capture_bin/aarch64-linux-gnu-gcc" ] ||
   [ ! -x "$FOPT_GCC" ]; then
  echo "put the real aarch64-linux-gnu-gcc on PATH before enabling capture" >&2
  return 1
fi
export FOPT_GCC
export PATH="$capture_bin:$PATH"
unset capture_bin gcc_on_path
