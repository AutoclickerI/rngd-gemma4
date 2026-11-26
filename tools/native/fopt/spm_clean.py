#!/usr/bin/env python3
"""The carried image of `ops_core::spm_clean::spm_clean`: its stock image rebuilt from its task C with a
D-cache clean of the PE firmware's shared fields at the start of each cluster's task.

The PE firmware's Initialize zeroes the shared fields (SPM 0xEB000..0x100000, `struct shared_field_t`) with CPU
stores. Its MMU maps only the DMA descriptor arena (0xEC000..0xFC000, TUC_DMA_DESC_ARENA_END) non-cacheable, so
the lead PE's D-cache keeps dirty zero lines over the rest. The gather index lists `dma_indirect_indices` lie there,
and DMA writes them. A dirty line evicted after an index load writes zeros over 16 offsets of that gather.
`dc civac` over [0xEB000, 0xEC000) and [0xFC000, 0x100000) (320 lines of 64 B) at the start of the engine's first
launch writes those lines back before any DMA of the process has written SPM. Nothing makes them dirty again: task C
and libpe store only to core-owned fields, never to the index lists, which are 256-B aligned.

1. Rebuild the unmodified task C with tools/native/fopt/direct_build.py. It must reproduce the stock image byte for
   byte (this checks the toolchain: gcc, libpe.a, relink, foptool).
2. Insert the clean before `tuc_init();` in each cluster's `task_0_chip0_cluster<c>` and rebuild. Every other byte
   of the task C is unchanged.

usage: spm_clean.py <stock image> <task.c> <out image> [--toolroot DIR] [--foptool PATH]
  <task.c>: the task C that build.sh captures beside the image.
  Supply the extracted Opt runtime with --toolroot or FOPT_TOOLROOT.
  Use --foptool or FOPTOOL for the linker. GCC and pyelftools must be available.
"""
import argparse, hashlib, re, sys, tempfile
from pathlib import Path

from direct_build import build

LO, ARENA, ARENA_END, END = 0xEB000, 0xEC000, 0xFC000, 0x100000
HELPER = r'''/* Write back and invalidate the D-cache lines over the cacheable shared
 * fields the PE firmware's Initialize zeroed with CPU stores, before any DMA writes SPM there. */
_Static_assert(SHARED_FIELDS_ADDR == 0x%X && SHARED_AREA_SIZE == 0x%X && TUC_DMA_DESC_ARENA_END == 0x%X &&
               SHARED_FIELDS_ADDR + sizeof(struct shared_field_t) == 0x%X, "spm_clean: the shared-field layout changed");
static inline __attribute__((always_inline)) void spm_clean_shared_fields(void)
{
    __asm__ __volatile__("dsb sy" ::: "memory");
    for (uintptr_t a = 0x%X; a < 0x%X; a += 64)
        __asm__ __volatile__("dc civac, %%0" : : "r"(a) : "memory");
    for (uintptr_t a = 0x%X; a < 0x%X; a += 64)
        __asm__ __volatile__("dc civac, %%0" : : "r"(a) : "memory");
    __asm__ __volatile__("dsb sy" ::: "memory");
}''' % (LO, ARENA - LO, ARENA_END, END, LO, ARENA, ARENA_END, END)
TASK = re.compile(r'^static void task_0_chip0_cluster(\d+)\(uint64_t \*addrs\) \{$')


def patch(src: str) -> tuple[str, list]:
    """The task C with the clean before each cluster task's tuc_init(); the clusters patched."""
    lines, out, done = src.split('\n'), [], []
    for i, ln in enumerate(lines):
        m = TASK.match(ln)
        if m and not done:
            out.append(HELPER)
        out.append(ln)
        if m:
            if i + 1 >= len(lines) or lines[i + 1].strip() != 'tuc_init();':
                sys.exit(f'cluster {m.group(1)}: task_0_chip0_cluster{m.group(1)} does not start with tuc_init()')
            out.append('    spm_clean_shared_fields();')
            done.append(int(m.group(1)))
    return '\n'.join(out), done


def main():
    ap = argparse.ArgumentParser()
    for name in ('base', 'task', 'out'):
        ap.add_argument(name, type=Path)
    ap.add_argument('--toolroot', type=Path)
    ap.add_argument('--foptool', type=Path)
    a = ap.parse_args()
    base, task = a.base, a.task
    stock = base.read_bytes()
    src = task.read_text()
    patched, clusters = patch(src)
    if sorted(clusters) != list(range(len(clusters))) or len(clusters) < 1:
        sys.exit(f'{task}: cluster tasks {clusters}, expected 0..n-1')
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        original = d / 'v0.bin'
        build(base, task, original, a.toolroot, a.foptool)
        again = original.read_bytes()
        if again != stock:
            sys.exit(f'the unmodified task C does not rebuild the stock image ({hashlib.sha256(again).hexdigest()[:12]} '
                     f'vs {hashlib.sha256(stock).hexdigest()[:12]}): wrong toolroot, foptool or task C')
        (d / 'clean.c').write_text(patched)
        build(base, d / 'clean.c', a.out, a.toolroot, a.foptool)
    image = a.out.read_bytes()
    print(f'spm_clean: clusters {clusters}: dc civac [0x{LO:X}, 0x{ARENA:X}) + [0x{ARENA_END:X}, 0x{END:X}) before tuc_init; '
          f'stock {len(stock)} B {hashlib.sha256(stock).hexdigest()[:12]} rebuilt byte-identical; '
          f'clean {len(image)} B {hashlib.sha256(image).hexdigest()[:12]}')


if __name__ == '__main__':
    main()
