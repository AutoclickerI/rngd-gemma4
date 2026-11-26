#!/usr/bin/env python3
"""Build an Opt image from task C and the image's existing header.

Compile one AArch64 ELF per cluster and relink its task before joining the
image. Supply the extracted Opt runtime via --toolroot or FOPT_TOOLROOT.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from relink import link

HERE = Path(__file__).resolve().parent
FLAGS = ['-r', '-fno-builtin', '-fno-zero-initialized-in-bss', '-Werror=stack-usage=102400', '-nostdlib', '-fwrapv',
         '-static', '-Wl,-n', '-xc', '-Werror', '-fno-omit-frame-pointer', '-O3', '-', '-lpe', '-std=c11']


def build(base, task, out, toolroot=None, foptool=None):
    toolroot = toolroot or os.environ.get('FOPT_TOOLROOT')
    if not toolroot:
        raise ValueError('supply the extracted Opt runtime with --toolroot or FOPT_TOOLROOT')
    foptool = foptool or os.environ.get('FOPTOOL') or HERE / 'target/release/foptool'
    src = task.read_bytes()
    gcc = os.environ.get('FOPT_GCC') or 'aarch64-linux-gnu-gcc'
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        if base.is_dir():
            shutil.copytree(base, td, dirs_exist_ok=True)
        else:
            subprocess.run([foptool, 'split', base, str(td)], check=True)
        meta = json.loads((td / 'parts.json').read_text())
        assert len(meta['tasks']) == 1, 'multi-chunk base: extend the per-chunk loop'
        for c in range(len(meta['tasks'][0])):
            elf = td / f'c{c}.elf'
            cmd = [gcc, *FLAGS, f'-L{toolroot}', f'-I{toolroot}',
                   f'-o{elf}', f'-DCLUSTER_INDEX={c}']
            r = subprocess.run(cmd, input=src, capture_output=True)
            if r.returncode:
                raise ValueError(r.stderr.decode())
            try:
                blob, _, undefined, _ = link(str(elf))
            except ValueError as error:
                raise ValueError(f'cluster {c}: invalid task link: {error}') from error
            if undefined:
                raise ValueError(f'cluster {c}: unresolved symbols: {", ".join(sorted(undefined))}')
            (td / f'task_0_{c}.bin').write_bytes(blob)
        meta['tasks'] = [[(td / f'task_0_{c}.bin').stat().st_size for c in range(len(meta['tasks'][0]))]]
        (td / 'parts.json').write_text(json.dumps(meta))
        subprocess.run([foptool, 'join', str(td), out], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'task', 'out'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--toolroot', type=Path)
    parser.add_argument('--foptool', type=Path)
    args = parser.parse_args()
    build(args.base, args.task, args.out, args.toolroot, args.foptool)


if __name__ == '__main__':
    main()
