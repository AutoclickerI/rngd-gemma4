#!/usr/bin/env python3
"""Compose and link task C translation units for carried STEP images."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from relink import link

FLAGS = ['-r', '-fno-builtin', '-fno-zero-initialized-in-bss', '-Werror=stack-usage=102400', '-nostdlib', '-fwrapv',
         '-static', '-Wl,-n', '-Werror', '-fno-omit-frame-pointer', '-O3']
TAIL = ['-lpe', '-std=c11']
CHUNK_LIMIT = 1 << 20
WEIGHT_ALIGN = 4096
ENTRY_RE = re.compile(r'^void task_0\(uint64_t \*addrs, uint32_t _, uint16_t _profile_base_uid\) \{$', re.M)
TRAMP = '    /* Jump to PERT via trampoline */\n    trampoline(TRAMPOLINE_EXIT, 0, 0);\n}'


def split(foptool, image, dst):
    subprocess.run([foptool, 'split', str(image), str(dst)], check=True)
    meta = json.loads((dst / 'parts.json').read_text())
    meta['weight_bytes'] = (dst / 'weight.bin').read_bytes() if meta['weight'] else b''
    return meta


def body_entry(src, k):
    """The body's task_0 as `dispatch_body_<k>`: the per-cluster call without the exit trampoline."""
    m = list(ENTRY_RE.finditer(src))
    assert len(m) == 1, f'body {k}: expected one task_0 definition, found {len(m)}'
    head, rest = src[:m[0].start()], src[m[0].start():]
    assert rest.count(TRAMP) == 1 and rest.rstrip().endswith(TRAMP), f'body {k}: task_0 does not end in the exit trampoline'
    rest = rest.replace(m[0].group(0), f'void dispatch_body_{k}(uint64_t *addrs) {{', 1)
    rest = rest.replace(TRAMP, '}', 1)
    return head + rest


WEIGHT_ACCESS = 'launch_tensor_offset(addrs, DRAM_WEIGHT, '


def rebase_weight(src, k, off):
    """Add the body's weight base to every Opt 0.9.0 weight access."""
    if not off:
        return src
    reads = src.count(WEIGHT_ACCESS)
    assert reads, f'body {k}: has a weight but its task C never reads DRAM_WEIGHT'
    assert src.count('DRAM_WEIGHT') == reads + 1 and src.count('DRAM_WEIGHT = 0,') == 1, \
        f'body {k}: DRAM_WEIGHT used outside launch_tensor_offset weight reads'
    assert src.count('#define launch_tensor_offset(addrs, slot, offset) ((addrs)[slot] + (offset))') == 1, \
        f'body {k}: launch_tensor_offset is not the expected macro'
    assert 'addrs[' not in src and src.count('(addrs)[') == 1, f'body {k}: a direct addrs index besides the macro'
    out = src.replace(WEIGHT_ACCESS, f'{WEIGHT_ACCESS}{off:#x} + ')
    assert out.count(f'{WEIGHT_ACCESS}{off:#x} + ') == reads
    return out


def compile_link(sources, cluster, toolroot, td, tag, *,
                 source_pool=False, cc_extra=(), enforce_chunk_limit=True):
    if source_pool:
        from source_pool import pool_sources, emitted_objects
        stock = td / 'stock'
        stock.mkdir()
        compile_link(sources, cluster, toolroot, stock, tag, enforce_chunk_limit=False)
        sources = pool_sources(sources, emitted_objects(stock / f'{tag}_c{cluster}.elf'))
    files = []
    for i, text in enumerate(sources):
        p = td / f'{tag}_{i}.c'
        p.write_text(text)
        files.append(str(p))
    elf = td / f'{tag}_c{cluster}.elf'
    # assert() in generated task C embeds __FILE__. Random TemporaryDirectory names used to change
    # the actual task bytes and section addresses. Normalize only the two identified build roots,
    # at compilation (never by editing or accepting a normalized image after the fact).
    prefix_maps = []
    for root, canonical in ((toolroot, '/fopt/dispatch/toolroot'), (td, '/fopt/dispatch/generated')):
        for spelling in sorted({str(root), str(Path(root).absolute()), str(Path(root).resolve())}):
            prefix_maps += [f'-fmacro-prefix-map={spelling}={canonical}',
                            f'-ffile-prefix-map={spelling}={canonical}']
    gcc = os.environ.get('FOPT_GCC') or 'aarch64-linux-gnu-gcc'
    cmd = [gcc, *FLAGS, *prefix_maps, '-xc', *files, *TAIL, f'-L{toolroot}', f'-I{toolroot}',
           f'-o{elf}', f'-DCLUSTER_INDEX={cluster}']
    if '-flto' in cc_extra:
        objects=[]
        for i,p in enumerate(files):
            obj=td/f'{tag}_{i}.o';objects.append(str(obj))
            own_flags=[f for f in FLAGS if f not in ('-r','-static','-Wl,-n')]
            r=subprocess.run([gcc,'-c',*own_flags,*cc_extra,*prefix_maps,
                              '-xc',p,'-std=c11',f'-I{toolroot}',f'-DCLUSTER_INDEX={cluster}',f'-o{obj}'],capture_output=True)
            if r.returncode:sys.exit(f'{tag} cluster {cluster}: LTO C compile failed\n'+r.stderr.decode()[-4000:])
        cmd=[gcc,*FLAGS,*cc_extra,'-flinker-output=nolto-rel',*objects,
             '-lpe',f'-L{toolroot}',f'-o{elf}']
    elif cc_extra:
        cmd[1+len(FLAGS):1+len(FLAGS)]=list(cc_extra)
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode:
        sys.exit(f'{tag} cluster {cluster}: gcc failed\n' + r.stderr.decode()[-4000:])
    blob, addr, undefined, _ = link(str(elf))
    if undefined:
        sys.exit(f'{tag} cluster {cluster}: unresolved symbols: {", ".join(sorted(undefined))}')
    print(f'{tag} cluster {cluster}: {len(blob)} B, sections ' + ', '.join(f'{k}@{v:#x}' for k, v in sorted(addr.items(), key=lambda x: x[1])), file=sys.stderr)
    if enforce_chunk_limit and len(blob) > CHUNK_LIMIT:
        sys.exit(f'{tag} cluster {cluster}: {len(blob)} B exceeds the {CHUNK_LIMIT} B chunk limit')
    return blob
