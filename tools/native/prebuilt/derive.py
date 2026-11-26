#!/usr/bin/env python3
"""Generate serving images, task-C digests and Rust tables from fresh Opt captures."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from pack import (
    CACHE, DRIVER_SHA256, GENERATED, G12_TABLES, PAYLOAD, PAYLOADS,
    ROOT, SOURCE, STEP, STEP_TABLE, TASK_DIGESTS, cache_for, clock_rows,
    carried_path, declarations, encode, kernel_specs, sha,
)
from fopt_header import header
from unpack import CRATE

TOOLS = ROOT / 'tools/native'
sys.path[:0] = [str(TOOLS / name) for name in ('fopt', 'postpass', 'step')]
from postpass import clock_push
from binding_check import slot_sizes, table, template_source, write_table
import layout
import dispatch_build
import rebuild_g12
sys.path.insert(0, str(ROOT / "tools/reproduce"))
from toolchain import prepare


def output_path(folder, row):
    return folder / row.split('::')[-1]


def run(command, log):
    with log.open('w') as stream:
        try:
            subprocess.run(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f'command failed, see {log}') from error


def prepare_source(scratch):
    """Create a compile-only crate with stubs for the generated runtime data."""
    driver = shutil.which('furiosa-opt-driver')
    if not driver or sha(Path(driver).read_bytes()) != DRIVER_SHA256:
        raise ValueError('put the pinned furiosa-opt-driver on PATH')
    sys.path.insert(0, str(TOOLS))
    from shard_build import MARK

    generated = {Path(name) for name in GENERATED}

    def skip(folder, names):
        relative = Path(folder).relative_to(SOURCE)
        return [name for name in names if relative / name in generated]

    shutil.copytree(SOURCE, scratch/'src', ignore=skip)
    shutil.copytree(ROOT/'.cargo', scratch/'.cargo')
    shutil.copytree(TOOLS, scratch/'tools/native', ignore=shutil.ignore_patterns('target', '__pycache__'))
    for name in ('Cargo.lock', 'rust-toolchain.toml'):
        shutil.copyfile(ROOT/name, scratch/name)
    manifest = (ROOT/'Cargo.toml').read_text().split(MARK, 1)[0]
    empty_payload = encode([])
    for payload in PAYLOADS:
        stub = scratch/'src'/payload
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.write_bytes(empty_payload)
    (scratch/'src'/STEP_TABLE).write_text(template_source(declarations()[0]))
    for path in G12_TABLES:
        if path.name.endswith('_persist.rs'):
            stub = ('use super::Recipe;\npub(super) const PATH: &str = "";\n'
                    'pub(super) const RECIPES: &[Recipe] = &[];\n')
        elif path.name == 'p941_g12_slots.rs':
            stub = 'pub(super) const RSS: &[usize] = &[];\npub(super) const RFS: &[usize] = &[];\n'
        else:
            stub = 'pub(super) const OPERANDS: &[&[usize]] = &[];\n'
        (scratch/'src'/path).write_text(stub)
    (scratch/'Cargo.toml').write_text(manifest)


def digest_rows(kernels, clocks):
    """Select the task-C digests published alongside the image manifest."""
    return clocks | {
        carried_path(path)
        for path, _, tool in kernels
        if (tool and not tool.endswith('/spm_clean.py'))
        or path.endswith('::rsf_int8_fp8kv_t1536_s8')
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path, help='Build directory, or destination for a compile-only crate')
    parser.add_argument('--cpus', type=int, default=8)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--reference-filters', action='store_true')
    mode.add_argument('--step-filters', action='store_true')
    mode.add_argument('--prepare-source', action='store_true')
    args = parser.parse_args()
    work = args.work.resolve()
    if args.prepare_source:
        prepare_source(work)
        return

    kernels = list(kernel_specs())
    paths = [path for path, _, _ in kernels]
    step_source = template_source(paths)
    members, _ = table(step_source)
    step_paths = {member['path'] for member in members}
    clocks = clock_rows()
    digests = digest_rows(kernels, clocks)
    needed_tasks = digests | {carried_path(path) for path in step_paths}
    if args.reference_filters:
        for path, folder, tool in kernels:
            if folder == '.' and (tool or carried_path(path) in needed_tasks):
                print(path)
        return
    if args.step_filters:
        for path in sorted(step_paths - set(paths)):
            print(path.removeprefix(CRATE))
        return

    output = work/'rust-derived'
    output.mkdir()
    packed = work/'packed'
    foptool, runtime = prepare(ROOT, DRIVER_SHA256, work/'toolchain')
    os.environ.update(FOPT_TOOLROOT=str(runtime), FOPTOOL=str(foptool))
    primary = output/'primary'
    primary.mkdir()

    def native(path, cache, tool, destination, need_task):
        stock, task = cache/f'{path}.bin', cache/f'{path}.task.c'
        image, out_task = destination.with_suffix('.bin'), destination.with_suffix('.c')
        if tool:
            command = [sys.executable, ROOT/tool, stock, task, image]
            if Path(tool).name != 'spm_clean.py':
                command.extend(('--task-out', out_task))
            run(command, destination.with_suffix('.log'))
        else:
            shutil.copyfile(stock, image)
            if need_task:
                shutil.copyfile(task, out_task)
        return image

    def primary_row(spec):
        path, folder, tool = spec
        name = carried_path(path)
        dest = output_path(primary, name)
        if name in clocks:
            base = native(path, cache_for(work, folder), tool, dest.with_name(dest.name + '-base'), True)
            task = dest.with_suffix('.c')
            task.write_bytes(clock_push(base.with_suffix('.c').read_bytes()))
            image = dest.with_suffix('.bin')
            run([sys.executable, TOOLS/'fopt/direct_build.py', base, task, image], dest.with_suffix('.log'))
        else:
            image = native(path, cache_for(work, folder), tool, dest, name in needed_tasks)
        return (PAYLOAD, name), image

    with ThreadPoolExecutor(max_workers=args.cpus) as pool:
        images = dict(pool.map(primary_row, kernels))

    def member_file(path, extension):
        name = carried_path(path)
        if (PAYLOAD, name) in images:
            return images[PAYLOAD, name].with_suffix('.'+extension)
        suffix = 'task.c' if extension == 'c' else extension
        return work/'step-stock'/CACHE/f'{path}.{suffix}'

    images.update(step_rows(work, packed, output, member_file, foptool, runtime, args.cpus, step_source))
    with (output/'rows.tsv').open('w', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(('carried file', 'row', 'rebuilt file'))
        writer.writerows((payload, name, image) for (payload, name), image in sorted(images.items()))
    hashes = {PAYLOAD: {
        name.rsplit('::', 1)[1]: sha(images[PAYLOAD, name].with_suffix('.c').read_bytes())
        for name in digests
    }}
    destination = packed/TASK_DIGESTS
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(hashes, indent=2, sort_keys=True) + '\n')


def family_sources(member_file, family, paths, offsets):
    name, slots = family['name'], family['slots']
    rss_image = member_file(paths[0], 'bin').read_bytes()
    chunks = []
    for chunk, opcodes in enumerate(layout.chunk_opcodes(name, paths)):
        units = []
        for opcode in opcodes:
            task = clock_push(member_file(paths[opcode], 'c').read_bytes()).decode()
            offset = offsets.get(opcode, 0) * dispatch_build.WEIGHT_ALIGN
            units.append(dispatch_build.body_entry(dispatch_build.rebase_weight(task, opcode, offset), opcode))
        if chunk == 0:
            variants = layout.rider_variants(name, slots, units[0], rss_image)
        units.append(layout.dispatcher(name, slots, chunk, opcodes, variants))
        chunks.append(units)
    return chunks, variants


def step_weights(work):
    path = CRATE + 'ops_core::ends::last_bf16_out128'
    source = (work/'reference'/CACHE/f'{path}.bin').read_bytes()
    meta = header(source)
    page = dispatch_build.WEIGHT_ALIGN
    start, length = meta['weight_offset'], meta['weight'][1]
    if length % page:
        raise ValueError(f'{path}: the stock weights are misaligned')
    stock = {i: source[at:at+page] for i, at in enumerate(range(start, start+length, page))}
    identity = bytes(255 if row == column else 0 for row in range(32) for column in range(32))
    stock[-1] = identity * (page // len(identity))
    return stock


def link_family(folder, metadata, chunks, weights, foptool, runtime, cpus):
    parts = folder/'parts'
    parts.mkdir()
    (parts/'weight.bin').write_bytes(weights)

    def link(pair):
        chunk, cluster = pair
        scratch = folder/f'chunk{chunk}-{cluster}'
        scratch.mkdir()
        blob = dispatch_build.compile_link(chunks[chunk], cluster, runtime, scratch, f'chunk{chunk}',
                                           source_pool=True, cc_extra=layout.LINK_FLAGS)
        (parts/f'task_{chunk}_{cluster}.bin').write_bytes(blob)
        return chunk, cluster, len(blob)

    pairs = [(i, c) for i, tasks in enumerate(metadata['tasks']) for c in range(len(tasks))]
    with ThreadPoolExecutor(max_workers=cpus) as pool:
        for chunk, cluster, size in pool.map(link, pairs):
            metadata['tasks'][chunk][cluster] = size
    (parts/'parts.json').write_text(json.dumps(metadata))
    image = folder/'family.bin'
    run([foptool, 'join', parts, image], folder/'join.log')
    return image


def write_g12_tables(generated, member_file, family, paths, variants):
    narrow = family['name'] == layout.G12
    stem = 'p941_g12' if narrow else 'p999_g12'
    comment = '// Generated by tools/native/prebuilt/derive.py.\n'
    (generated/f'{stem}_persist.rs').write_text(comment + layout.rider_source(paths[0], variants))

    def sizes(op):
        return [] if paths[op] is None else slot_sizes(member_file(paths[op], 'bin').read_bytes())

    if narrow:
        declarations = [f'pub(super) const {label}: &[usize] = &{sizes(op)};'
                        for op, label in ((0, 'RSS'), (2, 'RFS'))]
    else:
        declarations = ['pub(super) const OPERANDS: &[&[usize]] = &[',
                        *(f'    &{sizes(op)},' for op in range(len(paths))), '];']
    (generated/f'{stem}_slots.rs').write_text(comment + '\n'.join(declarations) + '\n')


def step_rows(work, packed, output, member_file, foptool, runtime, cpus, step_source):
    members, families = table(step_source)
    step_out = output/'step'
    step_out.mkdir()
    generated = (packed/G12_TABLES[0]).parent
    generated.mkdir(parents=True, exist_ok=True)
    stock = step_weights(work)
    for family in families:
        name, row = family['name'], family['row']
        paths = [None if i is None else members[i]['path'] for i in family['ops']]
        offsets, page_order = layout.layout(name)
        weights = b''.join(stock[index] for index in page_order)
        folder = step_out/name
        folder.mkdir()
        metadata = layout.metadata(name, family['slots'], len(weights))
        if name == layout.G12:
            weights = rebuild_g12.append_weights(paths, folder, foptool, member_file, offsets, weights, metadata)
        chunks, variants = family_sources(member_file, family, paths, offsets)
        write_g12_tables(generated, member_file, family, paths, variants)
        image = link_family(folder, metadata, chunks, weights, foptool, runtime, cpus)
        yield (STEP, row), image

    write_table(packed/STEP_TABLE, step_source,
                lambda path: member_file(path, 'bin').read_bytes(),
                lambda row: (step_out/row.split('::')[-1]/'family.bin').read_bytes())


if __name__ == '__main__':
    raise SystemExit(main())
