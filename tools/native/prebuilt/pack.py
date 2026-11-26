#!/usr/bin/env python3
"""Pack fresh serving images or compare a completed output against the checkout."""
import argparse
import csv
import hashlib
import json
import lzma
from pathlib import Path
import re
import struct
import sys

from unpack import CRATE, MAGIC, PREFIX, fnv1a64, rows

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "src"
PREBUILT = SOURCE / "host/prebuilt"
CACHE = "furiosa-opt-gemma4"
PAYLOAD = "host/prebuilt/payload.frcz"
STEP = "host/prebuilt/step.frcz"
PAYLOADS = (PAYLOAD, STEP)
MANIFEST = "host/prebuilt/manifest.json"
TASK_DIGESTS = "host/prebuilt/task-c-sha256.json"
ENGINE = Path("host/engine")
STEP_TABLE = ENGINE / "step_image_table.rs"
G12_TABLES = tuple(ENGINE / f"step_image/{stem}_{kind}.rs"
                   for stem in ("p941_g12", "p999_g12") for kind in ("persist", "slots"))
GENERATED = (*PAYLOADS, MANIFEST, TASK_DIGESTS, STEP_TABLE, *G12_TABLES)
DRIVER_SHA256 = '1363b347c2360d66d20d34ee3fcf2426d726defcefad311ec3c85f910e55fa24'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def carried_path(path):
    return PREFIX + path.removeprefix(CRATE)


def encode(records):
    raw = bytearray(struct.pack("<Q", len(records)))
    for name, guards, body in records:
        path = name.encode()
        raw += struct.pack("<Q", len(path)) + path
        raw += struct.pack("<Q", len(guards))
        for guard in guards:
            raw += struct.pack("<Q", guard)
        raw += struct.pack("<Q", len(body)) + body
    compressed = lzma.compress(bytes(raw), format=lzma.FORMAT_ALONE, preset=9 | lzma.PRESET_EXTREME)
    return MAGIC + struct.pack("<QQ", len(raw), fnv1a64(raw)) + compressed


def cache_for(work, folder):
    if folder != '.':
        return work / 'knobs' / Path(folder).name / CACHE
    return work / 'reference' / CACHE


def declarations():
    text = (SOURCE/'host/prebuilt.rs').read_text()
    carried = re.search(r'^carried_set!\(\$\s*(.*?)\);', text, re.M | re.S)[1]
    paths, knobs = [], set()
    for entry in re.sub(r'//[^\n]*', '', carried).strip().rstrip(',').split(','):
        mode, path = entry.split()
        path = CRATE + path
        paths.append(path)
        if mode != 'stock':
            knobs.add(path)
    return paths, knobs


def clock_rows():
    names = {
        'last_bf16_out128',
        'rsf_int8_fp8kv_t1024_s8',
        'mid_fs_q4_t1024',
        'mid_ss_q4_t1024',
        *(f'{member}_int8_fp8kv_t1024_g12_p941_pair' for member in ('rss', 'rfs')),
    }
    paths, _ = declarations()
    return {carried_path(path) for path in paths if path.rsplit('::', 1)[1] in names}


def payload_order():
    """Match the published container order using the serving declarations."""
    sys.path.insert(0, str(ROOT/'tools/native/step'))
    from binding_check import table, template_source

    paths, _ = declarations()
    _, families = table(template_source(paths))
    return {
        PAYLOAD: [carried_path(path) for path in paths],
        STEP: [family['row'] for family in families],
    }


def settings(folder):
    return {name: sha(path.read_bytes()) for name in ('config.yaml', 'ordering.txt')
            if (path := PREBUILT/folder/name).is_file()}


def kernel_specs():
    paths, knobs = declarations()
    for path in paths:
        kernel = path.rsplit('::', 1)[1]
        tool = None
        if '_q4_class512_' in kernel or '::cohort_q4::q4::' in path:
            tool = 'q4_cohort'
        elif kernel.startswith('last_bf16_out'):
            tool = 'compact_i4'
        elif path.endswith('::spm_clean'):
            tool = 'spm_clean'
        elif '::q4serve::' in path or ('::q4::' in path and '_t1536_' in kernel):
            tool = 'q4_wide'
        elif '::q4::' in path:
            tool = 'q4_n32_pvnormc'
        folder = 'knobs/' + path.removeprefix(CRATE).replace('::', '.') if path in knobs else '.'
        yield path, folder, f'tools/native/fopt/{tool}.py' if tool else None


def write_manifest(work, images):
    """Generate the stock and final image record using the fresh builds."""
    data = {
        'stock_driver_sha256': DRIVER_SHA256,
        'carried_config': {'file': 'config.yaml', 'sha256': settings('.')['config.yaml']},
        'kernels': [],
    }
    for path, folder, tool in kernel_specs():
        row = carried_path(path)
        image = images[PAYLOAD, row].read_bytes()
        evidence = {'path': path, 'carried': row, 'knobs_dir': folder, 'knobs': settings(folder),
                    'image_bytes': len(image), 'image_sha256': sha(image)}
        if tool:
            cache = cache_for(work, folder)
            evidence['rewrite'] = {
                'tool': tool,
                'base_image_sha256': sha((cache/f'{path}.bin').read_bytes()),
            }
        data['kernels'].append(evidence)
    output = work/'packed'/PAYLOAD
    data['payload_bytes'] = output.stat().st_size
    (work/'packed'/MANIFEST).write_text(json.dumps(data, indent=2) + '\n')


def compare(output):
    exact = True
    for name in GENERATED:
        actual, shipped = output / name, SOURCE / name
        missing = [label for file, label in ((actual, 'rebuilt'), (shipped, 'shipped')) if not file.is_file()]
        match = not missing and actual.read_bytes() == shipped.read_bytes()
        status = 'exact' if match else 'differs'
        detail = f' (missing {", ".join(missing)})' if missing else ''
        print(f'{name}: {status}{detail}')
        exact &= match
        if name in PAYLOADS:
            rebuilt, reference = row_bytes(actual), row_bytes(shipped)
            for row in sorted(rebuilt.keys() | reference.keys()):
                print(f'  {row}: {"exact" if rebuilt.get(row) == reference.get(row) else "differs"}')
    return int(not exact)


def row_bytes(path):
    if not path.is_file():
        return {}
    try:
        return {row: (guards, image) for row, guards, image in rows(path)}
    except (SystemExit, ValueError, struct.error, lzma.LZMAError) as error:
        print(f'  {path}: {error}')
        return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    work = args.work.resolve()
    if args.compare:
        return compare(work / "packed")
    with (work / "rust-derived/rows.tsv").open(newline="") as stream:
        table = list(csv.DictReader(stream, delimiter="\t"))
    images = {}
    for row in table:
        path = Path(row["rebuilt file"]).resolve()
        if not path.is_relative_to(work):
            parser.error(f"rebuilt file lies outside the work directory: {path}")
        identity = row["carried file"], row["row"]
        if identity in images:
            parser.error(f"duplicate rebuilt row: {identity}")
        images[identity] = path
    for payload, names in payload_order().items():
        records = []
        for name in names:
            image = images[payload, name].read_bytes()
            records.append((name, [], image))
        destination = work / "packed" / payload
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(encode(records))
        print(f'{payload}: packed {len(records)} rows')
    write_manifest(work, images)


if __name__ == "__main__":
    raise SystemExit(main())
