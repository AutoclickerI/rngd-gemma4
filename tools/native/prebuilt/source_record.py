#!/usr/bin/env python3
"""Record and verify the source inputs to the carried images."""
import argparse
import json
from pathlib import Path
import shutil

from pack import PAYLOADS, PREBUILT, ROOT, SOURCE, declarations, settings, sha
from unpack import rows


def record():
    manifest_path = PREBUILT/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    images = {name: data for name, _, data in rows(PREBUILT/'payload.frcz')}
    kernels = manifest['kernels']
    paths, knobs = declarations()
    if paths != [row['path'] for row in kernels]:
        raise ValueError('carried Rust declarations differ from the manifest')
    if knobs != {row['path'] for row in kernels if row['knobs_dir'] != '.'}:
        raise ValueError('per-kernel compiler build set differs from the manifest')
    if set(images) != {row['carried'] for row in kernels}:
        raise ValueError('carried image coverage differs from the manifest')
    if manifest['carried_config']['sha256'] != sha((PREBUILT/'config.yaml').read_bytes()):
        raise ValueError('common compiler config differs from the manifest')
    for row in kernels:
        if sha(images[row['carried']]) != row['image_sha256']:
            raise ValueError('carried image differs from the manifest: ' + row['path'])
        if settings(row['knobs_dir']) != row['knobs']:
            raise ValueError('compiler settings differ from the manifest: ' + row['path'])

    paths = [p for p in SOURCE.rglob('*.rs')
             if p.relative_to(SOURCE).parts[0] not in {'api', 'host', 'bin'}]
    paths += [manifest_path, ROOT/'Cargo.toml', ROOT/'Cargo.lock', ROOT/'rust-toolchain.toml',
              ROOT/'.cargo/config.toml', SOURCE/'host/prebuilt.rs',
              SOURCE/'host/engine/step_image_table.rs',
              ROOT/'tools/reproduce/toolchain.py',
              ROOT/'tools/native/step/step_image_table.rs.in']
    paths += sorted((SOURCE/'host/engine/step_image').glob('*g12*.rs'))
    paths += sorted(PREBUILT.glob('*.yaml')) + sorted(PREBUILT.glob('task-c-sha256.json'))
    paths += [p for p in (PREBUILT/'knobs').rglob('*') if p.is_file()]
    paths += [p for p in (ROOT/'tools/native').rglob('*')
              if p.is_file() and ('target' not in p.parts and '__pycache__' not in p.parts)
              and (p.suffix in {'.py', '.sh', '.rs', '.tsv', '.toml', '.lock'}
                   or p.name == 'aarch64-linux-gnu-gcc')]
    return {
        'schema': 'carried-source-record-v2',
        'sources_sha256': {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in sorted(set(paths))},
        'payloads_sha256': {name: sha((SOURCE/name).read_bytes()) for name in PAYLOADS},
    }


def verify():
    expected = json.loads((PREBUILT/'source-record.json').read_text())
    if record() != expected:
        raise ValueError('carried source record differs; regenerate it before comparing a fresh build')
    driver = shutil.which('furiosa-opt-driver')
    if not driver:
        raise ValueError('put the pinned furiosa-opt-driver on PATH')
    manifest = json.loads((PREBUILT/'manifest.json').read_text())
    if sha(Path(driver).read_bytes()) != manifest['stock_driver_sha256']:
        raise ValueError('external compiler identity differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['write', 'check'])
    args = parser.parse_args()
    if args.mode == 'write':
        (PREBUILT/'source-record.json').write_text(json.dumps(record(), indent=2, sort_keys=True) + '\n')
    else:
        verify()
        print('source and compiler bindings match')


if __name__ == '__main__':
    main()
