"""Build the image tool and bind native runtime files to the installed Opt driver."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


# Byte extents of the runtime embedded in the pinned Opt 0.9.0 driver.
RUNTIME = {'libpe.a': (24069191, 26232, '13852f208c41243f3fbc41eb32d3783b46bcffe6103e0f3b67b2a58cef9362d1'),
 'pe.h': (24037384, 14820, '69f386b4708eccfa7e9aef965aeb32f78de0f462c9ac7e730a65e85d126c5641'),
 'printf.h': (24095423, 4929, 'd8ee710660c8ec62c359ab36d0dd2664356d6ab98829ecf0a1beb4c0121e0e02'),
 'span.h': (24068955, 236, 'bb406874d298eb7f4e724bdff9d9a90ded03f0f120c79603a2e09d686e46d2f4'),
 'tuc.h': (24052204, 16751, '13a7ce40547dc476964bd129a38635adaef86c63222354db2b80c298dc463a07')}


def extract_runtime(driver, binding, output):
    """Validate the external compiler and every slice before creating output."""
    driver = Path(driver).resolve()
    output = Path(output)
    if sha(driver) != binding:
        raise ValueError('external Opt driver identity differs')
    pieces = {}
    with driver.open('rb') as stream:
        for name, (offset, length, digest) in RUNTIME.items():
            stream.seek(offset)
            data = stream.read(length)
            if len(data) != length or hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f'external runtime slice differs: {name}')
            pieces[name] = data
    output.mkdir(parents=True, exist_ok=False)
    for name, data in pieces.items():
        (output / name).write_bytes(data)


def prepare(repo, driver_sha256, output):
    """Return a source-built foptool and runtime extracted into the task output."""
    repo, output = Path(repo).resolve(), Path(output).resolve()
    selected = shutil.which('furiosa-opt-driver')
    if not selected:
        raise ValueError('put the pinned furiosa-opt-driver on PATH')
    runtime = output / 'runtime'
    extract_runtime(selected, driver_sha256, runtime)
    target = output / 'target'
    env = dict(os.environ, CARGO_TARGET_DIR=str(target))
    with (output / 'foptool-build.log').open('w') as log:
        subprocess.run(['cargo', 'build', '--manifest-path',
                        str(repo / 'tools/native/fopt/Cargo.toml'),
                        '--release', '--offline', '--locked'],
                       env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
    foptool = target / 'release/foptool'
    if not foptool.is_file():
        raise ValueError('source build did not produce foptool')
    return foptool, runtime
