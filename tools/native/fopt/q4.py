"""Rebuild the qualified Q4 Gate/Up descriptors and narrow Down exchange."""
import argparse
from pathlib import Path
import re
import shutil
import tempfile

from a2a_rotate import rebuild_rotated
from direct_build import build
from i4_select import PACKED, lower


def main(rotate_down=False):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('image', 'task', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--task-out', type=Path, required=True)
    args = parser.parse_args()

    bucket = re.search(r'_t(128|1024|1536)(?:_|\.)', args.task.name)
    assert bucket, 'task filename must identify its bucket'
    rows = int(bucket[1])
    if rows == 128:
        assert '_class512_t128' in args.task.name, 'expected the 128-row class512 body'
    tokens = max(rows, 512)
    seq = ((2, 2, PACKED, 64, 1, 1, 1, 1) if tokens == 512 else
           (4, 2, PACKED, 256, tokens // 512, 1, 1, 1))
    limits = (4, 4, PACKED, 2 if tokens == 512 else 4, 8, 1, 1, 1)

    stock = args.image.read_bytes()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rebuilt = tmp / 'rebuilt.bin'
        build(args.image, args.task, rebuilt)
        assert rebuilt.read_bytes() == stock, 'raw task reproduction differs'

        image, source = lower(stock, args.task.read_text(), (seq, limits))
        native, task = tmp / 'native.bin', tmp / 'native.c'
        native.write_bytes(image)
        task.write_text(source)
        build(native, task, rebuilt)
        assert rebuilt.read_bytes() == image, 'native INT4 task reproduction differs'
        if rotate_down:
            rebuild_rotated(native, task, args.output, args.task_out)
        else:
            shutil.copyfile(native, args.output)
            shutil.copyfile(task, args.task_out)
