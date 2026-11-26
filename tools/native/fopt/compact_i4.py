#!/usr/bin/env python3
"""Lower packed W4 descriptors and compile the PE command code for size."""
import argparse
from pathlib import Path

from direct_build import build
from i4_select import lower

PRAGMA = '#pragma GCC optimize ("Os")\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('stock', 'task', 'out'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--task-out', type=Path, required=True)
    args = parser.parse_args()
    _, source = lower(args.stock.read_bytes(), args.task.read_text())
    assert PRAGMA not in source
    args.task_out.write_text(PRAGMA + source)
    build(args.stock, args.task_out, args.out)


if __name__ == '__main__':
    main()
