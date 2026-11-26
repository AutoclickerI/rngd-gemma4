#!/usr/bin/env python3
"""Remove TU clock-control pushes while preserving task C line positions."""
import argparse
from pathlib import Path

CLOCK_PUSH = b'push_core_interrupt_command(core_interrupt_command_control_tu_clock, NULL, 0);'


def clock_push(source: bytes) -> bytes:
    return source.replace(CLOCK_PUSH, b';')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_bytes(clock_push(args.input.read_bytes()))


if __name__ == '__main__':
    main()
