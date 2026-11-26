#!/usr/bin/env python3
"""Read the Opt 0.9.0 header fields used for STEP relinking."""
import struct


def header(b: bytes) -> dict:
    if b[:4] != b"FOPT":
        raise ValueError("not a FOPT image")
    at = 10  # magic, version, chip count and PE count

    def take(fmt):
        nonlocal at
        v = struct.unpack_from("<" + fmt, b, at)
        at += struct.calcsize("<" + fmt)
        return v[0] if len(v) == 1 else v

    def vec(item):
        return [item() for _ in range(take("Q"))]

    tasks = vec(lambda: vec(lambda: take("I")))
    weight = weight_offset = None
    if take("B"):
        kind = take("I")
        length = take("Q")
        weight_offset = at
        weight = kind, length
        at += length
    stack_count = take("Q")
    at += 12 * stack_count  # a memory kind and extent for each stack
    slots = vec(lambda: (take("I"), take("Q")))
    inputs, outputs = vec(lambda: take("I")), vec(lambda: take("I"))
    return {"tasks": tasks, "weight": weight, "weight_offset": weight_offset,
            "slots": slots, "inputs": inputs, "outputs": outputs}
