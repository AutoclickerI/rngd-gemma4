"""Read rows from the serving FRCZv1 containers."""
import lzma
from pathlib import Path
import struct
import sys

MAGIC = b"FRCZv1\0\0"
CRATE = "furiosa_opt_gemma4::"
PREFIX = "furiosa_opt_gemma4::prebuilt::"


def fnv1a64(b: bytes) -> int:
    h = 0xCBF29CE484222325
    for x in b:
        h = ((h ^ x) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def rows(path: Path):
    b = path.read_bytes()
    if b[:8] != MAGIC:
        sys.exit(f"{path}: not an FRCZv1 payload")
    raw_len, fnv = struct.unpack_from("<QQ", b, 8)
    raw = lzma.decompress(b[24:], format=lzma.FORMAT_ALONE)
    if len(raw) != raw_len or fnv1a64(raw) != fnv:
        sys.exit(f"{path}: rows do not match the header")
    at = 0
    def q():
        nonlocal at
        v, = struct.unpack_from("<Q", raw, at); at += 8; return v
    out = []
    for _ in range(q()):
        n = q(); p = raw[at:at + n].decode(); at += n
        metadata = [q() for _ in range(q())]
        n = q(); blob = raw[at:at + n]; at += n
        out.append((p, metadata, blob))
    return out
