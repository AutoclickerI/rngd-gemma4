"""Read the pinned model's safetensors header and write calibration artifacts."""

import hashlib
import json
import struct


def model_header(weights):
    with weights.open("rb") as stream:
        length, = struct.unpack("<Q", stream.read(8))
        header = stream.read(length)
    if hashlib.sha256(header).hexdigest() != (
        "23a75ce46a6f005f9e53d84b7b6b5d015cc9840620f1f44bdce7071603bd5d55"
    ):
        raise ValueError("expected the pinned NVFP4 safetensors header")
    return 8 + length, json.loads(header)


def open_model(weights):
    from safetensors import safe_open

    model_header(weights)
    return safe_open(weights, framework="pt")


def lz4_block(raw, *, high_compression=False):
    import ctypes
    import ctypes.util

    lib = ctypes.CDLL(ctypes.util.find_library("lz4"))
    lib.LZ4_compressBound.argtypes = [ctypes.c_int]
    args = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.LZ4_compress_default.argtypes = args
    lib.LZ4_compress_HC.argtypes = args + [ctypes.c_int]
    dst = ctypes.create_string_buffer(lib.LZ4_compressBound(len(raw)))
    if high_compression:
        size = lib.LZ4_compress_HC(raw, dst, len(raw), len(dst), 12)
    else:
        size = lib.LZ4_compress_default(raw, dst, len(raw), len(dst))
    if size <= 0:
        raise ValueError("LZ4 block compression failed")
    return dst.raw[:size]
