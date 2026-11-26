"""Fit Q4 Gate/Up smoothing exponents from captured maxima (or keep the shipped ones) and refit row ratios."""

import hashlib
import json
import lzma
import math
from pathlib import Path

from activation import column_peaks, expand_mlp
from common import model_header


def signs(seed):
    mask = (1 << 64) - 1
    values = []
    for _ in range(60):
        seed = (seed + 0x9E3779B97F4A7C15) & mask
        word = ((seed ^ (seed >> 30)) * 0xBF58476D1CE4E5B9) & mask
        word = ((word ^ (word >> 27)) * 0x94D049BB133111EB) & mask
        word ^= word >> 31
        values.extend(-1.0 if (word >> bit) & 1 else 1.0 for bit in range(64))
    return values


def rotation_matrices(torch):
    residues = {i*i % 59 for i in range(1, 59)}
    paley = torch.eye(60)
    paley[0, 1:], paley[1:, 0] = 1, -1
    for i in range(1, 60):
        for j in range(1, 60):
            if i != j:
                paley[i, j] = 1 if (j-i) % 59 in residues else -1
    sylvester = torch.tensor([[(-1.) ** ((i & j).bit_count() % 2)
                               for j in range(64)] for i in range(64)])
    return paley, sylvester


def ratios(torch, rotated):
    def codes(scale):
        return (rotated / (2 * scale)).floor().clamp_(-8, 7).mul_(2).add_(1)

    def error(code, scale):
        return (code * scale - rotated).square_().sum(1, keepdim=True)

    amax = rotated.abs().amax(1, keepdim=True)
    base = amax.clamp_min(1e-30) / 15
    best_error = torch.full_like(base, torch.inf)
    best_scale = torch.zeros_like(base)
    best_codes = torch.zeros_like(rotated)
    for i in range(13):
        scale = (round(1 - .05 * i, 2) * amax / 7.5).clamp_min_(1e-30) / 2
        q = codes(scale)
        mse = error(q, scale)
        better = mse < best_error
        best_scale = torch.where(better, scale, best_scale)
        best_codes = torch.where(better, q, best_codes)
        best_error = torch.minimum(mse, best_error)
    ls = (best_codes * rotated).sum(1, keepdim=True) / best_codes.square().sum(1, keepdim=True)
    stored, updated = (best_scale / base).half(), (ls / base).half()
    scale0, scale1 = base * stored.float(), base * updated.float()
    return torch.where(error(codes(scale1), scale1) < error(codes(scale0), scale0), updated, stored)


LAYERS = tuple(i for i in range(47) if i % 6 != 4)
PROJECTIONS = ("gate_proj", "up_proj")


def layout():
    index = {"schema": 1, "endianness": "little", "tensors": {},
             "raw_sha256": "", "raw_bytes_static": 0, "lzma_sha256": "",
             "recipe": "shared neutral original-channel power-of-two smoothing (g only); "
                       "weight-only MSE13 grid then single LS scale update, fp16 relative scale; "
                       "same odd code grid; original f routing",
             "derivation": "ratio chosen per output row by weight reconstruction SSE, "
                           "no benchmark input; fixed before scoring",
             "supplied_checkpoint": "gemma-4-12b-it-NVFP4-unsloth-b1f6497",
             "smooth_source_sha256": "85c615e7229e9e2f997d03a2a62fec9d0ab476d5568178257e0247bebb0ec1b5",
             "source_sha256": "99318f79b309258e94fd7f1ed2c51cba84cd94713f36abc938e1d35849911c35"}
    offset = 0
    for layer in LAYERS:
        for name, shape, dtype, size in (
            ("smooth_log2", [3840], "torch.int8", 3840),
            ("gate_proj.ratio", [15360, 1], "torch.float16", 30720),
            ("up_proj.ratio", [15360, 1], "torch.float16", 30720),
        ):
            index["tensors"][f"L{layer}.{name}"] = {
                "offset": offset, "bytes": size, "shape": shape, "dtype": dtype,
            }
            offset += size
    return index


def shipped_smoothing(folder):
    """Signed smoothing exponents per Q4 layer, decoded from a startup table."""
    import numpy as np

    index = json.loads((folder / "startup.json").read_text())["tensors"]
    raw = lzma.decompress((folder / "startup.bin.lzma").read_bytes(), format=lzma.FORMAT_ALONE)
    return [np.frombuffer(raw, np.int8, 3840, index[f"L{layer}.smooth_log2"]["offset"]) for layer in LAYERS]


def q4(weights, shipped, out, maxima=None, capture_sha=None):
    import torch
    from safetensors import safe_open

    model_header(weights)
    index = layout()
    index["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if capture_sha:
        index["smooth_source_sha256"] = capture_sha
    diagonals = shipped_smoothing(shipped) if maxima is None else None
    paley, sylvester = rotation_matrices(torch)
    payload = bytearray()
    with safe_open(weights, framework="pt") as checkpoint:
        for row, layer in enumerate(LAYERS):
            if maxima is None:
                exponent = torch.from_numpy(diagonals[row].copy())
            else:
                peaks = torch.stack([column_peaks(checkpoint, layer, "mlp", proj)
                                     for proj in PROJECTIONS]).amax(0).clamp_min(1e-5)
                active = maxima[f"L{layer}.gate_proj"].clamp_min(1e-5)
                exponent = (.5 * (active.log2() - peaks.log2())).round().to(torch.int8)
            payload.extend(exponent.numpy().tobytes())
            exponent = exponent.int()
            signed = torch.tensor(signs(layer * 8 + 2))
            for projection in PROJECTIONS:
                prefix = f"model.language_model.layers.{layer}.mlp.{projection}"
                for first in range(0, 15360, 256):
                    w = expand_mlp(checkpoint, prefix, first, first + 256)
                    x = (torch.ldexp(w, exponent) * signed).reshape(-1, 60, 64)
                    rotated = ((paley @ x) @ sylvester.T).reshape(-1, 3840) * (1 / math.sqrt(3840))
                    payload.extend(ratios(torch, rotated).numpy().tobytes())
    write_export(payload, index, out)


def write_export(payload, index, out):
    expected = sum(entry["bytes"] for entry in index["tensors"].values())
    if len(payload) != expected:
        raise ValueError("wrong Q4 startup tensor count")
    compressed = lzma.compress(payload, format=lzma.FORMAT_ALONE, preset=9 | lzma.PRESET_EXTREME)
    raw_hash = hashlib.sha256(payload).hexdigest()
    digest = hashlib.sha256(compressed).hexdigest()
    index.update(raw_sha256=raw_hash, raw_bytes_static=len(payload), lzma_sha256=digest)
    storage = {"schema": 1, "file": "startup.bin.lzma", "encoding": "lzma-alone",
               "sha256": digest, "bytes": len(compressed),
               "raw_sha256": raw_hash, "raw_bytes": len(payload)}
    (out / "startup.bin.lzma").write_bytes(compressed)
    (out / "startup.json").write_text(json.dumps(index, separators=(",", ":")) + "\n")
    (out / "storage.json").write_text(json.dumps(storage, separators=(",", ":")) + "\n")
