"""Export static FP8 scales and INT8-S factors from captured BF16 maxima."""

import json
import struct

from common import lz4_block, open_model

OCTOBER_CAPTURE_SHA256 = "7f5b6419eb58f71cc899f88015c938d71fd2e1e74aaa0ff42e7c3b320bafef83"

SITES = {
    "qkv": ("q_proj", "k_proj", "v_proj"),
    "o": ("o_proj",),
    "mlp": ("gate_proj", "up_proj"),
    "down": ("down_proj",),
}


def load_maxima(path):
    import numpy as np
    import torch

    with np.load(path, allow_pickle=False) as archive:
        return {key: torch.from_numpy(archive[key]).view(torch.bfloat16).float()
                for key in archive.files}


def save_maxima(path, maxima):
    import numpy as np
    import torch

    arrays = {}
    for key, value in maxima.items():
        bf16 = value.bfloat16()
        if not torch.equal(bf16.float(), value.float()) or not torch.isfinite(bf16).all():
            raise ValueError(f"{key} must contain finite BF16 maxima")
        arrays[key] = bf16.view(torch.uint16).numpy()
    np.savez_compressed(path, **arrays)


def fp8_scales(peaks, out, capture_sha=OCTOBER_CAPTURE_SHA256):
    scales, payload = [], bytearray()
    for (layer, site), peak in peaks.items():
        scale = peak / 256
        word = struct.pack("<f", scale)
        payload.extend(word)
        scales.append({"layer": layer, "site": site,
                       "scale_f32_hex": f"0x{int.from_bytes(word, 'little'):08x}",
                       "scale": scale, "calib_amax_max": peak})
    meta = {"meta": {"calibration": {"sha256": capture_sha}}, "scales": scales}
    (out / "fp8_scales.bin").write_bytes(payload)
    (out / "fp8_scales.json").write_text(json.dumps(meta, indent=2) + "\n")


def captured_peaks(maxima):
    return {(layer, site): float(maxima[f"L{layer}.{projections[0]}"].float().max())
            for layer in range(48) for site, projections in SITES.items()}


def expand_mlp(checkpoint, prefix, first, last, *, device="cpu"):
    import torch

    packed = checkpoint.get_slice(prefix + ".weight_packed")[first:last].to(device)
    blocks = checkpoint.get_slice(prefix + ".weight_scale")[first:last].to(device, torch.float32)
    lut = blocks.new_tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])
    codes = torch.stack((packed & 15, packed >> 4), dim=-1)
    values = lut[codes.long()].reshape(*blocks.shape, 16) * blocks.unsqueeze(-1)
    return values.flatten(1)


def column_peaks(checkpoint, layer, site, projection):
    if site in ("mlp", "down"):
        prefix = f"model.language_model.layers.{layer}.mlp.{projection}"
        rows, _ = checkpoint.get_slice(prefix + ".weight_packed").get_shape()
        peak = None
        for first in range(0, rows, 512):
            current = expand_mlp(checkpoint, prefix, first, first + 512).abs().amax(0)
            peak = current if peak is None else peak.maximum(current)
        return peak

    prefix = f"model.language_model.layers.{layer}.self_attn.{projection}"
    codes = checkpoint.get_tensor(prefix + ".weight").float()
    scale = checkpoint.get_tensor(prefix + ".weight_scale").float().reshape(-1, 1)
    return (codes * scale).bfloat16().float().abs().amax(0)


def int8s(weights, maxima, out):
    import torch

    payload = bytearray()
    with open_model(weights) as checkpoint:
        for layer in range(48):
            for site in ("qkv", "mlp", "down", "o"):
                projections = SITES[site]
                if site == "qkv" and layer % 6 == 5:
                    projections = projections[:2]
                w = torch.stack([column_peaks(checkpoint, layer, site, p) for p in projections]).amax(0)
                x = maxima[f"L{layer}.{projections[0]}"].float()
                factors = (x.clamp_min(1e-5).sqrt() / w.clamp_min(1e-5).sqrt()).clamp_min(1e-5)
                exponents = factors.log2().round()
                if not ((exponents >= -126) & (exponents <= 126)).all():
                    raise ValueError("INT8-S exponent outside the supported range")
                payload.extend(exponents.to(torch.int8).numpy().tobytes())
    if len(payload) != 1335296:
        raise ValueError("wrong exponent count")
    (out / "int8s_exp.lz4").write_bytes(lz4_block(bytes(payload), high_compression=True))
