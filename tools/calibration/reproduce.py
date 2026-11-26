#!/usr/bin/env python3
"""Regenerate calibration tables into a new output directory."""

import argparse
import ctypes
import hashlib
import json
import mmap
import subprocess
from pathlib import Path

from activation import OCTOBER_CAPTURE_SHA256, captured_peaks, fp8_scales, int8s, load_maxima
from common import lz4_block, model_header

HERE = Path(__file__).resolve().parent
EMBEDDING = "model.language_model.embed_tokens.weight"


def tensor_span(base, header, name):
    start, end = header[name]["data_offsets"]
    return slice(base + start, base + end)


def compile_c(source, output, *flags):
    subprocess.run([
        "cc", "-O3", "-std=c11", "-fno-fast-math", "-ffp-contract=off", "-fopenmp",
        "-Wall", "-Wextra", "-Werror", *flags, str(Path(__file__).with_name(source)),
        "-lm", "-o", str(output),
    ], check=True)


def head_clip(weights, out, diagonal, threads):
    base, header = model_header(weights)
    offset = tensor_span(base, header, EMBEDDING).start
    binary = out / "head_clip"
    compile_c("head_clip.c", binary)
    subprocess.run([
        str(binary), str(weights), str(offset), str(diagonal),
        str(out / "head4_clip_indices.u8"), str(threads),
    ], check=True)
    binary.unlink()


def fp8_rows(weights, out, threads):
    base, header = model_header(weights)
    binary = out / "fp8_rows.so"
    compile_c("fp8_rows.c", binary, "-shared", "-fPIC")
    fit_rows = ctypes.CDLL(str(binary)).fit_rows
    fit_rows.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                         ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    fit_rows.restype = None
    payload, sites = bytearray(), []
    with weights.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        def tensor(name):
            return data[tensor_span(base, header, name)]

        for layer in range(47):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                prefix = f"model.language_model.layers.{layer}.mlp.{projection}"
                rows, packed_width = header[prefix + ".weight_packed"]["shape"]
                indices = ctypes.create_string_buffer(rows)
                fit_rows(tensor(prefix + ".weight_packed"), tensor(prefix + ".weight_scale"),
                         rows, 2 * packed_width, threads, indices)
                sites.append({"label": f"L{layer}.{projection}", "offset": len(payload), "rows": rows})
                payload.extend(indices.raw)
    raw = bytes(payload)
    compressed = lz4_block(raw)
    meta = {
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "sites": sites,
    }
    (out / "fp8_row_mse.lz4").write_bytes(compressed)
    (out / "fp8_row_mse.json").write_text(json.dumps(meta, indent=2) + "\n")
    binary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", choices=("head-clip", "head-balance", "fp8-rows", "activation", "q4-refit"))
    parser.add_argument("input", type=Path, help="NVFP4 model.safetensors")
    parser.add_argument("output", type=Path)
    parser.add_argument("--maxima", type=Path, help="recaptured maxima NPZ for activation or q4-refit")
    parser.add_argument("--moments", type=Path, help="activation moments JSON for head-balance")
    parser.add_argument("--diagonal", type=Path, help="head balance exponent file for head-clip")
    parser.add_argument("--threads", type=int, choices=range(1, 25), default=8, metavar="1..24")
    args = parser.parse_args()
    if args.table == "head-balance" and not args.moments:
        parser.error("head-balance requires --moments")
    shipped = Path(__file__).resolve().parents[2] / "src/host/engine"
    out = args.output.resolve()
    out.mkdir(parents=True)
    if args.table == "head-clip":
        head_clip(args.input, out, args.diagonal or shipped / "head4_balance_exponents.i8", args.threads)
    elif args.table == "fp8-rows":
        fp8_rows(args.input, out, args.threads)
    else:
        import torch

        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        with torch.inference_mode():
            capture_sha = hashlib.sha256(args.maxima.read_bytes()).hexdigest() if args.maxima else None
            if args.table == "activation":
                maxima = load_maxima(args.maxima or HERE / "activation-recorded.npz")
                fp8_scales(captured_peaks(maxima), out, capture_sha or OCTOBER_CAPTURE_SHA256)
                int8s(args.input, maxima, out)
            elif args.table == "q4-refit":
                from q4 import q4

                maxima = load_maxima(args.maxima) if args.maxima else None
                q4(args.input, shipped / "int8/q4cal_artifacts", out, maxima, capture_sha)
            else:
                from head_balance import head_balance

                head_balance(args.input, args.moments, out)


if __name__ == "__main__":
    main()
