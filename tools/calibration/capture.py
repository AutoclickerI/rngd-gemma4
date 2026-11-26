#!/usr/bin/env python3
"""Capture teacher continuations, activation maxima and head moments."""

import argparse
import hashlib
import json
import struct
from pathlib import Path

from activation import load_maxima, save_maxima
from corpus import text_windows
from fetch import media_path, read_selection, sha256

HERE = Path(__file__).resolve().parent
BF16_SHA256 = "5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d"
NVFP4_SHA256 = "7c2ee23298e7c3a9247e8947597dca5a38f8b791a0322487466d2bfad8ce704b"
BF16_CONFIG = {
    "config.json": "478c46e8d2c52d5c2d85bf67e3b3e8c90e7c9d91086cee27e3c267907e936bd9",
    "tokenizer.json": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
    "tokenizer_config.json": "a62f4e85a47c0c136edaaa3a4f591fd6783717299a9def47e5ad03a49f6a5eb9",
    "chat_template.jinja": "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4",
}
NVFP4_CONFIG = {
    "config.json": "bcc0ec0398a9dd0b09586f835f17c05ed2cce99d958dd59ef629ce77e618ee49",
    "tokenizer.json": "adbaa8175acf7609b4359724f40eff359ec4fac1a8647eeb99d4422be708e1cf",
}
SUPPRESSED_TOKENS = ("<eos>", "<turn|>", "<|think|>", "<|channel>", "<channel|>", "<image|>", "<audio|>")


def verify_weights(path, mode):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    expected = BF16_SHA256 if mode == "activation" else NVFP4_SHA256
    if digest != expected:
        raise ValueError(f"teacher SHA-256 differs: {digest}")
    configuration = BF16_CONFIG if mode == "activation" else NVFP4_CONFIG
    for name, expected in configuration.items():
        if sha256((path.parent / name).read_bytes()) != expected:
            raise ValueError(f"teacher configuration digest differs for {name}")
    print(f"Verified teacher SHA-256 {digest}", flush=True)


def media_prompt(checkpoint, tokenizer, path, raw):
    import io

    import torch
    from torch.nn import functional as F

    def param(prefix, name):
        return checkpoint.get_tensor(f"model.{prefix}.{name}").float()

    def rn(x):
        return (x / torch.sqrt(x.square().mean(-1, keepdim=True) + 1e-6)).bfloat16().float()

    if path.endswith(".jpg"):
        from PIL import Image

        with Image.open(io.BytesIO(raw)) as loaded:
            image = loaded.convert("RGB")
        width, height = image.size
        factor = (280 * 48**2 / (width * height)) ** .5
        w, h = int(width * factor / 48), int(height * factor / 48)
        if not w:
            w, h = 1, min(height // width, 280)
        if not h:
            w, h = min(width // height, 280), 1
        image = image.resize((48 * w, 48 * h), Image.Resampling.BICUBIC)
        pixel = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        x = (pixel.float() / 255.).reshape(h, 48, w, 48, 3).transpose(1, 2).reshape(-1, 6912)
        x = x.bfloat16().float()

        def norm(x, name):
            return F.layer_norm(x, (x.shape[-1],),
                                param("vision_embedder", name + ".weight"),
                                param("vision_embedder", name + ".bias"), 1e-6).bfloat16().float()

        x = norm(x, "patch_ln1")
        x = F.linear(x, param("vision_embedder", "patch_dense.weight"),
                     param("vision_embedder", "patch_dense.bias")).bfloat16().float()
        x = norm(x, "patch_ln2")
        pos = param("vision_embedder", "pos_embedding").bfloat16()
        positions = (pos[:w, 0][None, :] + pos[:h, 1][:, None]).reshape_as(x)
        x = norm((x + positions.float()).bfloat16().float(), "pos_norm")
        soft = F.linear(rn(x), param("embed_vision", "embedding_projection.weight")).bfloat16()
        part = {"type": "image_url", "image_url": {"url": "image"}}
        prompt, marker, opening, closing = "Describe this image in one sentence.", 258880, 255999, 258882
    else:
        import soundfile as sf

        samples, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        if rate != 16000 or samples.shape[1] != 1:
            raise ValueError("expected the selected 16 kHz mono WAV")
        x = torch.from_numpy(samples[:, 0])
        x = F.pad(x, (0, -len(x) % 640)).reshape(-1, 640).bfloat16().float()
        soft = F.linear(rn(x), param("embed_audio", "embedding_projection.weight")).bfloat16()
        part = {"type": "input_audio", "input_audio": {"data": "audio", "format": "wav"}}
        prompt, marker, opening, closing = "Transcribe this audio.", 258881, 256000, 258883
    messages = [{"role": "user", "content": [part, {"type": "text", "text": prompt}]}]
    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ids = tokenizer(rendered, add_special_tokens=False).input_ids
    if ids.count(marker) != 1:
        raise ValueError("chat template must produce one media marker")
    at = ids.index(marker)
    ids[at:at+1] = [opening, *([marker] * len(soft)), closing]
    return ids, (at + 1, soft), None


def media_inputs(folder, expected):
    fetched = json.loads((folder / "manifest.json").read_text())
    if fetched != expected:
        raise ValueError("fetched selection differs from the pinned manifest")
    for suite, source in expected["media"].items():
        for row in source["rows"]:
            path = media_path(suite, source, row)
            raw = (folder / path).read_bytes()
            if sha256(raw) != source["sha256"][str(row)]:
                raise ValueError(f"file digest differs for {path}")
            yield path, raw


def prompt_sha256(ids):
    return sha256(struct.pack(f"<{len(ids)}I", *ids))


def q4_prompts(windows, continuations_path):
    import torch

    continuations = json.loads(continuations_path.read_text())
    for ids, entry in zip(windows, continuations, strict=True):
        if prompt_sha256(ids) != entry["prompt_sha256"]:
            raise ValueError("the Q4 token window differs from the original selection")
        continuation = entry["continuation"]
        if len(continuation) != 16:
            raise ValueError("expected 16 Q4 continuation tokens")
        sampled = torch.linspace(0, len(ids) - 1, 64).round().long().tolist()
        sampled.extend(range(len(ids), len(ids) + len(continuation)))
        yield ids + continuation, None, sampled


def capture_continuations(checkpoint, config, windows, maxima, tokenizer_path, device):
    from tokenizers import Tokenizer

    from teacher import greedy

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    suppressed = [tokenizer.token_to_id(name) for name in SUPPRESSED_TOKENS]
    if None in suppressed:
        raise ValueError("tokenizer is missing a suppressed token")
    continuations = greedy(checkpoint, config, windows, maxima, suppressed, device)
    return [{"prompt_sha256": prompt_sha256(ids), "continuation": tokens[:16]}
            for ids, tokens in zip(windows, continuations, strict=True)]


def save_moments(path, rows):
    import torch

    second_moment = torch.stack([x.double().square_().mean(0) for x in rows]).mean(0)
    path.write_text(json.dumps({"second_moment": second_moment.tolist()},
                              separators=(",", ":"), allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", choices=("activation", "continuations", "q4"))
    parser.add_argument("teacher", type=Path, help="local model snapshot directory")
    parser.add_argument("input", type=Path, help="fetched dataset directory or WikiText-2 train parquet")
    parser.add_argument("output", type=Path, help="output NPZ, or JSON for continuations")
    parser.add_argument("--device", default="cpu", help="cpu or cuda[:index]")
    parser.add_argument("--maxima", type=Path, default=HERE / "activation-recorded.npz",
                        help="input activation NPZ for continuations and q4")
    parser.add_argument("--continuations", type=Path, default=HERE / "q4-train.json",
                        help="input continuation JSON for q4 replay")
    parser.add_argument("--threads", type=int, choices=range(1, 25), default=12, metavar="1..24")
    args = parser.parse_args()

    import torch
    from safetensors import safe_open

    from teacher import PREFIX, teacher

    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("--device must be cpu or cuda[:index]")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    path = args.teacher / "model.safetensors"
    tokenizer_path = args.teacher / "tokenizer.json"
    verify_weights(path, args.capture)
    config = json.loads((args.teacher / "config.json").read_text())["text_config"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.capture != "activation":
        windows = text_windows(args.input, tokenizer_path, 1000, 16, 512)
        scales = load_maxima(args.maxima)
    maxima = {}
    if args.capture == "q4":
        from q4 import LAYERS

        needed = set(LAYERS)
    else:
        needed = set(range(len(config["layer_types"])))

    def observe(layer, site, x):
        if layer not in needed:
            return
        key = f"L{layer}.{site}"
        current = x.abs().amax(0)
        maxima[key] = current if key not in maxima else torch.maximum(maxima[key], current)

    with torch.inference_mode(), safe_open(path, framework="pt", device="cpu") as checkpoint:
        if args.capture == "continuations":
            entries = capture_continuations(checkpoint, config, windows, scales, tokenizer_path, device)
            args.output.write_text(json.dumps(entries, separators=(",", ":")) + "\n")
            return
        elif args.capture == "activation":
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.teacher, local_files_only=True)
            parquet = args.input / "wikitext.parquet"
            prompts = [(ids, None, None) for ids in text_windows(parquet, tokenizer_path, 20000, 8, 1024)]
            prompts.extend(media_prompt(checkpoint, tokenizer, path, raw)
                           for path, raw in media_inputs(args.input, read_selection()))
            scales = None
        else:
            prompts = list(q4_prompts(windows, args.continuations))
        print(f"{args.capture}: {len(prompts)} prompts, {sum(len(ids) for ids, _, _ in prompts)} tokens",
              flush=True)
        embedding = checkpoint.get_tensor(PREFIX + "embed_tokens.weight").to(device)
        rows = teacher(checkpoint, config, prompts, args.capture, embedding, observe=observe, maxima=scales)
    save_maxima(args.output, {key: value.cpu() for key, value in maxima.items()})
    if args.capture == "activation":
        save_moments(args.output.with_suffix(".moments.json"), rows)


if __name__ == "__main__":
    main()
