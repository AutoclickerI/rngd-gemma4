#!/usr/bin/env python3
"""Fetch the pinned calibration rows from dataset repositories."""

import argparse
import hashlib
import io
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "selection.json"
WIKI = "wikitext-2-v1/train-00000-of-00001.parquet"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_selection():
    selection = json.loads(MANIFEST.read_text())
    for suite, src in selection["media"].items():
        rows = sorted(random.Random(f"20261010-{suite}").sample(range(20000), len(src["rows"])))
        if rows != src["rows"]:
            raise ValueError(f"selection does not match the frozen seed for {suite}")
    return selection


def media_path(suite, source, row):
    extension = "jpg" if source["column"] == "jpg" else "wav"
    return f"{suite}/{row:06d}.{extension}"


def wave16(raw):
    import numpy as np
    import soundfile as sf

    samples, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if rate != 16000:
        length = round(len(mono) * 16000 / rate)
        mono = np.interp(np.linspace(0, len(mono) - 1, length), np.arange(len(mono)), mono).astype("float32")
    output = io.BytesIO()
    sf.write(output, mono, 16000, format="WAV", subtype="PCM_16")
    return output.getvalue()


def raw_asset(asset, repo, revision):
    if asset["bytes"] is not None:
        return asset["bytes"]
    import fsspec

    # Streaming WebDataset/Parquet supplies bytes. Script-based datasets may
    # return a local or archive-member path from their pinned dataset builder.
    path = asset["path"]
    if path.startswith(("http://", "https://", "hf://")):
        raise ValueError(f"{repo}@{revision} returned an unbound external media URL")
    with fsspec.open(path, "rb") as stream:
        return stream.read()


def fetch_media(suite, source, output, cache):
    from datasets import Audio, Image, load_dataset

    repo, revision = source["dataset"], source["revision"]
    data = load_dataset(repo, source["config"], split=source["split"], revision=revision,
                        streaming=True, trust_remote_code=True, cache_dir=str(cache))
    column = source["column"]
    feature = Image(decode=False) if column == "jpg" else Audio(decode=False)
    data = data.cast_column(column, feature).select_columns([column])
    needed = iter(source["rows"])
    target = next(needed)
    (output / suite).mkdir(exist_ok=True)
    for index, row in enumerate(data):
        if index != target:
            continue
        raw = raw_asset(row[column], repo, revision)
        payload = raw if column == "jpg" else wave16(raw)
        path = media_path(suite, source, index)
        if sha256(payload) != source["sha256"][str(index)]:
            raise ValueError(f"file digest differs for {path}")
        (output / path).write_bytes(payload)
        target = next(needed, None)
        if target is None:
            return
    raise ValueError(f"{repo}/{source['split']} ended before row {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download

    selection = read_selection()
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "cache"
    wiki = selection["wikitext"]
    parquet = hf_hub_download(wiki["dataset"], WIKI, repo_type="dataset",
                              revision=wiki["revision"], cache_dir=cache)
    raw = Path(parquet).read_bytes()
    if sha256(raw) != wiki["sha256"]:
        raise ValueError("WikiText-2 train parquet digest differs")
    (args.output / "wikitext.parquet").write_bytes(raw)
    for suite, source in selection["media"].items():
        fetch_media(suite, source, args.output, cache)
        print(f"{suite}: {len(source['rows'])} rows", flush=True)
    (args.output / "manifest.json").write_text(json.dumps(selection, indent=2) + "\n")


if __name__ == "__main__":
    main()
    # Pinned streaming readers can leave native threads alive during Python teardown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
