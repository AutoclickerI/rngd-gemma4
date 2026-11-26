#!/usr/bin/env python3
"""Match captured task C to an Opt image by its TU records or compiled task bytes.

The capture shim records each GCC invocation without a kernel name. Complete
task bytes resolve ambiguous record matches and identify DMA-only kernels.
"""
import re, sys
from pathlib import Path


def arrays(path: Path) -> dict:
    src = path.read_text(errors="replace")
    return {m.group(1): bytes(int(x, 16) for x in re.findall(r"0x([0-9a-fA-F]{2})", m.group(2)))
            for m in re.finditer(r"const static uint8_t (\w+)\[\] = \{\n(.*?)\n\};", src, re.S)}


def records(path: Path) -> list:
    return [v for k, v in arrays(path).items() if k.startswith(("main_tu_", "sub_tu_"))]


def task_blob(capture: Path):
    """The capture's compiled task as it appears in an image (relink.py), or None."""
    elf = capture / "output.elf"
    if not elf.exists():
        return None
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fopt"))
    from relink import link
    try:
        blob, _, undefined, _ = link(str(elf))
    except ValueError:
        return None
    return None if undefined else blob


def capture_records(root):
    return {c: records(c) for c in sorted(root.glob("captures/*/effective.c"))}


def match_task(image, path, recs):
    match = [c for c, tus in recs.items() if tus and all(image.find(r) >= 0 for r in tus)]
    if not match:
        blobs = [(c, task_blob(c.parent)) for c, tus in recs.items() if not tus]
        match = [c for c, blob in blobs if blob and image.find(blob) >= 0]
    texts = {c.read_bytes() for c in match}
    if len(texts) > 1:
        # Different DMA/control programs can contain the same TU records.
        # Resolve that ambiguity by the complete compiled cluster task bytes.
        exact = [(c, task_blob(c.parent)) for c in match]
        exact = [c for c, blob in exact if blob and image.find(blob) >= 0]
        if exact:
            print(f"capture: {path}: complete task bytes narrow {len(match)} TU matches to {len(exact)}")
            match = exact
            texts = {c.read_bytes() for c in match}
    if not match:
        raise ValueError(f"capture: {path}: no captured task C matches its image (was it compiled in this build?)")
    if len(texts) > 1:
        raise ValueError(f"capture: {path}: {len(match)} different captured task Cs match its image: {[str(c) for c in match]}")
    return match[0]


def main():
    opt, root, paths = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    cache = opt / "furiosa-opt-gemma4"
    recs = capture_records(root)
    bad = 0
    for path in paths:
        image = (cache / f"{path}.bin").read_bytes()
        try:
            task = match_task(image, path, recs)
        except ValueError as error:
            print(error)
            bad = 1
        else:
            (cache / f"{path}.task.c").write_bytes(task.read_bytes())
            print(f"capture: {path}: task C from {task.parent.name} ({len(recs[task])} task records) -> {path}.task.c")
    sys.exit(bad)


if __name__ == "__main__":
    main()
