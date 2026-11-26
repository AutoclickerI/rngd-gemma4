#!/usr/bin/env python3
"""Track each cached image's Opt driver and compiler config.

Opt's cache key omits both. Build gates evict entries whose recorded inputs
differ, and record commands use the compiler log to bind freshly built images.
"""
import hashlib, json, re, shutil, sys, time
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(opt: Path) -> dict:
    f = opt / ".provenance.json"
    return json.loads(f.read_text()) if f.exists() else {}


def save(opt: Path, rec: dict):
    (opt / ".provenance.json").write_text(json.dumps(dict(sorted(rec.items())), indent=1) + "\n")


def entries(opt: Path):
    return sorted(p.name[:-5] for p in opt.glob("*/*.hash"))


def load_knobs(opt: Path) -> dict:
    f = opt / ".knobs.json"
    return json.loads(f.read_text()) if f.exists() else {}


def save_knobs(opt: Path, rec: dict):
    (opt / ".knobs.json").write_text(json.dumps(dict(sorted(rec.items())), indent=1) + "\n")


def evict(opt: Path, k: str, out: Path):
    """Moves kernel k's entry (.bin, .hash, .edf) out of the cache into out/<subdir>/."""
    for f in [f for d in opt.iterdir() if d.is_dir() for f in d.iterdir()]:
        if f.name in (f"{k}.bin", f"{k}.hash", f"{k}.edf"):
            dest = out / f.parent.name
            dest.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), dest / f.name)


def knobs_cmd(cmd: str, opt: Path, args: list):
    config = None if args[0] == "-" else sha(Path(args[0]))
    if cmd == "knobgate":
        carried = set(args[1:])
    else:
        log, carried = Path(args[1]).read_text(errors="replace"), set(args[2:])
    rec, prov = load_knobs(opt), load(opt)
    out = Path(f"{opt}.evicted") / time.strftime("%Y%m%d-%H%M%S")
    want = lambda k: config if k in carried else None
    if cmd == "knobgate":
        moved = [k for k in entries(opt) if rec.get(k) != want(k)]
        for k in moved:
            evict(opt, k, out)
            rec.pop(k, None)
            prov.pop(k, None)
        print(f"knob gate: carried config {config[:16] if config else 'none (driver defaults)'}; "
              + (f"moved {len(moved)} entries compiled otherwise to {out}" if moved else "every cached entry matches"))
    else:
        compiled = re.findall(r"^\s*Compiling (furiosa_opt_gemma4::\S+)\s*$", log, re.M)
        stray = []
        for k in compiled:
            if want(k) is not None:
                rec[k] = config
            elif config is not None:
                evict(opt, k, out)
                rec.pop(k, None)
                prov.pop(k, None)
                stray.append(k)
            else:
                rec.pop(k, None)
        print(f"knob record: {len(compiled) - len(stray)} kernels compiled with "
              + (f"carried config {config[:16]}" if config else "the driver defaults")
              + (f"; {len(stray)} uncarried kernels compiled under the config moved to {out}: {stray}" if stray else ""))
    save_knobs(opt, rec)
    save(opt, prov)


def main():
    cmd, opt = sys.argv[1], Path(sys.argv[2])
    if cmd in ("knobgate", "knobrecord"):
        return knobs_cmd(cmd, opt, sys.argv[3:])
    rec = load(opt)
    driver = sha(Path(sys.argv[3]))
    if cmd == "gate":
        out = Path(f"{opt}.evicted") / time.strftime("%Y%m%d-%H%M%S")
        moved = 0
        for k in entries(opt):
            if rec.get(k) == driver:
                continue
            for f in [f for d in opt.iterdir() if d.is_dir() for f in d.iterdir()]:
                if f.name in (f"{k}.bin", f"{k}.hash", f"{k}.edf"):
                    dest = out / f.parent.name
                    dest.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(f), dest / f.name)
            rec.pop(k, None)
            moved += 1
        save(opt, rec)
        print(f"provenance gate: driver {driver[:16]}; moved {moved} entries it did not produce to {out}" if moved
              else f"provenance gate: driver {driver[:16]}; every cached entry is its own")
    elif cmd == "record":
        log = Path(sys.argv[4]).read_text(errors="replace")
        compiled = re.findall(r"^\s*Compiling (furiosa_opt_gemma4::\S+)\s*$", log, re.M)
        # `cargo furiosa-opt compile` (build.sh's knob mode) prints `Compiling [i/n] <path after the crate>`.
        compiled += ["furiosa_opt_gemma4::" + k for k in re.findall(r"^\s*Compiling \[\d+/\d+\] ([A-Za-z_]\w*(?:::\w+)+)\s*$", log, re.M)]
        for k in compiled:
            rec[k] = driver
        save(opt, rec)
        print(f"provenance: {len(compiled)} kernels compiled by driver {driver[:16]}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
