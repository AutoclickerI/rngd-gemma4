#!/usr/bin/env python3
"""Keep private Opt 0.9.0 image caches valid across axis-geometry edits.

Compiler keys omit axis sizes. Geometry records invalidate affected keys while
ordinary source changes remain the compiler's responsibility. Use a private
output directory for the cache.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, tempfile
from pathlib import Path


SIZES = ".axes-sizes.json"

def source_files(src: Path):
    """Every Rust source in the serving crate."""
    src = src.resolve()
    return sorted((os.path.relpath(f, src), f) for f in src.rglob("*.rs"))


def axis_sizes(src: Path):
    """{"file::Name": size} for every literal `axes![Name = N, ...]`; names of macro-built axes apart."""
    sizes, built = {}, []
    for rel, f in source_files(src):
        text = re.sub(r"//[^\n]*", "", f.read_text())
        for body in re.findall(r"axes!\s*\[(.*?)\]", text, re.S):
            if "$" in body: built.append(rel); continue
            for name, n in re.findall(r"([A-Za-z_]\w*)\s*=\s*(\d[\d_]*)", body):
                sizes[f"{rel}::{name}"] = int(n.replace("_", ""))
    return sizes, built

SET_RECORD = ".axes-set.json"

def axis_set(src: Path):
    """sha256 of the crate's set of axis declarations, sizes excluded (see the module docstring), with counts."""
    texts = {rel: re.sub(r"//[^\n]*", "", f.read_text()) for rel, f in source_files(src)}
    names, macros, impls = [], set(), []
    for rel, text in texts.items():
        for body in re.findall(r"axes!\s*\[(.*?)\]", text, re.S):
            if "$" in body:
                macros.update(re.findall(r"macro_rules!\s*([A-Za-z_]\w*)", text)); macros.add(rel)
            else:
                names += [f"{rel}::{name}" for name in re.findall(r"([A-Za-z_]\w*)\s*=\s*\d", body)]
        impls += [f"{rel}::" + re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if "AxisName" in line]
    files = sorted(rel for rel, text in texts.items()
                   if rel in macros
                   or any(re.search(r"\b" + re.escape(m) + r"!\s*[\(\[\{]", text) for m in macros if "." not in m))
    parts = sorted(names) + [f"{rel}::" + re.sub(r"\s+", " ", texts[rel]) for rel in files] + sorted(impls)
    return hashlib.sha256("\n".join(parts).encode()).hexdigest(), len(names), files



def axes_record(src: Path, opt: Path):
    sizes, _ = axis_sizes(src)
    (opt / SIZES).write_text(json.dumps(sizes, indent=0, sort_keys=True))
    print(f"recorded {len(sizes)} axis sizes in {opt / SIZES}")
    now, n_axes, files = axis_set(src)
    rec = opt / SET_RECORD
    record = {"set_sha256": now, "literal_axes": n_axes, "macro_files": files,
              "key_strategy": "conservative-v2"}
    record.update(axis_parts(src)); record["literal_rule"] = LITERAL_RULE
    rec.write_text(json.dumps(record, indent=0))
    print(f"recorded the axis set ({n_axes} literal axes, macro files {files}) in {rec}")


LITERAL_RULE = "impls-v3"

_NOT_IMPL = re.compile(r"\bimpl\b|\bAxisName\s+for\b")
_FN_BOUND = re.compile(r"\bfn\s+\w+\s*<")
_PARAM_BOUND = re.compile(r"^\s*(?:where\s+)?[\w']+\s*:[^{};]*\bAxisName\b[^{};]*,?\s*$")

def is_axis_bound(line: str) -> bool:
    """True only for a line that is clearly a generic BOUND on AxisName (`T: AxisName + X,`, `where T: AxisName`,
    `fn f<T: AxisName, ..>(`): it neither declares nor removes an `impl AxisName for ..`. Anything else naming
    AxisName (impl headers, `AxisName for X`, `impl<T: AxisName> Tr for T`, traits, aliases) cannot be told apart
    from a declaration and stays in the record."""
    if "AxisName" not in line or _NOT_IMPL.search(line): return False
    return bool(_FN_BOUND.search(line) or _PARAM_BOUND.match(line))

def axis_parts(src: Path):
    """The axis record split by cause: the literal axes and AxisName lines (any change there displaces every key),
    and the comment-stripped text of each macro-axes file (a change there displaces only its dependents).
    Generic bounds do not change AxisName implementations."""
    texts = {rel: re.sub(r"//[^\n]*", "", f.read_text()) for rel, f in source_files(src)}
    _, _, files = axis_set(src)
    names, impls = [], []
    for rel, text in texts.items():
        for body in re.findall(r"axes!\s*\[(.*?)\]", text, re.S):
            if "$" not in body:
                names += [f"{rel}::{name}" for name in re.findall(r"([A-Za-z_]\w*)\s*=\s*\d", body)]
        impls += [f"{rel}::" + re.sub(r"\s+", " ", line).strip() for line in text.splitlines()
                  if "AxisName" in line and not is_axis_bound(line)]
    literal = hashlib.sha256("\n".join(sorted(names) + sorted(impls)).encode()).hexdigest()
    return {"literal_sha256": literal,
            "macro_file_sha256": {rel: hashlib.sha256(re.sub(r"\s+", " ", texts[rel]).encode()).hexdigest() for rel in files}}

def module_map(src: Path):
    """{rel path: module path} for the crate's files, honouring `#[path = ".."] mod x;` declarations."""
    texts = {rel: re.sub(r"//[^\n]*", "", f.read_text()) for rel, f in source_files(src)}
    def by_location(rel):
        parts = rel[:-3].split("/")
        if parts[-1] == "mod": parts = parts[:-1]
        return "" if parts in (["lib"], ["main"]) else "::".join(parts)
    mods = {rel: by_location(rel) for rel in texts}
    for rel, text in texts.items():
        for p, name in re.findall(r'#\[path\s*=\s*"([^"]+)"\]\s*(?:#\[[^\]]*\]\s*)*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;', text):
            child = os.path.normpath(os.path.join(os.path.dirname(rel), p))
            if child in mods: mods[child] = (mods[rel] + "::" if mods[rel] else "") + name
    return texts, mods

def macro_bodies(text):
    """{macro name: body text} of the `macro_rules!` definitions in text; a definition that does not open with `{`
    maps to None (its body is not extracted; macro_dependents then treats it as dependent)."""
    out = {}
    for m in re.finditer(r"macro_rules!\s*(\w+)\s*(\{)?", text):
        if not m.group(2):
            out[m.group(1)] = None; continue
        i, depth = m.end(), 1
        while i < len(text) and depth:
            depth += (text[i] == "{") - (text[i] == "}"); i += 1
        out[m.group(1)] = text[m.end():i]
    return out

def macro_dependents(src: Path, changed):
    """Module paths whose kernels may use an axis declared in the changed macro-axes files. CARRIERS can hand those
    axes on: the changed files' modules and, transitively, modules re-exporting from a carrier (`pub use` /
    `pub type`). A macro is dependent when it is defined in a changed file or its body invokes a dependent macro or
    names a carrier (a body that is not `{..}`-delimited counts as dependent in a carrier file). Dependent modules:
    the carriers, every module that names a carrier (`leaf` with its parent in the text, or a file below the parent:
    `super::` paths) and every module that invokes a dependent macro (`dsl_prefill_edge!` from prefill.rs expands in
    prompt_blocks.rs). A module that only names a dependent module (a host module launching its kernels) passes
    nothing on."""
    texts, mods = module_map(src)
    decl = re.compile(r"(?:#\[[^\]]*\]\s*)*(?:pub(?:\([^)]*\))?\s+)?mod\s+\w+\s*;")
    def names(text, rel, m):
        text = decl.sub("", text)   # declaring a child module is not using it
        parts = m.split("::")
        if not re.search(rf"\b{re.escape(parts[-1])}\b", text): return False
        if len(parts) == 1: return True
        return bool(re.search(rf"\b{re.escape(parts[-2])}\b", text)) or mods[rel].startswith("::".join(parts[:-1]))
    bodies = {rel: macro_bodies(text) for rel, text in texts.items()}
    carriers = {mods[r] for r in changed}; found = set(carriers); frontier = set(carriers)
    tainted = {n for r in changed for n in bodies[r]}
    while True:
        invokes = re.compile(r"\b(" + "|".join(map(re.escape, sorted(tainted))) + r")\s*!") if tainted else None
        new = {mods[rel] for rel, text in texts.items() if mods[rel] not in found
               and (any(names(text, rel, m) for m in frontier) or (invokes and invokes.search(text)))}
        found |= new
        leaves = {m.split("::")[-1] for m in carriers if m}
        reexport = re.compile(r"\bpub(?:\([^)]*\))?\s+(?:use|type)\b[^;]*\b(" + "|".join(map(re.escape, leaves)) + r")\b") if leaves else None
        frontier = {m for m in new if reexport and any(reexport.search(texts[r]) for r in texts if mods[r] == m)}
        carriers |= frontier
        more = {n for rel, defs in bodies.items() for n, body in defs.items() if n not in tainted and (
                    (body is None and mods[rel] in carriers)
                    or (body is not None and ((invokes and invokes.search(body)) or any(names(body, rel, m) for m in carriers))))}
        tainted |= more
        if not new and not more:
            return found

def conservative_check(src: Path, opt: Path):
    """Invalidate opaque Opt keys when the recorded axis geometry changes."""
    opt.mkdir(parents=True, exist_ok=True)
    sizes, _ = axis_sizes(src)
    now, _, _ = axis_set(src)
    old_set = json.loads((opt / SET_RECORD).read_text()) if (opt / SET_RECORD).exists() else {}
    old_sizes = json.loads((opt / SIZES).read_text()) if (opt / SIZES).exists() else None
    current = old_set.get("key_strategy") == "conservative-v2" and old_set.get("literal_rule") == LITERAL_RULE
    if current and old_set.get("set_sha256") == now and old_sizes == sizes:
        print("opaque cache keys: axis set and sizes match; compiler checks all source dependencies")
        return
    keys = sorted(set(opt.glob("*/*.hash")) | set(opt.glob("*.hash")))
    parts = axis_parts(src)
    if (current and old_sizes == sizes and keys
            and old_set.get("literal_sha256") == parts["literal_sha256"]
            and old_set.get("macro_file_sha256") == parts["macro_file_sha256"]):
        # Only generic bounds or unrelated lines changed.
        print("opaque cache keys: AxisName impls, literal axes, sizes and macro files match (only AxisName bounds or "
              "other lines differ); compiler checks all source dependencies")
        return
    if (current and old_sizes == sizes and keys
            and old_set.get("literal_sha256") == parts["literal_sha256"]
            and sorted(old_set.get("macro_file_sha256", {})) == sorted(parts["macro_file_sha256"])):
        # Only the text of macro-axes files changed (same files, same literal axes and sizes). The compiler re-checks
        # every key's code dependencies, and omits axis sizes, so a size edit inside one of these
        # files must not reuse the images of the kernels that use its axes: evict exactly the kernels of the modules
        # that depend on the changed files (and every generic kernel); everything else stays.
        changed = sorted(f for f, h in parts["macro_file_sha256"].items() if old_set["macro_file_sha256"][f] != h)
        modules = macro_dependents(src, changed)
        owners = sorted(set(module_map(src)[1].values()), key=len, reverse=True)
        def hit(key):
            path = key.name[:-len(".hash")].split("::", 1)[-1]
            # A kernel belongs to the file that defines it: the longest file module prefixing its path.
            return "[" in path or next((m for m in owners if path.startswith(m + "::")), "") in modules
        evict = [k for k in keys if hit(k)]
        _preserve(opt, evict)
        print(f"opaque cache keys: macro-axes files changed {changed}; evicted the {len(evict)} keys of their "
              f"{len(modules)} dependent modules (+ generic kernels), kept {len(keys) - len(evict)}")
        return
    if not keys:
        print("opaque cache keys: no keys to reuse"); return
    print("opaque cache keys: axis record absent or changed (literal axes, axis sizes or the set of macro-axes "
          "files); every kernel recompiles")
    _preserve(opt, keys)

def _preserve(opt: Path, keys):
    """Move keys out of the cache (the build then recompiles their kernels), kept beside it for diagnosis."""
    if not keys:
        return
    saved = Path(tempfile.mkdtemp(prefix=opt.name + ".axis-keys-", dir=opt.parent))
    for key in keys:
        dest = saved / key.relative_to(opt)
        dest.parent.mkdir(parents=True, exist_ok=True)
        key.rename(dest)
    print(f"opaque cache keys: preserved {len(keys)} displaced keys in {saved}; their kernels recompile")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    actions = ap.add_mutually_exclusive_group(required=True)
    actions.add_argument("--axes-record", nargs=2, metavar=("SRC", "OPT"))
    actions.add_argument("--conservative-check", nargs=2, metavar=("SRC", "OPT"))
    a = ap.parse_args()
    if a.conservative_check:
        conservative_check(*map(Path, a.conservative_check))
    else:
        axes_record(*map(Path, a.axes_record))


if __name__ == "__main__":
    main()
