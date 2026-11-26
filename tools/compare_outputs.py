#!/usr/bin/env python3
"""Compare measured texts from a public replay with the bundled reference or other runs."""
import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile

REFERENCE = Path(__file__).with_name("reference_outputs.json")


def read_run(run):
    failed = 0
    outputs = defaultdict(set)
    plan_bytes = (run / "replay/replay-plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    planned = [(index, plan["cases"][case]["id"]) for index, case in enumerate(plan["measurement_order"], 1)]
    measured = []
    with (run / "replay/records.jsonl").open(encoding="utf-8") as stream:
        marker, _, payload = next(stream, "").partition(" ")
        if marker != "REPLAY_START":
            raise ValueError("records.jsonl must start with REPLAY_START")
        start = json.loads(payload)
        if start["plan_sha256"] != hashlib.sha256(plan_bytes).hexdigest():
            raise ValueError("records.jsonl was written for a different plan")
        mode = start["mode"]
        for line in stream:
            marker, _, payload = line.partition(" ")
            if marker != "REPLAY_REQUEST":
                continue
            event = json.loads(payload)
            if event.get("phase") != "measurement":
                continue
            record = event["record"]
            measured.append((record["request_index"], record["id"]))
            case = record["id"].split(".r", 1)[0]
            output = record.get("output")
            failed += not (record.get("passed") and isinstance(output, str))
            outputs[case].add(output if isinstance(output, str) else None)
    if sorted(measured) != planned:
        raise ValueError("measured requests differ from the plan (missing, repeated or unplanned)")
    return mode, outputs, failed, len(measured), plan, start


def coverage_delta(expected, actual):
    return f"(missing {sorted(expected - actual)}, extra {sorted(actual - expected)})"


def write_reference(parser, path, runs):
    reference, provenance = {}, defaultdict(list)
    for run, mode, outputs, failed, total, plan, start in runs:
        if failed or not total:
            parser.error(f"{run}: expected successful measured requests, found {failed} failed of {total}")
        server = start.get("server_sha256")
        if not isinstance(server, str) or not re.fullmatch(r"[0-9a-f]{64}", server):
            parser.error(f"{run}: invalid REPLAY_START server_sha256")
        if mode not in ("accuracy", "performance") or plan.get("mode") != mode:
            parser.error(f"{run}: invalid replay mode or mode differs from its plan")
        if plan.get("server_sha256") != server:
            parser.error(f"{run}: replay server hash differs from its plan")
        ambiguous = sorted(case for case, texts in outputs.items() if len(texts) != 1)
        if ambiguous:
            parser.error(f"{run}: cases with more than one output: {', '.join(ambiguous)}")
        texts = {case: next(iter(values)) for case, values in outputs.items()}
        expected = reference.setdefault(mode, texts)
        if texts.keys() != expected.keys():
            parser.error(f"{run}: different {mode} case coverage " +
                         coverage_delta(expected.keys(), texts.keys()))
        differing = sorted(case for case, output in texts.items() if expected[case] != output)
        if differing:
            parser.error(f"{run}: cases differing from earlier {mode} runs: {', '.join(differing)}")
        provenance[mode].append({"run": run.name, "plan_sha256": start["plan_sha256"], "server_sha256": server})
        print(f"{run.name}: {total} {mode} requests, {len(texts)} cases, one text per case")

    reference["provenance"] = provenance
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent) as stream:
            json.dump(reference, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            Path(stream.name).replace(path)
    except OSError as error:
        parser.error(f"{path}: {error}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+", metavar="RUN",
                        help="one run to check against the reference, or multiple runs of the same mode to compare")
    parser.add_argument("--write", type=Path, metavar="FILE",
                        help="write reference texts and provenance, comparing runs within each mode")
    args = parser.parse_args()

    def validated_runs():
        for run in args.runs:
            try:
                yield run, *read_run(run)
            except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as error:
                parser.error(f"{run}: {error}")

    if args.write is not None:
        return write_reference(parser, args.write, validated_runs())

    combined, failed, total = defaultdict(set), 0, 0
    mode, coverage = None, None
    reference = None
    for run, run_mode, outputs, run_failed, run_total, _, _ in validated_runs():
        if mode is None:
            mode = run_mode
        elif mode != run_mode:
            parser.error(f"{run} has a different replay mode from {args.runs[0]}")
        cases = set(outputs)
        if coverage is None:
            if len(args.runs) == 1:
                try:
                    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))[mode]
                except (OSError, ValueError, KeyError, TypeError) as error:
                    parser.error(f"{REFERENCE}: {error}")
                coverage = set(reference)
            else:
                coverage = cases
        if cases != coverage:
            baseline = REFERENCE.name if reference is not None else args.runs[0]
            parser.error(f"{run} has different case coverage from {baseline} " +
                         coverage_delta(coverage, cases))
        total += run_total
        failed += run_failed
        for case, texts in outputs.items():
            combined[case].update(texts)
    if reference is not None:
        mismatches = sorted(case for case, text in reference.items() if combined[case] != {text})
        result = f"cases differing from the reference: {len(mismatches)}"
    else:
        mismatches = sorted(case for case, texts in combined.items() if len(texts) > 1)
        result = f"cases with more than one output: {len(mismatches)}"
    print(f"{total} {mode} requests, {len(combined)} cases, {failed} failed, {result}"
          + (f" ({', '.join(mismatches)})" if mismatches else ""))
    return 1 if failed or mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
