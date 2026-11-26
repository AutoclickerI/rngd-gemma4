#!/usr/bin/env python3
"""Replay the contest's performance stream, or its public accuracy cases, against a server binary on this
machine's RNGD, and report the score with the card's power, clock and temperature.

    export RNGD_MODEL_DIR=/path/to/gemma-4-12b-it-NVFP4
    tools/run_stream.py target/release/server                 # C warm-ups, then 15 C measured requests, C = 100
    tools/run_stream.py --accuracy target/release/server      # the six public cases
    tools/run_stream.py --accuracy --rounds 19 -c 100 SERVER  # the public cases 19 times at C100
    tools/run_stream.py --chip 1 SERVER                       # the second card of a host with several

The stream is the organizer's (scripts/stage2/performance.py): five request types in turn, 64 generated tokens
each, C requests in flight. The score is generated tokens per second over the measured window; the gates are
TTFT p90 <= 4 s and TPOT p90 <= 100 ms; the exit status is 0 only when every request passes and, for the stream,
the gates hold. One server lifetime per run; nothing else may use the card meanwhile.

The card's state decides several percent of the score (docs/HARDWARE.md, "Power and clock"), so a run waits
until the card is at or below --cool degrees or has stopped cooling, and reports the share of loaded samples with
the power-cap flag (0x4). "at 2.0 GHz" is the score scaled by 2.0 over the mean clock of the loaded samples, given
only when the power cap is the only throttle; compare builds by raw score on one card.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools/replay"))
import replay_stream  # noqa: E402

PUBLIC_CASES = 6  # public-test/cases.jsonl


def card_temperature(chip):
    """Degrees C of the card's hottest sensor, or None without the driver's hwmon node."""
    for node in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        try:
            if (node / "name").read_text().strip() == f"rngd{chip}":
                return int((node / "temp1_input").read_text()) / 1000
        except OSError:
            continue
    return None


def wait_cool(chip, limit, settle=90):
    """Wait until the card is at or below `limit`, or has not cooled for `settle` seconds (its idle temperature)."""
    start = fell = time.monotonic()
    lowest = None
    while (now := card_temperature(chip)) is not None and now > limit and time.monotonic() - fell < settle:
        if lowest is None:
            print(f"waiting for card {chip} to cool: {now:.0f} C, target {limit:.0f} C", flush=True)
        if lowest is None or now < lowest - 0.5:
            lowest, fell = now, time.monotonic()
        time.sleep(5)
    return card_temperature(chip), time.monotonic() - start


def telemetry(log, chip):
    """The server's 10 Hz samples of its card over the loaded part of the run (board power at or above 150 W)."""
    samples = []
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith("NPU_TELEMETRY") and '"event":"sample"' in line:
            try:
                samples.append(json.loads(line.split(" ", 1)[1]))
            except json.JSONDecodeError:
                pass
    loaded = [s for s in samples if s.get("npu") == chip and (s.get("power_uw") or 0) >= 150e6 and s.get("clocks_hz")]
    if not loaded:
        return {}
    ghz = statistics.fmean(statistics.fmean(s["clocks_hz"]) for s in loaded) / 1e9
    temperatures = [s["temp_mc"] for s in loaded if s.get("temp_mc") is not None]
    flags = [s.get("throttle_reason") for s in loaded]
    flags_known = all(type(flag) is int and flag >= 0 for flag in flags)
    return {"watts_median": statistics.median(s["power_uw"] for s in loaded) / 1e6,
            "celsius_max": max(temperatures) / 1000 if temperatures else None,
            "cap_share": statistics.fmean(bool(flag & 0x4) for flag in flags) if flags_known else None,
            "cap_only": flags_known and all(flag in (0, 0x4) for flag in flags),
            "mean_ghz": ghz}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("server", type=Path)
    parser.add_argument("-c", "--concurrency", type=int, default=None, help="default 100 (4 with --accuracy)")
    parser.add_argument("--accuracy", action="store_true", help="the public accuracy cases instead of the stream")
    parser.add_argument("--rounds", type=int, default=1, help="with --accuracy: repeat the six cases this many times")
    parser.add_argument("--chip", type=int, default=0, help="index of the card to use (default 0)")
    parser.add_argument("--cpus", help="CPU list for the server and the client (taskset), e.g. 16-23")
    parser.add_argument("--cool", type=float, default=42.0, help="card temperature to start at or below (deg C); starts anyway once the card stops cooling")
    parser.add_argument("--budget", type=float, default=600.0, help="seconds for the replay, after the cooling wait")
    parser.add_argument("--out", type=Path, help="run directory (default runs/<time>)")
    args = parser.parse_args()
    model = os.environ.get("RNGD_MODEL_DIR")
    if not model or not Path(model).is_dir():
        parser.error("set RNGD_MODEL_DIR to the checkpoint directory")
    server = args.server.resolve()
    mode = "accuracy" if args.accuracy else "performance"
    concurrency = args.concurrency or (4 if args.accuracy else 100)
    out = (args.out or ROOT / "runs" / time.strftime("%Y%m%d-%H%M%S")).resolve()
    out.mkdir(parents=True)

    selection = list(range(PUBLIC_CASES * args.rounds)) if args.accuracy and args.rounds > 1 else None
    setting = {"FURIOSA_VISIBLE_CHIPS": str(args.chip)}
    plan = replay_stream.build_plan(mode, concurrency, args.budget, candidate_environment=setting, public_selection=selection)
    plan["server_sha256"] = replay_stream.file_digest(server)
    plan_path = out / "replay-plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")

    start_c, waited = wait_cool(args.chip, args.cool)
    command = [sys.executable, "-u", str(ROOT / "tools/replay/replay_stream.py"), "--plan", str(plan_path),
               "--server", str(server), "--out", str(out / "replay"), "--public-test", str(ROOT / "public-test")]
    if args.cpus:
        command = ["taskset", "-c", args.cpus, *command]
    env = {"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "HOME": os.environ.get("HOME", "/"),
           "LANG": "C.UTF-8", "RNGD_MODEL_DIR": model}
    if "LD_LIBRARY_PATH" in os.environ:
        env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
    with open(out / "replay.stdout", "w") as stdout, open(out / "replay.stderr", "w") as stderr:
        code = subprocess.run(command, env=env, stdout=stdout, stderr=stderr).returncode

    end = {}
    with (out / "replay.stdout").open() as stdout:
        for line in stdout:
            if line.startswith("REPLAY_END "):
                end = json.loads(line[len("REPLAY_END "):])
    summary = {"server": str(server), "server_sha256": plan["server_sha256"], "mode": mode, "concurrency": concurrency, "chip": args.chip,
               "start_celsius": start_c, "cool_wait_seconds": round(waited, 1), "status": end.get("status"),
               "counts": end.get("counts"), "metrics": end.get("metrics"), "gates": end.get("gates"),
               "server_ready_seconds": end.get("server_ready_seconds")}
    log = out / "replay/server.log"
    card = telemetry(log, args.chip) if log.exists() else {}
    metrics = end.get("metrics") or {}
    score = metrics.get("generated_tokens_per_second")
    if card.get("cap_only") and score and card["mean_ghz"] > 0:
        card["tok_s_at_2ghz"] = score * 2.0 / card["mean_ghz"]
    summary["card"] = card
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"run {out}")
    if end.get("status") != "complete":
        print(f"replay {end.get('status', 'did not finish')} (exit {code}): {end.get('error')}; see {out}/replay.stderr")
        return 1
    measured = end["counts"]["measurement"]
    if mode == "accuracy":
        print(f"accuracy {measured['passed']}/{measured['planned']}")
    else:
        gates = end["gates"]
        print(f"{score:.1f} tok/s at C{concurrency} ({measured['completed']} requests in "
              f"{metrics['measurement_window_seconds']:.1f} s); TTFT p50/p90 {metrics['ttft_ms']['p50']:.0f}/"
              f"{metrics['ttft_ms']['p90']:.0f} ms, TPOT p50/p90 {metrics['estimated_tpot_ms']['p50']:.1f}/"
              f"{metrics['estimated_tpot_ms']['p90']:.1f} ms; gates {'passed' if gates['passed'] else 'FAILED'}")
    if card:
        parts = []
        if start_c is not None:
            parts.append(f"start {start_c:.0f} C")
        if card["celsius_max"] is not None:
            parts.append(f"max {card['celsius_max']:.0f} C")
        parts.append(f"{card['watts_median']:.0f} W median")
        if card["cap_share"] is not None:
            parts.append(f"cap engaged {100 * card['cap_share']:.0f} % of loaded samples")
        else:
            parts.append("throttle flags unavailable")
        parts.append(f"mean clock {card['mean_ghz']:.3f} GHz")
        print("card: " + ", ".join(parts)
              + (f"; {card['tok_s_at_2ghz']:.1f} tok/s at 2.0 GHz" if "tok_s_at_2ghz" in card else ""))
    return 0 if measured["passed"] == measured["planned"] and (mode == "accuracy" or end["gates"]["passed"]) else 1


if __name__ == "__main__":
    sys.exit(main())
