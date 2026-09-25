"""Score the voice assistant on the same scenarios, several times, and compare.

The gate for every change to the voice stack. A single run proves nothing:
the same broker passed a check, then failed it, then passed it again within
the hour. So each variant runs every scenario --trials times through OpenAI's
harness, and what comes out is a pass rate per scenario and P50/P90 latency.

    uv run --project evals/gpt_live_evals python evals/gate.py --variants raw guided --trials 3
    ... --save-baseline     record this run as the bar
    ... --against-baseline  exit 1 if any variant's pass rate fell below the bar

Variants:
    raw     gpt-live-1 with the production persona prompt and nothing else
    guided  the same plus the broker's delegation guidance
    broker  the production broker under test, through evals/broker_adapter.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "gpt_live_evals"
RESULTS = HARNESS / "crawl_harness" / "results"
BASELINE = HERE / "baseline.json"
ENV_FILE = HERE.parent / "broker" / ".env"
BACKEND_MODEL = "gpt-5.4-mini"  # production's LIVE_BACKEND_MODEL

VARIANTS = {
    "raw": {"GPT_LIVE_EVALS_PROMPT": "raw"},
    "guided": {"GPT_LIVE_EVALS_PROMPT": "guided"},
    "broker": {"GPT_LIVE_EVALS_PROMPT": "raw", "OPENAI_LIVE_ENDPOINT": "ws://127.0.0.1:8790/v1/live/sessions",
               "OPENAI_LIVE_ALLOW_INSECURE_LOOPBACK": "true", "GPT_LIVE_EVALS_HOUSE_URL": "http://127.0.0.1:8791"},
}
# One broker, one fake house: its scenarios cannot overlap.
SERIAL = {"broker"}


def run_trial(variant: str, trial: int, data: Path, concurrency: int, only: list[str]) -> Path:
    name = f"gate_{variant}_t{trial}_{int(time.time())}"
    env = {
        **os.environ,
        "GPT_LIVE_EVALS_ENV_FILE": str(ENV_FILE),
        "GPT_LIVE_EVALS_DOMAIN": "smart_home",
        "OPENAI_LIVE_VOICE": "cedar",
        **VARIANTS[variant],
    }
    cmd = ["uv", "run", "crawl-eval", "--data", str(data), "--backend-model", BACKEND_MODEL,
           "--concurrency", str(1 if variant in SERIAL else concurrency), "--run-name", name]
    for example in only:
        cmd += ["--example", example]
    subprocess.run(cmd, cwd=HARNESS, env=env, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    found = sorted(RESULTS.glob(f"{name}_*"))
    if not found:
        raise SystemExit(f"{variant} trial {trial}: the harness wrote no results")
    return found[-1] / "results.json"


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, round(q * (len(values) - 1)))]


def summarize(files: list[Path]) -> dict:
    per: dict[str, dict] = {}
    latencies: list[float] = []
    infra = 0
    for f in files:
        for r in json.loads(f.read_text())["results"]:
            s = per.setdefault(r["id"] if "id" in r else r["scenario_id"], {"passed": 0, "runs": 0, "infra": 0})
            s["runs"] += 1
            if r["status"] == "passed":
                s["passed"] += 1
            elif r["status"] == "infrastructure_error":
                s["infra"] += 1
                infra += 1
            lat = (r.get("metrics") or {}).get("audio", {}).get("response_latency_ms")
            if lat is not None:
                latencies.append(lat)
    graded = sum(s["runs"] - s["infra"] for s in per.values())
    passed = sum(s["passed"] for s in per.values())
    return {
        "pass_rate": round(passed / graded, 3) if graded else 0.0,
        "passed": passed,
        "graded": graded,
        "infrastructure_errors": infra,
        "latency_p50_ms": pct(latencies, 0.5),
        "latency_p90_ms": pct(latencies, 0.9),
        "scenarios": per,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["raw", "guided"], choices=sorted(VARIANTS))
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--data", type=Path, default=HARNESS / "smart_home_data" / "crawl.json")
    ap.add_argument("--example", action="append", default=[])
    ap.add_argument("--save-baseline", action="store_true")
    ap.add_argument("--against-baseline", action="store_true")
    args = ap.parse_args()

    report = {}
    for variant in args.variants:
        files = [run_trial(variant, t, args.data.resolve(), args.concurrency, args.example)
                 for t in range(1, args.trials + 1)]
        report[variant] = summarize(files)
        report[variant]["result_files"] = [str(f.relative_to(HERE)) for f in files]

    scenario_ids = sorted({sid for v in report.values() for sid in v["scenarios"]})
    print(f"\n{'scenario':32}" + "".join(f"{v:>10}" for v in report))
    for sid in scenario_ids:
        cells = []
        for v in report.values():
            s = v["scenarios"].get(sid)
            cells.append(f"{s['passed']}/{s['runs'] - s['infra']:<3}".rjust(10) if s else "-".rjust(10))
        print(f"{sid:32}" + "".join(cells))
    print(f"{'PASS RATE':32}" + "".join(f"{v['pass_rate']:>10.0%}" for v in report.values()))
    print(f"{'latency p50 ms':32}" + "".join(f"{v['latency_p50_ms'] or 0:>10.0f}" for v in report.values()))
    print(f"{'latency p90 ms':32}" + "".join(f"{v['latency_p90_ms'] or 0:>10.0f}" for v in report.values()))
    print(f"{'infra errors':32}" + "".join(f"{v['infrastructure_errors']:>10}" for v in report.values()))

    out = HERE / "last_gate.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    if args.save_baseline:
        BASELINE.write_text(json.dumps({v: {"pass_rate": r["pass_rate"], "latency_p50_ms": r["latency_p50_ms"]}
                                        for v, r in report.items()}, indent=1) + "\n")
        print(f"\nbaseline saved to {BASELINE.relative_to(HERE.parent)}")
    if args.against_baseline and BASELINE.exists():
        bar = json.loads(BASELINE.read_text())
        worse = [v for v, r in report.items() if v in bar and r["pass_rate"] < bar[v]["pass_rate"]]
        for v in worse:
            print(f"WORSE: {v} {report[v]['pass_rate']:.0%} < baseline {bar[v]['pass_rate']:.0%}")
        return 1 if worse else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
