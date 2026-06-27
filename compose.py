"""Compose measurement runs (local here + Colab later) into one comparison.  Each `results/<run>/` holds the
same files a run produces; this merges them so the FIXED public baselines appear once and OUR model is shown
per run (its checkpoint differs).  Drop the Colab `DRIVE_WS/results` folder in as `results/colab/` and re-run.

  python compose.py                      # composes every results/<run>/ found

Writes results/comparison.md (+ .json): F-score closed/open, signed-distance error, normal-consistency,
open-shell component count, runtime -- the quantitative table the reviewers asked for, per run."""
import os, glob, json
import numpy as np

METHODS = ["SPSR", "BPA", "APSS", "RIMLS", "tori", "ours"]
RUNS = sorted(d for d in glob.glob("results/*") if os.path.isdir(d) and os.path.exists(f"{d}/metrics_val.json"))


def agg(rows, m, metric, kind=None):
    rs = rows if kind is None else [r for r in rows if r.get("kind") == kind]
    vals = [r["methods"][m][metric] for r in rs if m in r.get("methods", {})
            and isinstance(r["methods"][m].get(metric), (int, float)) and r["methods"][m][metric] == r["methods"][m][metric]]
    return float(np.mean(vals)) if vals else float("nan")


table, run_names = {}, []
for d in RUNS:
    run = os.path.basename(d); run_names.append(run)
    rows = json.load(open(f"{d}/metrics_val.json"))
    wt = [r for r in rows if r.get("watertight")]
    op = [r for r in rows if r.get("kind") == "open"]
    for m in METHODS:
        table.setdefault(m, {})[run] = dict(
            n=len(rows), F_closed=agg(rows, m, "fscore", "closed"), F_open=agg(rows, m, "fscore", "open"),
            sdf=agg(wt, m, "sdf_err"), ncons=agg(rows, m, "ncons"),
            parts_open=agg(op, m, "parts"), time=agg(rows, m, "time"))


def cell(v, f="{:.1f}"):
    return "--" if v != v else f.format(v)


lines = [f"# Composed comparison — runs: {', '.join(run_names) or '(none)'}", "",
         "Public baselines are FIXED across runs; **ours** changes with the trained checkpoint, so its row is",
         "repeated per run to show training's effect against the same baselines.", "",
         "| method | run | F closed | F open | SDF-err | normal-cons | parts (open) | time (s) |",
         "|---|---|---|---|---|---|---|---|"]
for m in METHODS:
    for run in run_names:
        t = table.get(m, {}).get(run)
        if not t:
            continue
        rl = run if m == "ours" else (run if len(run_names) == 1 else "all")
        lines.append(f"| {m} | {rl} | {cell(t['F_closed'])} | {cell(t['F_open'])} | {cell(t['sdf'],'{:.2f}')} | "
                     f"{cell(t['ncons'],'{:.3f}')} | {cell(t['parts_open'],'{:.0f}')} | {cell(t['time'],'{:.2f}')} |")
        if m != "ours":
            break   # baselines identical across runs -> show once

# learned baselines (if a run scored them)
learned_runs = {os.path.basename(d): json.load(open(f"{d}/learned_metrics.json"))
                for d in RUNS if os.path.exists(f"{d}/learned_metrics.json")}
if learned_runs:
    lines += ["", "## Learned baselines (per-shape F-score; * = zero-shot/OOD)", ""]
    for run, lm in learned_runs.items():
        ms = sorted({m for s in lm for m in lm[s]})
        means = {m: np.mean([lm[s][m]["fscore"] for s in lm if m in lm[s]]) for m in ms}
        lines.append(f"- **{run}**: " + ", ".join(f"{m} {means[m]:.0f}" for m in ms))

os.makedirs("results", exist_ok=True)
open("results/comparison.md", "w").write("\n".join(lines) + "\n")
json.dump(table, open("results/comparison.json", "w"), indent=1)
print("\n".join(lines))
print(f"\nwrote results/comparison.md  ({len(run_names)} run(s): {run_names})")
