"""Aggregate compare/ablation.json (+ optional ablation_noise.json) into the numbers the paper quotes:
per-method means under BOTH stratifications (our gate vs GT watertightness), F(tau) rows, directional-Chamfer
offset split, residual magnitude, gate-delta stratum flips.  Prints a compact report + writes
compare/ablation_summary.json."""
import json, os, sys
import numpy as np

rows = json.load(open("compare/ablation.json"))
methods = sorted({m for r in rows for m in r["methods"]})
TAUS = ["0.005", "0.01", "0.02", "0.03", "0.05", "0.075", "0.1"]


def agg(rs, m, key):
    vals = [r["methods"][m][key] for r in rs if m in r["methods"]
            and isinstance(r["methods"][m].get(key), (int, float)) and r["methods"][m][key] == r["methods"][m][key]
            and r["methods"][m][key] >= 0]
    return float(np.mean(vals)) if vals else float("nan")


def fmean(rs, m, tau):
    vals = [r["methods"][m]["f_curve"].get(tau) for r in rs if m in r["methods"] and tau in r["methods"][m]["f_curve"]]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else float("nan")


def ci95(vals):
    vals = np.asarray([v for v in vals if v == v])
    return (float(vals.mean()), float(1.96 * vals.std(ddof=1) / max(np.sqrt(len(vals)), 1))) if len(vals) > 1 else (float("nan"), 0.0)


out = {"n": len(rows)}
strats = {
    "gate": (lambda r: r["thin_frac"] <= 0.30, lambda r: r["thin_frac"] > 0.30),
    "gt_watertight": (lambda r: r["gt_watertight"], lambda r: not r["gt_watertight"]),
}
for sname, (isc, iso) in strats.items():
    closed = [r for r in rows if isc(r)]; opn = [r for r in rows if iso(r)]
    out[sname] = {"n_closed": len(closed), "n_open": len(opn), "methods": {}}
    for m in methods:
        fc, fch = ci95([r["methods"][m]["f_curve"]["0.05"] for r in closed if m in r["methods"]])
        fo, foh = ci95([r["methods"][m]["f_curve"]["0.05"] for r in opn if m in r["methods"]])
        out[sname]["methods"][m] = {
            "f_closed": round(fc, 1), "f_closed_ci": round(fch, 1),
            "f_open": round(fo, 1), "f_open_ci": round(foh, 1),
            "chamfer": round(agg(rows, m, "chamfer"), 2),
            "holes_open": round(agg(opn, m, "holes"), 1), "selfx_open": round(agg(opn, m, "self_x"), 1),
            "parts_open": round(agg(opn, m, "parts"), 1),
            "wt_pct": round(100 * np.mean([1 if r["methods"][m].get("watertight") else 0 for r in rows if m in r["methods"]]), 1),
            "time": round(agg(rows, m, "time"), 2),
        }
# F(tau) full curves (all shapes) + open-only + recon->GT / GT->recon split
out["f_tau"] = {m: {t: round(fmean(rows, m, t), 1) for t in TAUS} for m in methods}
opn_g = [r for r in rows if r["thin_frac"] > 0.30]
out["f_tau_open"] = {m: {t: round(fmean(opn_g, m, t), 1) for t in TAUS} for m in methods}
out["f_auc"] = {m: round(agg(rows, m, "f_auc"), 1) for m in methods}
out["dir_split_open"] = {m: {"recon_to_gt": round(agg(opn_g, m, "r2g"), 2),
                             "gt_to_recon": round(agg(opn_g, m, "g2r"), 2)} for m in methods}
# residual magnitude + gate-delta stratum flips
rr = [r["resid_ratio"] for r in rows]; rrd = [r["resid_ratio_detail"] for r in rows]
out["resid_ratio"] = {"mean": round(float(np.mean(rr)), 4), "max": round(float(np.max(rr)), 4)}
out["resid_ratio_detail"] = {"mean": round(float(np.mean(rrd)), 4), "max": round(float(np.max(rrd)), 4)}
kind = lambda tf: "open" if tf > 0.30 else "closed"
flips_lo = sum(1 for r in rows if kind(r["thin_frac_lo"]) != kind(r["thin_frac"]))
flips_hi = sum(1 for r in rows if kind(r["thin_frac_hi"]) != kind(r["thin_frac"]))
out["stratum_flips"] = {"delta_-25%": flips_lo, "delta_+25%": flips_hi, "n": len(rows)}
agree = sum(1 for r in rows if (r["thin_frac"] <= 0.30) == r["gt_watertight"] or
            ((r["thin_frac"] > 0.30) and not r["gt_watertight"]))
out["gate_vs_gtwt"] = {"gt_watertight": sum(1 for r in rows if r["gt_watertight"]),
                       "gate_closed": sum(1 for r in rows if r["thin_frac"] <= 0.30)}

if os.path.exists("compare/ablation_noise.json"):
    nrows = json.load(open("compare/ablation_noise.json"))
    lv = sorted({r["noise"] for r in nrows})
    out["noise"] = {f"{n:g}": {m: round(float(np.mean([r["methods"][m]["f_curve"]["0.05"] for r in nrows
                    if r["noise"] == n and m in r["methods"]])), 1)
                    for m in sorted({m for r in nrows for m in r["methods"]})} for n in lv}

json.dump(out, open("compare/ablation_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
