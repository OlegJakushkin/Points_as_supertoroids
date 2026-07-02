"""Emit the reviewer-demanded ablation artifacts from compare/ablation.json + ablation_noise.json (NO GPU):
  paper/tab_ablation.tex     -- anchor-only vs direct-TSDF vs full, the 'what does the net add' table
  paper/figs/ablation_ftau.png -- F(tau) curves (offset absorbs loose tau) + full-vs-anchor-under-noise panel
"""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = json.load(open("compare/ablation_summary.json"))
NZ = json.load(open("compare/ablation_noise.json"))

# ---- table: anchor / direct / full, both stratifications -------------------------------------------
g = S["gate"]["methods"]; w = S["gt_watertight"]["methods"]
def row(lbl, k):
    a = g[k]; b = w[k]
    return (f"{lbl} & ${b['f_closed']:.0f}$ & ${a['f_open']:.1f}$ & ${a['chamfer']:.1f}$ & "
            f"${a['holes_open']:.0f}$ & ${a['selfx_open']:.0f}$ & ${a['parts_open']:.0f}$ & "
            f"${a['wt_pct']:.0f}$ & ${a['time']:.1f}$ \\\\")
lines = [
 r"\begin{table}[t]\centering\footnotesize",
 r"\begin{tabular}{lcccccccc}",
 r"\toprule",
 r"\textbf{variant} & \textbf{F}$_{\mathrm{cl}}$ & \textbf{F}$_{\mathrm{op}}$ & \textbf{Ch} & "
 r"\textbf{holes} & \textbf{self-X} & \textbf{\#c} & \textbf{wt\%} & \textbf{s} \\",
 r"\midrule",
 row(r"direct TSDF (no composition)", "direct"),
 row(r"region-composed anchor (untrained net)", "anchor"),
 row(r"\textbf{full model} (trained)", "full"),
 r"\bottomrule",
 r"\end{tabular}",
 (r"\caption{\textbf{What the network adds over its own analytic anchor} (80 ModelNet40 shapes; "
  r"F$_{\mathrm{cl}}$ on the $5$ GT-watertight solids, all other columns on the $75$ open shells). "
  r"\emph{Region composition} is the decisive step: it takes the plain nearest-point TSDF from "
  f"${g['direct']['holes_open']:.0f}$ hole edges / ${g['direct']['wt_pct']:.0f}$\\% watertight / "
  f"F$_{{\\mathrm{{op}}}}$ ${g['direct']['f_open']:.0f}$ to ${g['anchor']['holes_open']:.0f}$~/~"
  f"${g['anchor']['wt_pct']:.0f}$\\%~/~${g['anchor']['f_open']:.0f}$. "
  r"The \emph{trained} network is a zero-initialised residual on that composed anchor: on clean inputs it "
  f"changes the field by only ${100*S['resid_ratio']['mean']:.1f}$\\% of the anchor coefficient magnitude "
  f"(${100*S['resid_ratio_detail']['mean']:.0f}$\\% on the detail bands) and moves no quality metric "
  r"(it trims components $10.5\!\to\!8.4$ but leaves F, Chamfer, holes and watertight fraction unchanged). "
  r"We report this plainly: on this benchmark the reconstruction quality is produced by the analytic "
  r"pipeline, and the learned residual is, so far, a near-identity refinement.}"),
 r"\label{tab:ablation}",
 r"\end{table}",
]
open("paper/tab_ablation.tex", "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
print("wrote paper/tab_ablation.tex")

# ---- figure: F(tau) curves + noise panel ------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4))
taus = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1]
show = [("full", "ours (full)", "#c0392b", "-"), ("anchor", "anchor-only", "#e67e22", "--"),
        ("SPSR", "SPSR", "#2980b9", "-"), ("BPA", "BPA", "#27ae60", "-"), ("APSS", "APSS", "#8e44ad", "-")]
for k, lbl, c, ls in show:
    if k in S["f_tau_open"]:
        ax1.plot(taus, [S["f_tau_open"][k][str(t)] for t in taus], ls, color=c, label=lbl, lw=1.8, marker="o", ms=3)
ax1.axvline(0.05, color="#999", lw=0.8, ls=":"); ax1.axvline(0.02, color="#999", lw=0.8, ls=":")
ax1.set_xlabel(r"F-score threshold $\tau$"); ax1.set_ylabel("open-shell F-score"); ax1.set_xscale("log")
ax1.set_title("F$(\\tau)$: the loose $\\tau{=}0.05$ flatters the offset shell"); ax1.legend(fontsize=7); ax1.grid(alpha=.25)

lv = sorted({r["noise"] for r in NZ})
def nm(m, nz): return float(np.mean([r["methods"][m]["f_curve"]["0.05"] for r in NZ if r["noise"] == nz]))
ax2.plot([n*100 for n in lv], [nm("full", n) for n in lv], "-o", color="#c0392b", label="ours (full)", lw=1.8)
ax2.plot([n*100 for n in lv], [nm("anchor", n) for n in lv], "--o", color="#e67e22", label="anchor-only", lw=1.8)
ax2.set_xlabel("position noise (% of scale)"); ax2.set_ylabel("F-score @ $\\tau{=}0.05$")
ax2.set_title("Under noise the learned residual does not help"); ax2.legend(fontsize=8); ax2.grid(alpha=.25)
fig.tight_layout(); fig.savefig("paper/figs/ablation_ftau.png", dpi=140); plt.close(fig)
print("wrote paper/figs/ablation_ftau.png")
