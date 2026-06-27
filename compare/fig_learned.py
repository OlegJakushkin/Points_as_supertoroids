"""Comparison vs the DOCKERISED learned baselines, on the SAME 7 clouds (baselines_ext/clouds).  Renders
GT | <available learned methods> | ours with F-score under each, and writes a metrics table.  Feed-forward
methods (POCO/NKSR/ConvONet, marked *) run on ShapeNet/ABC weights -> zero-shot/OOD; per-shape optimizers
(Neural-Pull/CAP-UDF/SAP) are the fair apples-to-apples.  Run AFTER the dockerised runners have written
baselines_ext/out/<method>/<id>.ply (and `score.py`)."""
import sys, os, glob, json; sys.path.insert(0, "compare")
import numpy as np, trimesh
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core import recon_ours, draw3d, fscore, sdf_error, ncomp

CLOUDS = os.environ.get("BL_CLOUDS", "baselines_ext/clouds")
OUTROOT = os.environ.get("BL_OUT", "baselines_ext/out")   # set BL_OUT to a Drive folder on Colab
LABEL = {"neuralpull": "Neural-Pull", "capudf": "CAP-UDF", "sap": "SAP", "poco": "POCO*", "nksr": "NKSR*", "convonet": "ConvONet*"}
ORDER = ["neuralpull", "capudf", "sap", "poco", "nksr", "convonet"]      # * = zero-shot/OOD

shapes = sorted(f[:-7] for f in os.listdir(CLOUDS) if f.endswith("_gt.ply"))
present = [m for m in ORDER if os.path.isdir(f"{OUTROOT}/{m}") and glob.glob(f"{OUTROOT}/{m}/*.ply")]
cols = ["GT"] + present + ["ours"]
print(f"{len(shapes)} shapes | learned present: {present}", flush=True)
if not present:
    print("no learned-baseline outputs in baselines_ext/out/ yet -- run the dockerised runners first", flush=True)
    sys.exit(0)

table = {}
fig = plt.figure(figsize=(2.0 * len(cols), 2.1 * len(shapes)))
for i, s in enumerate(shapes):
    gt = trimesh.load(f"{CLOUDS}/{s}_gt.ply", force="mesh")
    d = np.load(f"{CLOUDS}/{s}.npz"); vo, fo, _ = recon_ours(d["points"], d["normals"])
    panel = {"GT": (gt.vertices, gt.faces), "ours": (vo, fo)}
    for m in present:
        mp = glob.glob(f"{OUTROOT}/{m}/{s}.ply")
        if mp:
            mm = trimesh.load(mp[0], force="mesh"); panel[m] = (np.asarray(mm.vertices), np.asarray(mm.faces))
        else:
            panel[m] = (None, None)
    table[s] = {}
    for j, m in enumerate(cols):
        ax = fig.add_subplot(len(shapes), len(cols), i * len(cols) + j + 1, projection="3d")
        v, f = panel.get(m, (None, None)); draw3d(ax, v, f)
        if i == 0:
            ax.set_title({"GT": "GT", "ours": "ours"}.get(m, LABEL.get(m, m)), fontsize=9, weight="bold")
        if m != "GT":
            fsv = fscore(v, f, gt) if v is not None and len(f) else 0.0
            table[s][m] = {"fscore": fsv, "sdf_err": (sdf_error(v, f, gt) if v is not None else float("nan")),
                           "parts": (ncomp(v, f) if v is not None else 0)}
            ax.text2D(0.5, -0.02, ("fail" if v is None else f"F={fsv:.0f}"), transform=ax.transAxes,
                      ha="center", fontsize=8, color=("#207020" if m == "ours" else "#444"))
        if j == 0:
            ax.text2D(-0.1, 0.5, s, transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=9)
fig.subplots_adjust(left=0.04, right=0.99, top=0.95, bottom=0.03, wspace=0.0, hspace=0.12)
os.makedirs("paper/figs", exist_ok=True)
fig.savefig("paper/figs/cmp_learned.png", dpi=130); plt.close(fig)
json.dump(table, open("baselines_ext/learned_vs_ours.json", "w"), indent=1)
print("\nmean F-score (* = zero-shot/OOD feed-forward):")
for m in present + ["ours"]:
    fs = [table[s][m]["fscore"] for s in shapes if m in table[s]]
    if fs: print(f"  {LABEL.get(m, m):13s} {np.mean(fs):5.1f}", flush=True)
print("wrote paper/figs/cmp_learned.png", flush=True)
