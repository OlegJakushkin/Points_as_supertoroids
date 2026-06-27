"""Score the dockerised learned baselines' output meshes with the SAME metrics our harness uses, so they fold
straight into the comparison.  Reads baselines_ext/out/<method>/<id>.ply against baselines_ext/clouds/<id>_gt.ply.
CPU-only.  Writes baselines_ext/learned_metrics.json (per shape, per method)."""
import sys; sys.path.insert(0, "compare")
import os, glob, json, numpy as np, trimesh
from core import fscore, sdf_error, normal_consistency, mesh_defects, chamfer, ncomp

CLOUDS = os.environ.get("BL_CLOUDS", "baselines_ext/clouds")
OUTROOT = os.environ.get("BL_OUT", "baselines_ext/out")   # set BL_OUT to a Drive folder on Colab
ids = sorted(f[:-7] for f in os.listdir(CLOUDS) if f.endswith("_gt.ply"))
methods = sorted(d for d in os.listdir(OUTROOT) if os.path.isdir(f"{OUTROOT}/{d}")) if os.path.isdir(OUTROOT) else []
print(f"scoring {len(methods)} methods on {len(ids)} shapes: {methods}", flush=True)

res = {}
for s in ids:
    gt = trimesh.load(f"{CLOUDS}/{s}_gt.ply", force="mesh")
    res[s] = {}
    for m in methods:
        mp = sorted(glob.glob(f"{OUTROOT}/{m}/{s}.*"))
        mp = [p for p in mp if p.rsplit(".", 1)[-1].lower() in ("ply", "obj", "off", "stl")]
        if not mp:
            continue
        try:
            mesh = trimesh.load(mp[0], force="mesh"); v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
            d = mesh_defects(v, f)
            res[s][m] = {"fscore": fscore(v, f, gt), "sdf_err": sdf_error(v, f, gt),
                         "ncons": normal_consistency(v, f, gt), "chamfer": chamfer(v, f, gt),
                         "parts": ncomp(v, f), "faces": int(len(f)), "watertight_out": d["watertight"]}
            print(f"  {s:14s} {m:11s} F={res[s][m]['fscore']:5.1f}  sdf={res[s][m]['sdf_err']:.2f}  parts={res[s][m]['parts']}", flush=True)
        except Exception as e:
            print(f"  {s} {m}: FAIL {str(e)[:80]}", flush=True)

json.dump(res, open("baselines_ext/learned_metrics.json", "w"), indent=1)
# compact per-method means (F-score, where present)
for m in methods:
    fs = [res[s][m]["fscore"] for s in ids if m in res[s]]
    if fs: print(f"MEAN {m:11s} F={np.mean(fs):5.1f}  ({len(fs)} shapes)", flush=True)
print("wrote baselines_ext/learned_metrics.json", flush=True)
