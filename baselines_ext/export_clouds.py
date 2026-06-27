"""Export the EXACT comparison clouds (same points/normals our model sees) + GT meshes, so the dockerised learned
baselines run apples-to-apples.  CPU-only.  Writes baselines_ext/clouds/<id>.npz (points,normals in [-1,1]) and
<id>_gt.ply (GT mesh in the same frame)."""
import sys; sys.path.insert(0, "compare")
import os, glob, numpy as np
from core import sample, sample_path

OUT = "baselines_ext/clouds"; os.makedirs(OUT, exist_ok=True)
N_PTS = int(os.environ.get("N_PTS", "4096"))

# the 7 qualitative shapes from fig_baselines (canonical + ModelNet instances)
SHAPES = ["cube", "teapot", "bunny", "knurl", "chair", "guitar", "table"]
for s in SHAPES:
    gt, P, N = sample(s, n=N_PTS, seed=0)
    np.savez(f"{OUT}/{s}.npz", points=P.astype("float32"), normals=N.astype("float32"))
    gt.export(f"{OUT}/{s}_gt.ply")
    print(f"exported {s}: {len(P)} pts", flush=True)

# optional small ModelNet40 quantitative sample (MN_K shapes per category, 0 = skip)
MN_K = int(os.environ.get("MN_K", "0"))
if MN_K:
    cats = sorted(os.path.basename(d) for d in glob.glob("data/ModelNet40/*") if os.path.isdir(d))
    for c in cats:
        for p in sorted(glob.glob(f"data/ModelNet40/{c}/test/*.off"))[:MN_K]:
            sid = f"mn_{c}_{os.path.basename(p)[:-4]}"
            try:
                gt, P, N = sample_path(p, n=N_PTS, seed=0)
                np.savez(f"{OUT}/{sid}.npz", points=P.astype("float32"), normals=N.astype("float32"))
                gt.export(f"{OUT}/{sid}_gt.ply")
            except Exception as e:
                print(f"  skip {sid}: {e}", flush=True)
    print(f"+ ModelNet40 sample ({MN_K}/category)", flush=True)

print(f"wrote clouds to {OUT}/ ({len(glob.glob(OUT + '/*.npz'))} shapes)", flush=True)
