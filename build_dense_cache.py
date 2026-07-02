"""Build an ALIGNED (sparse 1536 + dense N) cloud cache pair from ModelNet40 meshes, in ONE loop with shared
skip logic, so sparse[i] and dense[i] are always the same mesh.  Writes data/se_clouds_aln.pt (sparse) and
data/se_clouds_dense.pt (dense).  Point train_detail.py at them with --sparse/--dense."""
import argparse, glob, os
import numpy as np, torch
from waveshape import eval3d as E
from waveshape.datasets import load_mesh_normalized

ap = argparse.ArgumentParser()
ap.add_argument("--cap", type=int, default=0, help="max meshes (0 = all)")
ap.add_argument("--dense-n", type=int, default=8192)
ap.add_argument("--root", default="data/ModelNet40")
a = ap.parse_args()

files = sorted(glob.glob(f"{a.root}/*/train/*.off"))
if a.cap: files = files[:a.cap]
print(f"building aligned sparse(1536)+dense({a.dense_n}) from {len(files)} meshes...", flush=True)
Ps, Ns, Pds, Nds = [], [], [], []
for i, p in enumerate(files):
    try:
        m = load_mesh_normalized(p, max_faces=200000)
        Pp, Nn = E.sample_cloud(m, n=1536, noise=0.0, seed=0)
        Pv, Nv = E.sample_cloud(m, n=a.dense_n, noise=0.0, seed=1)
        if not (np.isfinite(Pp).all() and np.isfinite(Nn).all() and np.isfinite(Pv).all() and np.isfinite(Nv).all()):
            continue
        Ps.append(Pp.astype(np.float32)); Ns.append(Nn.astype(np.float32))
        Pds.append(Pv.astype(np.float32)); Nds.append(Nv.astype(np.float32))
    except Exception:
        continue
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(files)} ({len(Ps)} ok)", flush=True)
torch.save({"P": torch.tensor(np.stack(Ps)),  "N": torch.tensor(np.stack(Ns))},  "data/se_clouds_aln.pt")
torch.save({"P": torch.tensor(np.stack(Pds)), "N": torch.tensor(np.stack(Nds))}, "data/se_clouds_dense.pt")
print(f"wrote {len(Ps)} aligned shapes -> data/se_clouds_aln.pt + data/se_clouds_dense.pt", flush=True)
