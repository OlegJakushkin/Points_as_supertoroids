"""Train CoeffNet(supertoroid) on ModelNet40 (>=10000 real models) WITH NOISE.

This is the local, time-boxed counterpart to the Colab notebook: it streams noisy
point clouds sampled from the 12,311 real ModelNet40 CAD models and trains the
supertoroid coefficient network with the paper's L1 + eikonal loss.

To stay feasible on CPU, ground-truth signed distance is computed against a dense
KD-tree of surface samples (fast, ~surface-spacing accurate) instead of the exact
per-triangle query used in ``pat.datasets`` (which is what the Colab notebook uses
on GPU).  The *training set* is the full ModelNet40 index; we iterate over a
shuffled prefix so the run genuinely consumes >=10000 distinct real models.

Usage:  python train_noisy.py --models 10000 --out renders/pat_supertoroid_noisy.pt
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

from pat import core
from pat.datasets import load_mesh_normalized, modelnet_index, noisy_point_cloud
from pat.model import CoeffNet
from pat.neighbors import knn_neighborhoods
from pat.shapes import sample_mesh
from pat.train import pat_loss


def off_nfaces(path):
    """Read the face count from an OFF header without loading the whole mesh.

    Lets us cheaply skip the rare giant ModelNet meshes (up to ~240k faces) whose
    loading alone dominates the per-example cost, before paying for them.
    """
    try:
        with open(path, "r", errors="ignore") as f:
            parts = f.readline().strip().split()
            p0 = parts[0].upper()
            if p0 == "OFF":
                if len(parts) >= 4:                  # "OFF nv nf ne" on one line
                    return int(parts[2])
                line = f.readline().strip()          # counts on the next line
                while not line or line.startswith("#"):
                    line = f.readline().strip()
                return int(line.split()[1])
            if p0.startswith("OFF"):                 # malformed "OFFnv nf ne" (ModelNet)
                return int(parts[1])
    except Exception:
        pass
    return 0


def fast_gt(mesh, queries, dense=25000, rng=None):
    """Signed distance via a KD-tree of dense surface samples (fast approx GT)."""
    pts, nrm = sample_mesh(mesh, dense, rng)
    tree = cKDTree(pts)
    d, idx = tree.query(queries)
    sign = np.einsum("ij,ij->i", queries - pts[idx], nrm[idx])
    return (np.where(sign >= 0, 1.0, -1.0) * d).astype(np.float32)


def make_example(path, rng, *, n_points=256, k=16, n_query=160, noise_std=0.01,
                 bound=1.0):
    """One noisy training example from a real mesh, with fast KD-tree ground truth."""
    mesh = load_mesh_normalized(path)
    pts, nrm = noisy_point_cloud(mesh, n_points, rng, noise_std=noise_std)
    idx = knn_neighborhoods(pts, k)
    nb = torch.as_tensor(idx, dtype=torch.long)
    P = torch.as_tensor(pts, dtype=torch.float32)
    Nn = torch.as_tensor(nrm, dtype=torch.float32)
    Nn = Nn / Nn.norm(dim=1, keepdim=True).clamp_min(1e-9)
    # queries: narrow band around the (noisy) surface + bulk
    surf, _ = sample_mesh(mesh, n_query // 2, rng)
    band = surf + rng.normal(scale=max(3 * noise_std, 0.02), size=surf.shape)
    cube = rng.uniform(-bound, bound, size=(n_query - len(band), 3))
    q = np.concatenate([band, cube], 0).astype(np.float32)
    phi = fast_gt(mesh, q.astype(np.float64), rng=rng)
    return dict(P=P, Nn=Nn, nbr_pos=P[nb], nbr_nrm=Nn[nb],
                q=torch.as_tensor(q), phi=torch.as_tensor(phi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=int, default=10000,
                    help="number of distinct real models to train on (>=10000 requested)")
    ap.add_argument("--out", default="renders/pat_supertoroid_noisy.pt")
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-faces", type=int, default=40000,
                    help="skip ModelNet meshes larger than this (loading them dominates cost)")
    args = ap.parse_args()

    paths = modelnet_index()
    assert len(paths) >= 10000, f"ModelNet index has only {len(paths)} models"
    rng = np.random.default_rng(0)
    # keep only meshes small enough to process quickly (cheap OFF-header check)
    small = [p for p in paths if 0 < off_nfaces(p) <= args.max_faces]
    print(f"training set: {len(paths)} ModelNet40 models; "
          f"{len(small)} within {args.max_faces} faces; consuming up to {args.models}")
    rng.shuffle(small)
    order = small[: args.models]

    cfg = dict(d_embed=128, n_layers=6, n_heads=8, d_ff=512, supertoroid=True)
    net = CoeffNet(**cfg)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(order))

    hist = []
    used = 0
    t0 = time.time()
    for step, path in enumerate(order):
        # noise curriculum: ramp 0.005 -> 0.02 across the run
        noise = 0.005 + 0.015 * step / max(len(order) - 1, 1)
        try:
            ex = make_example(path, rng, noise_std=noise)
        except Exception:
            continue
        coeffs, _, sq = net(ex["nbr_pos"], ex["nbr_nrm"])
        loss, ld, le = pat_loss(ex["P"], ex["Nn"], coeffs, ex["q"], ex["phi"],
                                eikonal_weight=0.1, supertoroid=(sq[:, 0], sq[:, 1]))
        if not torch.isfinite(loss):
            continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        hist.append(float(loss.detach())); used += 1
        if used % 50 == 0:
            r = used / (time.time() - t0)
            print(f"step {used:5d}/{len(order)}  noise {noise:.3f}  loss {np.mean(hist[-50:]):.4f} "
                  f"dist {float(ld):.4f} eik {float(le):.4f}  {r:.1f} ex/s", flush=True)
        if used % args.ckpt_every == 0:
            torch.save({"state_dict": net.state_dict(), "config": cfg,
                        "models_consumed": used, "loss": hist}, args.out)
    torch.save({"state_dict": net.state_dict(), "config": cfg,
                "models_consumed": used, "loss": hist}, args.out)
    print(f"DONE: trained on {used} distinct noisy real models; saved {args.out}; "
          f"final loss {np.mean(hist[-100:]):.4f}")


if __name__ == "__main__":
    main()
