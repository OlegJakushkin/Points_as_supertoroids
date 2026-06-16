"""Train BOTH a plain-torus and a supertoroid CoeffNet on the SAME dataset (GPU).

This is the "do it properly" trainer behind the README figures:

* It builds one shared dataset that deliberately **explores the facets the
  supertoroid adds** -- supertoroids with a wide range of squareness exponents,
  plus sharp/faceted assets (cubes, the knurled :class:`TexturedCylinder`, the
  :class:`BoltPlate`) -- mixed with smooth shapes and noisy **ModelNet40** real
  models (the >=10,000-model set), all with input noise.
* It then trains two networks on that identical example stream: a **plain torus**
  model (`supertoroid=False`, the paper's primitive) and our **supertoroid** model
  (`supertoroid=True`), so the two are compared apples-to-apples.
* Each epoch it validates by reconstructing a *default torus* and reports the mean
  absolute SDF error; the goal is an error small enough to be invisible by eye.
* Best checkpoints are written to ``assets/pat_torus.pt`` and
  ``assets/pat_supertoroid.pt`` for use by the renderer, demo, tests and Docker.

Usage:  python train_gpu.py --epochs 40 --cache 4000 --modelnet 1500
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from pat import core, shapes
from pat.assets import BoltPlate, BoxWithCylinders, Cube, TexturedCylinder
from pat.model import CoeffNet
from pat.neighbors import knn_neighborhoods
from pat.train import pat_loss

if not torch.cuda.is_available():
    raise SystemExit(
        "train_gpu.py requires a CUDA GPU and is meant to run via Docker.\n"
        "Use:  docker compose run --rm train\n"
        "(training is GPU-only by policy; see the train-gpu-docker skill).")
DEVICE = "cuda"


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #
def random_analytic_shape(rng):
    """A shape drawn to exercise the supertoroid's expressive range + sharp features."""
    r = rng.random()
    if r < 0.34:                     # supertoroids: the squareness/facet explorer
        R = rng.uniform(0.4, 0.7); rr = rng.uniform(0.15, 0.32) * R / 0.6
        return shapes.SuperToroid(R=R, r=rr, p_tube=rng.uniform(2.0, 6.0),
                                  p_ring=rng.uniform(2.0, 4.0), axis=rng.normal(size=3))
    if r < 0.54:                     # tori (the clean-torus target)
        R = rng.uniform(0.4, 0.7); rr = rng.uniform(0.14, 0.30) * R / 0.6
        return shapes.Torus(R=R, r=rr, axis=rng.normal(size=3))
    if r < 0.70:                     # smooth basics
        c = rng.random()
        if c < 0.4:
            return shapes.Sphere(rng.uniform(0.3, 0.8))
        if c < 0.8:
            return shapes.RoundedBox(half=rng.uniform(0.3, 0.6, size=3),
                                     radius=rng.uniform(0.05, 0.2))
        return shapes.Plane(normal=rng.normal(size=3))
    if r < 0.86:                     # cubes (sharp corners/edges)
        return Cube(half=rng.uniform(0.4, 0.6), rounding=rng.uniform(0.01, 0.06))
    c = rng.random()                 # faceted "texture" assets
    if c < 0.55:
        return TexturedCylinder(radius=rng.uniform(0.28, 0.38), amp=rng.uniform(0.03, 0.06),
                                n_around=int(rng.integers(18, 30)),
                                n_axial=int(rng.integers(14, 26)))
    if c < 0.8:
        return BoltPlate()
    return BoxWithCylinders()


def _pack(pts, nrm, idx, q, phi):
    nb = torch.as_tensor(idx, dtype=torch.long)
    P = torch.as_tensor(pts, dtype=torch.float32)
    Nn = torch.as_tensor(nrm, dtype=torch.float32)
    Nn = Nn / Nn.norm(dim=1, keepdim=True).clamp_min(1e-9)
    return dict(P=P, Nn=Nn, nbr_pos=P[nb], nbr_nrm=Nn[nb],
                q=torch.as_tensor(q, dtype=torch.float32),
                phi=torch.as_tensor(phi, dtype=torch.float32))


def analytic_example(rng, n_points, k, n_query, noise_std):
    shape = random_analytic_shape(rng)
    pts, nrm = shape.sample_surface(n_points, rng)
    pts = pts + rng.normal(scale=noise_std, size=pts.shape)        # input noise
    idx = knn_neighborhoods(pts, k)
    surf, _ = shape.sample_surface(n_query, rng)
    nb_band = n_query // 2
    band = surf[:nb_band] + rng.normal(scale=max(3 * noise_std, 0.03), size=(nb_band, 3))
    bulk = rng.uniform(-1, 1, size=(n_query - nb_band, 3))
    q = np.concatenate([band, bulk], 0)
    phi = shape.sdf(q)               # ground truth to the CLEAN analytic surface
    return _pack(pts, nrm, idx, q, phi)


def build_cache(n_analytic, n_modelnet, n_points, k, n_query, noise_std, seed=0):
    """Generate a fixed pool of examples (CPU tensors): analytic + noisy ModelNet40."""
    rng = np.random.default_rng(seed)
    cache = []
    t0 = time.time()
    for i in range(n_analytic):
        cache.append(analytic_example(rng, n_points, k, n_query, noise_std))
        if (i + 1) % 500 == 0:
            print(f"  analytic {i+1}/{n_analytic}  ({(i+1)/(time.time()-t0):.0f}/s)", flush=True)

    if n_modelnet > 0:
        from pat.datasets import modelnet_index
        from train_noisy import make_example, off_nfaces
        paths = [p for p in modelnet_index() if 0 < off_nfaces(p) <= 40000]
        print(f"  ModelNet training set: {len(paths)} models (>=10000); "
              f"caching {n_modelnet} noisy examples", flush=True)
        rng.shuffle(paths)
        got = 0
        for p in paths:
            if got >= n_modelnet:
                break
            try:
                ns = float(rng.uniform(0.005, 0.02))
                cache.append(make_example(p, rng, n_points=n_points, k=k,
                                          n_query=n_query, noise_std=ns))
                got += 1
                if got % 250 == 0:
                    print(f"  modelnet {got}/{n_modelnet}", flush=True)
            except Exception:
                continue
    print(f"cache: {len(cache)} examples in {time.time()-t0:.0f}s", flush=True)
    return cache


# --------------------------------------------------------------------------- #
#  Loss / validation
# --------------------------------------------------------------------------- #
def loss_on(net, ex, eik=0.1):
    coeffs, _, sq = net(ex["nbr_pos"], ex["nbr_nrm"])
    st = (sq[:, 0], sq[:, 1]) if sq is not None else None
    loss, ld, le = pat_loss(ex["P"], ex["Nn"], coeffs, ex["q"], ex["phi"],
                            eikonal_weight=eik, supertoroid=st)
    return loss, ld, le


@torch.no_grad()
def validate_default_torus(net, res=64, bound=1.2, npoints=1024, seed=123):
    """Reconstruct a default torus from a clean cloud; return mean abs SDF error."""
    from pat import PAT
    from pat.shapes import Torus
    net.eval()
    rng = np.random.default_rng(seed)
    sh = Torus(0.6, 0.24)
    pts, nrm = sh.sample_surface(npoints, rng)
    pat = PAT(pts, nrm, model=net.to("cpu"), k=16, C=16)
    grid = rng.uniform(-bound, bound, (4000, 3))
    err = float(np.mean(np.abs(pat.sdf(grid, neighbors=64) - sh.sdf(grid))))
    net.to(DEVICE).train()
    return err


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--cache", type=int, default=4000, help="analytic examples to cache")
    ap.add_argument("--modelnet", type=int, default=1500, help="noisy ModelNet examples to cache")
    ap.add_argument("--n-points", type=int, default=256)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--n-query", type=int, default=192)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--cache-file", default="assets/train_cache.pt")
    ap.add_argument("--outdir", default="assets")
    ap.add_argument("--log-every", type=int, default=400,
                    help="log running training progress every N steps")
    ap.add_argument("--val-every", type=int, default=3,
                    help="reconstruct-a-torus validation every N epochs (it runs on CPU)")
    args = ap.parse_args()

    print(f"device: {DEVICE}", flush=True)
    os.makedirs(args.outdir, exist_ok=True)

    if os.path.exists(args.cache_file):
        print(f"loading cached dataset {args.cache_file}", flush=True)
        cache = torch.load(args.cache_file, weights_only=False)
    else:
        cache = build_cache(args.cache, args.modelnet, args.n_points, args.k,
                            args.n_query, args.noise)
        torch.save(cache, args.cache_file)
    print(f"dataset: {len(cache)} examples", flush=True)
    # Keep the whole cache resident on the GPU so steps aren't bottlenecked by
    # per-step host->device transfers (the 634 MB cache fits easily in 8 GB).
    cache = [{k: v.to(DEVICE) for k, v in ex.items()} for ex in cache]

    cfg_t = dict(d_embed=128, n_layers=6, n_heads=8, d_ff=512, supertoroid=False)
    cfg_s = dict(d_embed=128, n_layers=6, n_heads=8, d_ff=512, supertoroid=True)
    net_t = CoeffNet(**cfg_t).to(DEVICE)
    net_s = CoeffNet(**cfg_s).to(DEVICE)
    opt_t = torch.optim.Adam(net_t.parameters(), lr=args.lr)
    opt_s = torch.optim.Adam(net_s.parameters(), lr=args.lr)
    steps = args.epochs * len(cache)
    sch_t = torch.optim.lr_scheduler.CosineAnnealingLR(opt_t, T_max=steps)
    sch_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=steps)

    best = {"torus": 1e9, "super": 1e9}
    hist = {"torus": [], "super": [], "val_torus": [], "val_super": []}
    vt = vs = 1e9
    rng = np.random.default_rng(0)
    total_steps = args.epochs * len(cache)
    print(f"training {total_steps} steps ({args.epochs} epochs x {len(cache)} examples) "
          f"x 2 models on {torch.cuda.get_device_name(0)}", flush=True)
    t0 = time.time()
    done = 0
    for epoch in range(args.epochs):
        order = rng.permutation(len(cache))
        lt = ls = 0.0
        run_t = run_s = 0.0          # running window loss for live progress
        for j, i in enumerate(order):
            ex = cache[i]                                  # already on GPU
            loss_t, _, _ = loss_on(net_t, ex)
            opt_t.zero_grad(); loss_t.backward()
            torch.nn.utils.clip_grad_norm_(net_t.parameters(), 1.0)
            opt_t.step(); sch_t.step()
            loss_s, _, _ = loss_on(net_s, ex)
            opt_s.zero_grad(); loss_s.backward()
            torch.nn.utils.clip_grad_norm_(net_s.parameters(), 1.0)
            opt_s.step(); sch_s.step()
            vt_, vs_ = float(loss_t.detach()), float(loss_s.detach())
            lt += vt_; ls += vs_; run_t += vt_; run_s += vs_; done += 1
            if done % args.log_every == 0:                 # live intra-epoch progress
                rate = done / (time.time() - t0)
                eta = (total_steps - done) / max(rate, 1e-6)
                print(f"  [ep {epoch:02d} {j+1:4d}/{len(cache)}] step {done}/{total_steps} "
                      f"loss T {run_t/args.log_every:.4f} S {run_s/args.log_every:.4f} "
                      f"| {rate:.0f} it/s | ETA {eta/60:.1f} min", flush=True)
                run_t = run_s = 0.0
        lt /= len(cache); ls /= len(cache)
        hist["torus"].append(lt); hist["super"].append(ls)
        # validation reconstructs a torus on CPU (slow), so only every --val-every epochs
        if epoch % args.val_every == 0 or epoch == args.epochs - 1:
            vt = validate_default_torus(net_t)
            vs = validate_default_torus(net_s)
        hist["val_torus"].append(vt); hist["val_super"].append(vs)
        dt = time.time() - t0
        print(f"epoch {epoch:3d}  loss T {lt:.4f} S {ls:.4f}  "
              f"val-torus-err  T {vt:.4f} S {vs:.4f}   [{dt:.0f}s]", flush=True)
        if vt < best["torus"]:
            best["torus"] = vt
            torch.save({"state_dict": net_t.state_dict(), "config": cfg_t,
                        "val_torus_err": vt, "hist": hist},
                       os.path.join(args.outdir, "pat_torus.pt"))
        if vs < best["super"]:
            best["super"] = vs
            torch.save({"state_dict": net_s.state_dict(), "config": cfg_s,
                        "val_torus_err": vs, "hist": hist},
                       os.path.join(args.outdir, "pat_supertoroid.pt"))
    print(f"DONE. best val-torus-err: torus {best['torus']:.4f}  supertoroid {best['super']:.4f}",
          flush=True)
    print(f"saved {args.outdir}/pat_torus.pt and {args.outdir}/pat_supertoroid.pt", flush=True)


if __name__ == "__main__":
    main()
