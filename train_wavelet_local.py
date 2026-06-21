"""Local GPU head-to-head on 5 canonical shapes (torus, teapot, bunny, cube, sphere):
train BOTH the original tori CoeffNet and the wavelet denoiser on the SAME shapes/clouds,
then render `ground truth | tori | wavelet` per shape and score MD / IoU* / Chamfer for both.

The point of this debug run: make sure NEITHER reconstruction is a "disjoint mess" — the
tori net must blend into a coherent surface and the wavelet net must denoise (not pass the
noisy TSDF through).

GPU + Docker only (policy):
  docker compose run --rm train python train_wavelet_local.py
  docker compose run --rm train python train_wavelet_local.py --res 64 --wave-epochs 500

NOISE IS DYNAMIC: a fresh noisy cloud (random magnitude in [noise_lo, noise_hi]) is drawn
every epoch and re-voxelized, so the wavelet net learns a true noise-robust denoiser (never a
fixed pair).  Several fresh draws per shape are averaged for a low-variance gradient so it
converges on just 5 shapes; the bank is refreshed each epoch (dynamic) but reused for a few
sub-steps so the (expensive) TSDF rebuild is amortized.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as Fn
import trimesh

from pat import wavelet as WV
from pat import compare as CMP
from pat import eval3d as E
from pat import render3d as R3
from pat.pat import PAT
from pat.shapes import normalize_to_unit_cube
from pat.bunny import load_bunny


def five_shapes():
    return [
        ("torus",  normalize_to_unit_cube(trimesh.creation.torus(major_radius=0.5, minor_radius=0.2))),
        ("teapot", normalize_to_unit_cube(E._teapot_mesh())),
        ("bunny",  load_bunny(normalize=True)),
        ("cube",   normalize_to_unit_cube(trimesh.creation.box(extents=[1.0, 1.0, 1.0]))),
        ("sphere", normalize_to_unit_cube(trimesh.creation.uv_sphere(radius=0.7, count=[48, 48]))),
    ]


def build_cache(shapes, gts, dense, n_query, bound, seed=0):
    """{P,N,Q,PHI} for the tori trainer: clean cloud + GT signed-distance at band+bulk queries."""
    rng = np.random.default_rng(seed)
    Ps, Ns, Qs, PHIs = [], [], [], []
    for (name, m), gt in zip(shapes, gts):
        P, N = E.sample_cloud(m, n=dense, noise=0.0, seed=seed)
        surf, _ = E.sample_cloud(m, n=n_query, noise=0.0, seed=seed + 1)
        band = surf + rng.normal(scale=0.04, size=surf.shape)
        bulk = rng.uniform(-bound, bound, size=(n_query, 3))
        q = np.concatenate([band[: n_query // 2], bulk[: n_query - n_query // 2]], 0).astype(np.float32)
        phi = gt.sdf(q).astype(np.float32)
        Ps.append(P); Ns.append(N); Qs.append(q); PHIs.append(phi)
    t = lambda a: torch.as_tensor(np.stack(a))
    return {"P": t(Ps), "N": t(Ns), "Q": t(Qs), "PHI": t(PHIs)}


def train_wavelet_dynamic(net, Ps, Ns, *, res, trunc, bound, epochs, draws, substeps,
                          noise_lo, noise_hi, lam_wave, lam_grad, lr, dev, log_every):
    """Dynamic-noise denoiser training: fresh noisy bank each epoch (random magnitude),
    `draws` draws/shape averaged (low variance), `substeps` SGD steps per refreshed bank."""
    S = len(Ps); M = max(1, draws)
    Pt = torch.tensor(np.stack(Ps)).to(dev); Nt = torch.tensor(np.stack(Ns)).to(dev)
    haar = WV.haar_filters_3d(dev)
    with torch.no_grad():
        clean = WV.tsdf_from_clouds(Pt, Nt, res, trunc, bound, dev) / trunc           # (S,1,R,R,R)
        target_c = WV.dwt3d(clean, haar)
        clean_r = clean.repeat(M, 1, 1, 1, 1); tc_r = target_c.repeat(M, 1, 1, 1, 1)
        Pr, Nr = Pt.repeat(M, 1, 1), Nt.repeat(M, 1, 1)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    g = torch.Generator(device="cpu").manual_seed(0)
    net.train()
    for ep in range(epochs):
        with torch.no_grad():                                                         # DYNAMIC refresh
            ns = torch.empty(S * M, 1, 1, device=dev).uniform_(noise_lo, noise_hi)
            noisy = WV.tsdf_from_clouds(Pr + torch.randn(Pr.shape, device=dev) * ns,
                                        Nr, res, trunc, bound, dev) / trunc
        for _ in range(substeps):
            idx = torch.randperm(S * M, generator=g)[: max(S, (S * M) // substeps)].to(dev)
            pred, _, c_pred = net(noisy[idx])
            l_t = Fn.smooth_l1_loss(pred, clean_r[idx], beta=0.1)
            l_w = (c_pred - tc_r[idx]).abs().mean()
            gp, gc = WV._grad3d(pred), WV._grad3d(clean_r[idx])
            l_g = sum((a - b).abs().mean() for a, b in zip(gp, gc)) / 3.0
            loss = l_t + lam_wave * l_w + lam_grad * l_g
            opt.zero_grad(); loss.backward()
            for p in net.parameters():
                if p.grad is not None:
                    torch.nan_to_num_(p.grad, 0., 0., 0.)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            with torch.no_grad():
                Pv = Pt + torch.randn(Pt.shape, device=dev) * (0.5 * (noise_lo + noise_hi))
                held = float((net(WV.tsdf_from_clouds(Pv, Nt, res, trunc, bound, dev) / trunc)[0]
                              - clean).abs().mean())
            print(f"  wavelet ep {ep:4d}: loss {float(loss.detach()):.4f} | held-out TSDF-L1 {held:.4f}",
                  flush=True)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=64, help="wavelet TSDF resolution (mult of 8)")
    ap.add_argument("--trunc", type=float, default=0.1)
    ap.add_argument("--base", type=int, default=40)
    ap.add_argument("--dense", type=int, default=2048, help="cloud points per shape")
    ap.add_argument("--nquery", type=int, default=4096, help="GT SDF queries per shape (tori loss)")
    # wavelet (dynamic noise)
    ap.add_argument("--wave-epochs", type=int, default=400)
    ap.add_argument("--draws", type=int, default=8, help="fresh noise draws/shape averaged per epoch")
    ap.add_argument("--substeps", type=int, default=4, help="SGD steps per refreshed noisy bank")
    ap.add_argument("--noise-lo", type=float, default=0.005)
    ap.add_argument("--noise-hi", type=float, default=0.03)
    ap.add_argument("--lam-wave", type=float, default=0.3)
    ap.add_argument("--lam-grad", type=float, default=0.05)
    ap.add_argument("--wave-lr", type=float, default=2e-3)
    # tori
    ap.add_argument("--tori-ckpt", default="", help="load a pre-trained CoeffNet (e.g. "
                    "assets/pat_torus.pt) instead of training one on the 5 shapes")
    ap.add_argument("--tori-epochs", type=int, default=300)
    ap.add_argument("--tori-batch", type=int, default=5)
    ap.add_argument("--tori-npoints", type=int, default=1024)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--res-recon", type=int, default=96, help="tori marching-cubes grid")
    # eval
    ap.add_argument("--eval-noise", type=float, default=0.015, help="noise on the held-out eval clouds")
    ap.add_argument("--tag", default="h2h")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("GPU-only — run via docker compose run --rm train python train_wavelet_local.py")
    dev = "cuda"; bound = 1.1
    os.makedirs("renders", exist_ok=True); os.makedirs("assets", exist_ok=True)
    print(f"GPU {torch.cuda.get_device_name(0)} | res {args.res} base {args.base}", flush=True)

    shapes = five_shapes()
    gts = [E.mesh_gt(m) for _, m in shapes]
    Ps, Ns = [], []
    for _, m in shapes:
        P, N = E.sample_cloud(m, n=args.dense, noise=0.0, seed=0)
        Ps.append(P); Ns.append(N)

    # ---- Network A: tori CoeffNet --------------------------------------------
    if args.tori_ckpt and os.path.exists(args.tori_ckpt):
        ck = torch.load(args.tori_ckpt, weights_only=False)
        cfg = ck.get("config", {"d_embed": ck.get("d_embed", 128), "n_layers": ck.get("n_layers", 8)})
        tori = CMP.CoeffNet(**cfg).to(dev)
        tori.load_state_dict(ck.get("state_dict", ck.get("state")))
        print(f"\n== loaded pre-trained TORI {args.tori_ckpt} (config {cfg}) ==", flush=True)
    else:
        cache = build_cache(shapes, gts, args.dense, args.nquery, bound, seed=0)
        print(f"\n== training TORI CoeffNet on the 5 shapes ({args.tori_epochs} epochs) ==", flush=True)
        tori, _ = CMP.train_tori_cache(cache, k=args.k, epochs=args.tori_epochs, batch=args.tori_batch,
                                       n_points=args.tori_npoints, noise_std=args.eval_noise,
                                       d_embed=128, n_layers=8, device=dev, n_val=0,
                                       log_every=max(1, args.tori_epochs // 10), seed=0)
        torch.save({"state": tori.state_dict(), "d_embed": 128, "n_layers": 8}, "assets/tori_5shapes.pt")

    # ---- Network B: wavelet denoiser (dynamic noise) -------------------------
    print(f"\n== training WAVELET denoiser on the 5 shapes ({args.wave_epochs} epochs, dynamic noise) ==",
          flush=True)
    wave = WV.WaveletDenoiser(base=args.base).to(dev)
    print(f"wavelet params {wave.count_params():,} | tori params {sum(p.numel() for p in tori.parameters()):,}",
          flush=True)
    wave = train_wavelet_dynamic(wave, Ps, Ns, res=args.res, trunc=args.trunc, bound=bound,
                                 epochs=args.wave_epochs, draws=args.draws, substeps=args.substeps,
                                 noise_lo=args.noise_lo, noise_hi=args.noise_hi,
                                 lam_wave=args.lam_wave, lam_grad=args.lam_grad, lr=args.wave_lr,
                                 dev=dev, log_every=max(1, args.wave_epochs // 14))
    torch.save({"state": wave.state_dict(), "base": args.base, "res": args.res, "trunc": args.trunc},
               "assets/wavelet_5shapes.pt")

    # ---- head-to-head reconstruct + render -----------------------------------
    print(f"\n{'shape':8s} | {'tori MD':>8s} {'tori IoU':>8s} {'tori vts':>8s} | "
          f"{'wav MD':>7s} {'wav IoU':>8s} {'wav band%':>9s} {'wav vts':>8s}", flush=True)
    rng = np.random.default_rng(0); mt, iw_t, mw, iw_w = [], [], [], []
    for (name, m), P, N, gt in zip(shapes, Ps, Ns, gts):
        Pn = (P + rng.normal(scale=args.eval_noise, size=P.shape)).astype(np.float32)   # SAME noisy cloud
        # tori
        pat = PAT(Pn, N, model=tori, k=args.k, C=64.0, device=dev)
        pm_t = E.proper_metrics(gt, CMP._SdfAdapter(lambda q: pat.sdf(q, neighbors=64)), n=40000)
        vt, ft = pat.reconstruct(res=args.res_recon, bound=bound, neighbors=64)
        # wavelet
        wr = WV.WaveletReconstruction(Pn, N, wave, res=args.res, trunc=args.trunc, bound=bound, device=dev)
        pm_w = E.proper_metrics(gt, wr, n=40000)
        vw, fw = wr.reconstruct()
        band = float((np.abs(wr.grid) < 0.3 * args.trunc).mean())
        mt.append(pm_t["md"]); iw_t.append(pm_t["iou"]); mw.append(pm_w["md"]); iw_w.append(pm_w["iou"])
        nvt = 0 if vt is None else len(vt); nvw = 0 if vw is None else len(vw)
        print(f"{name:8s} | {pm_t['md']:8.3f} {pm_t['iou']:8.3f} {nvt:8d} | "
              f"{pm_w['md']:7.3f} {pm_w['iou']:8.3f} {100*band:9.2f} {nvw:8d}", flush=True)
        try:
            R3.render_meshes([("ground truth", m.vertices, m.faces),
                              (f"tori  MD {pm_t['md']:.3f} | IoU* {pm_t['iou']:.2f}", vt, ft),
                              (f"wavelet  MD {pm_w['md']:.3f} | IoU* {pm_w['iou']:.2f}", vw, fw)],
                             f"renders/h2h_{name}_{args.tag}.png", title=name)
        except Exception as exc:
            print("  render skip", name, exc, flush=True)
    print(f"\nMEAN: tori MD {np.mean(mt):.3f} IoU* {np.mean(iw_t):.3f} | "
          f"wavelet MD {np.mean(mw):.3f} IoU* {np.mean(iw_w):.3f} | renders -> renders/h2h_*_{args.tag}.png",
          flush=True)


if __name__ == "__main__":
    main()
