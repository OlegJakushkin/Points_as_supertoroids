"""Reviewer-demanded ablation on the 80-shape benchmark: WHAT DOES THE NETWORK ADD OVER ITS OWN ANCHOR?
Rows per shape: full model | full w/o floater-collapse (post-processing fairness) | anchor-only (an UNTRAINED
net: zero-init heads + refiner => exactly the clamped region-composed anchor, same canonical mesher) | direct
non-composed nearest-point TSDF (same mesher; isolates the region composition).  Baselines re-run for the
F(tau) sweep.  Also per shape: F(tau) curves + AUC (does the eps-offset absorb the tolerance?), directional
Chamfer split recon->GT vs GT->recon (offset decomposition), learned-residual magnitude ||c_clean-c_anchor|| /
||c_anchor|| (whole + detail bands), gate stratum at thin delta x{0.75,1,1.25} (stratification sensitivity),
and the GT watertight flag (gate-independent stratification).  Writes compare/ablation.json."""
import sys, os, glob, json, time; sys.path.insert(0, "compare")
import numpy as np, trimesh, torch
from skimage import measure
from scipy.spatial import cKDTree
from core import sample_path, mesh_defects, fscore_curve, chamfer, DEV, BOUND, TRUNC, mixed_net
import baselines as B
from waveshape import wavelet as WV

K = int(os.environ.get("VAL_K", "2"))
N_PTS = 4096
TAUS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1)
OUT = os.environ.get("OUT", "compare/ablation.json")

net = mixed_net()                                        # trained released checkpoint @128
ck = torch.load("assets/waveshape.pt", weights_only=False)
FM = ck.get("field_mode") or "mixed"
torch.manual_seed(0)
anchor_net = WV.PerceiverWaveNet(res=128, trunc=ck.get("trunc", 0.1), bound=BOUND,
                                 with_seg=ck.get("with_seg", True), unsigned=ck.get("unsigned", False),
                                 field_mode=FM).to(DEV).eval()   # UNTRAINED: zero-init => clamp(anchor)


def mesh_from_grid(raw, collapse=True):
    if collapse:
        return WV.mesh_field(raw, FM, bound=BOUND, trunc=TRUNC)
    g = WV._smooth_grid(raw, 0.5)                        # same light blur, NO keep_main_grid collapse
    if not (g.min() < 0 < g.max()):
        return None, None
    v, f, _, _ = measure.marching_cubes(g.astype(np.float64), 0.0)
    return v / (g.shape[0] - 1) * (2 * BOUND) - BOUND, f


def directional(v, f, gt, n=30000):
    """Mean recon->GT and GT->recon distances (x100): splits Chamfer into the offset-shell component
    (recon->GT ~ eps for the unsigned band) vs missing-geometry error (GT->recon)."""
    if v is None or not len(f):
        return -1.0, -1.0
    a = trimesh.Trimesh(v, f, process=False).sample(n); b, _ = trimesh.sample.sample_surface(gt, n)
    da, _ = cKDTree(b).query(a); db, _ = cKDTree(a).query(b)
    return float(da.mean() * 100), float(db.mean() * 100)


def rec(v, f, gt, sec):
    curve, auc = fscore_curve(v, f, gt, taus=TAUS)
    d = mesh_defects(v, f)
    r2g, g2r = directional(v, f, gt)
    return {"f_curve": {str(k): round(x, 2) for k, x in curve.items()}, "f_auc": round(auc, 2),
            "chamfer": chamfer(v, f, gt), "r2g": r2g, "g2r": g2r,
            "holes": d["boundary_edges"], "self_x": d["self_intersections"], "parts": d["components"],
            "watertight": d["watertight"], "time": sec, "faces": 0 if f is None else int(len(f))}


cats = sorted(os.path.basename(d) for d in glob.glob("data/ModelNet40/*") if os.path.isdir(d))
shapes = [(c, p) for c in cats for p in sorted(glob.glob(f"data/ModelNet40/{c}/test/*.off"))[:K]]
print(f"ablation over {len(shapes)} shapes ({len(cats)} cats, K={K})", flush=True)
av = B.available()
rows, t0 = [], time.time()
for i, (cat, path) in enumerate(shapes):
    try:
        gt, P, N = sample_path(path, n=N_PTS, noise=0.0, seed=0)
        Pt = torch.tensor(P[None]).float().to(DEV); Nt = torch.tensor(N[None]).float().to(DEV)
        Pc, Nc = Pt.cpu(), Nt.cpu()
        tf = {s: float(WV.point_thinness(Pc, Nc, thin=0.10 * s).mean()) for s in (0.75, 1.0, 1.25)}
        row = {"cat": cat, "file": os.path.basename(path), "gt_watertight": bool(gt.is_watertight),
               "thin_frac": tf[1.0], "thin_frac_lo": tf[0.75], "thin_frac_hi": tf[1.25], "methods": {}}
        # FULL (one forward, two meshings) + learned-residual magnitude
        t = time.time()
        with torch.no_grad():
            out, c_anchor, c_clean, _ = net(Pt, Nt)
        raw = out[0, 0].cpu().numpy() * TRUNC; sec_fwd = time.time() - t
        dc = c_clean - c_anchor
        row["resid_ratio"] = float(dc.norm() / (c_anchor.norm() + 1e-9))
        row["resid_ratio_detail"] = float(dc[:, 1:].norm() / (c_anchor[:, 1:].norm() + 1e-9))
        v, f = mesh_from_grid(raw, collapse=True);  row["methods"]["full"] = rec(v, f, gt, sec_fwd)
        v, f = mesh_from_grid(raw, collapse=False); row["methods"]["full_nocollapse"] = rec(v, f, gt, sec_fwd)
        # ANCHOR-ONLY: untrained net = clamped composed anchor through the identical pipeline
        t = time.time()
        with torch.no_grad():
            outa = anchor_net(Pt, Nt)[0]
        rawa = outa[0, 0].cpu().numpy() * TRUNC; seca = time.time() - t
        v, f = mesh_from_grid(rawa, collapse=True); row["methods"]["anchor"] = rec(v, f, gt, seca)
        # DIRECT non-composed nearest-point TSDF, same canonical mesher (isolates the composition)
        t = time.time()
        rawd = WV.tsdf_from_clouds(Pt, Nt, 128, TRUNC, BOUND, DEV, mode=FM)[0, 0].cpu().numpy() * TRUNC
        secd = time.time() - t
        v, f = mesh_from_grid(rawd, collapse=True); row["methods"]["direct"] = rec(v, f, gt, secd)
        # library baselines, re-run for the F(tau) sweep
        for name, fn in av.items():
            try:
                v, f, dt = fn(P, N); row["methods"][name] = rec(v, f, gt, dt)
            except Exception:
                pass
        rows.append(row)
        if len(rows) % 10 == 0 or i == 0:
            json.dump(rows, open(OUT, "w"), indent=1)
            m = row["methods"]
            print(f"  [{i+1}/{len(shapes)}] {cat:12s} F.05 full {m['full']['f_curve']['0.05']:.0f} "
                  f"anchor {m['anchor']['f_curve']['0.05']:.0f} direct {m['direct']['f_curve']['0.05']:.0f} "
                  f"| resid {row['resid_ratio']:.4f} | {time.time()-t0:.0f}s", flush=True)
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  skip {cat}/{os.path.basename(path)}: {e}", flush=True)
json.dump(rows, open(OUT, "w"), indent=1)
print(f"wrote {OUT} ({len(rows)} shapes, {time.time()-t0:.0f}s)", flush=True)
