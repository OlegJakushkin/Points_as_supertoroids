"""Does the NETWORK earn its keep under NOISE?  full vs anchor-only (untrained = clamped composed anchor) at
position-noise 0 / 1 / 2 / 5% on a 20-shape subset (first shape of every other category).  The training story
says the learned residual pulls the noisy anchor back toward the clean surface -- this measures exactly that.
Writes compare/ablation_noise.json."""
import sys, os, glob, json, time; sys.path.insert(0, "compare")
import numpy as np, torch
from core import sample_path, fscore_curve, chamfer, mesh_defects, DEV, BOUND, TRUNC, mixed_net
from waveshape import wavelet as WV

net = mixed_net()
ck = torch.load("assets/waveshape.pt", weights_only=False)
FM = ck.get("field_mode") or "mixed"
torch.manual_seed(0)
anchor_net = WV.PerceiverWaveNet(res=128, trunc=ck.get("trunc", 0.1), bound=BOUND,
                                 with_seg=ck.get("with_seg", True), unsigned=ck.get("unsigned", False),
                                 field_mode=FM).to(DEV).eval()
NOISE = (0.0, 0.01, 0.02, 0.05)
cats = sorted(os.path.basename(d) for d in glob.glob("data/ModelNet40/*") if os.path.isdir(d))[::2]
shapes = [(c, sorted(glob.glob(f"data/ModelNet40/{c}/test/*.off"))[0]) for c in cats]
print(f"noise ablation: {len(shapes)} shapes x {NOISE}", flush=True)


def run(n, Pt, Nt):
    with torch.no_grad():
        raw = n(Pt, Nt)[0][0, 0].cpu().numpy() * TRUNC
    return WV.mesh_field(raw, FM, bound=BOUND, trunc=TRUNC)


rows, t0 = [], time.time()
for i, (cat, path) in enumerate(shapes):
    for nz in NOISE:
        try:
            gt, P, N = sample_path(path, n=4096, noise=nz, seed=0)
            Pt = torch.tensor(P[None]).float().to(DEV); Nt = torch.tensor(N[None]).float().to(DEV)
            row = {"cat": cat, "noise": nz, "methods": {}}
            for name, mdl in (("full", net), ("anchor", anchor_net)):
                v, f = run(mdl, Pt, Nt)
                curve, auc = fscore_curve(v, f, gt)
                d = mesh_defects(v, f)
                row["methods"][name] = {"f_curve": {str(k): round(x, 2) for k, x in curve.items()},
                                        "chamfer": chamfer(v, f, gt), "parts": d["components"],
                                        "holes": d["boundary_edges"]}
            rows.append(row)
        except Exception as e:
            print(f"  skip {cat}@{nz}: {e}", flush=True)
    json.dump(rows, open("compare/ablation_noise.json", "w"), indent=1)
    print(f"  [{i+1}/{len(shapes)}] {cat:12s} {time.time()-t0:.0f}s", flush=True)
json.dump(rows, open("compare/ablation_noise.json", "w"), indent=1)
print(f"wrote compare/ablation_noise.json ({len(rows)} rows, {time.time()-t0:.0f}s)", flush=True)
