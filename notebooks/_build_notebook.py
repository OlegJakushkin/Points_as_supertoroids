"""Generate notebooks/train_pat_colab.ipynb (kept as a generator to avoid hand-editing JSON)."""
import json, os

def md(*lines): return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}
def code(*lines): return {"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": _src(lines)}
def _src(lines):
    flat = []
    for l in lines:
        flat.extend(l.split("\n"))
    return [s + "\n" for s in flat[:-1]] + [flat[-1]] if flat else []

cells = []

cells.append(md(
"# Points as **Supertoroids** — self-contained training notebook",
"",
"Trains the per-point coefficient network from *Feng, Gkioulekas & Crane, \"Points as Tori\"*",
"(ACM TOG 2026), in the **supertoroid** generalization, and saves a checkpoint that plugs",
"directly into the `pat` library, its tests, and the reconstruction/visualization code.",
"",
"Runs end-to-end on Google Colab (CPU or GPU). It uses the **same `pat` package** as the",
"rest of the project, so there is no risk of the notebook and library drifting apart.",
"",
"**What it does**",
"1. Sets up the `pat` package (clones your repo, or uses a local copy).",
"2. Generates synthetic training data on the fly from analytic SDF primitives",
"   (spheres, tori, supertoroids, rounded boxes, planes) — exact ground-truth distance.",
"3. Trains `CoeffNet(supertoroid=True)` with the paper's L1 + eikonal blend loss (Eq. 27),",
"   **in the presence of noise**, on either noisy synthetic shapes or >=10,000 real ModelNet40",
"   models (`DATA_MODE`), following a sphere → torus → supertoroid → mixed curriculum.",
"4. Saves `pat_supertoroid.pt` and shows how to load it back into `pat.PAT`.",
))

cells.append(md("## 1. Setup"))
cells.append(code(
"# If running on Colab, point this at YOUR fork/repo so the notebook uses the exact",
"# same `pat` package as your tests. Leave empty if the `pat/` folder is already importable",
"# (e.g. you uploaded it, or you are running inside the repo).",
"REPO_URL = \"\"   # e.g. \"https://github.com/<you>/Points_as_supertoroids.git\"",
"",
"import importlib, subprocess, sys, os",
"",
"def _have_pat():",
"    try:",
"        importlib.import_module('pat'); return True",
"    except Exception:",
"        return False",
"",
"if not _have_pat():",
"    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',",
"                    'torch', 'numpy', 'scipy', 'scikit-image', 'matplotlib'], check=False)",
"    if REPO_URL:",
"        subprocess.run(['git', 'clone', '-q', REPO_URL, 'pat_repo'], check=False)",
"        sys.path.insert(0, os.path.abspath('pat_repo'))",
"    # also try a couple of common local locations",
"    for p in ['.', '..', 'Points_as_supertoroids']:",
"        if os.path.isdir(os.path.join(p, 'pat')):",
"            sys.path.insert(0, os.path.abspath(p))",
"",
"assert _have_pat(), ('Could not import `pat`. Set REPO_URL to your repo, or upload the '",
"                     '`pat/` folder to this Colab session.')",
"import torch, numpy as np, matplotlib.pyplot as plt",
"from pat import core, shapes",
"from pat.model import CoeffNet",
"from pat.neighbors import knn_neighborhoods",
"from pat.train import pat_loss, sample_queries",
"device = 'cuda' if torch.cuda.is_available() else 'cpu'",
"print('pat ready — device:', device)",
))

cells.append(md(
"## 2. Synthetic training data",
"",
"Each training example is a small point cloud sampled from a random analytic shape, with",
"**exact** ground-truth signed distance. We mix primitive families and randomize their",
"parameters, scale and pose. Supertoroid targets (boxy tubes) are what teach the network",
"to use the squareness exponents.",
))
cells.append(code(
"K = 16             # neighbors per neighborhood (paper uses k-NN attention)",
"N_POINTS = 256     # points per training cloud",
"BOUND = 1.0        # cube half-extent for bulk query sampling",
"NOISE_STD = 0.01   # Gaussian noise added to the INPUT cloud (gt distance stays to the clean surface)",
"",
"def random_shape(rng, family=None):",
"    fam = family or rng.choice(['sphere', 'torus', 'supertoroid', 'rbox', 'plane'])",
"    if fam == 'sphere':",
"        return shapes.Sphere(radius=rng.uniform(0.3, 0.8))",
"    if fam == 'torus':",
"        R = rng.uniform(0.4, 0.7); r = rng.uniform(0.12, 0.3) * R / 0.6",
"        return shapes.Torus(R=R, r=r, axis=rng.normal(size=3))",
"    if fam == 'supertoroid':",
"        R = rng.uniform(0.4, 0.7); r = rng.uniform(0.15, 0.32) * R / 0.6",
"        return shapes.SuperToroid(R=R, r=r, p_tube=rng.uniform(2.0, 5.0),",
"                                  p_ring=rng.uniform(2.0, 3.5), axis=rng.normal(size=3))",
"    if fam == 'rbox':",
"        h = rng.uniform(0.3, 0.6, size=3)",
"        return shapes.RoundedBox(half=h, radius=rng.uniform(0.05, 0.15))",
"    return shapes.Plane(normal=rng.normal(size=3), extent=1.0)",
"",
"def make_example(rng, family=None, noise_std=None):",
"    shape = random_shape(rng, family)",
"    pts, nrm = shape.sample_surface(N_POINTS, rng)",
"    ns = NOISE_STD if noise_std is None else noise_std",
"    if ns > 0:                       # noise on the INPUT cloud only",
"        pts = pts + rng.normal(scale=ns, size=pts.shape)",
"    idx = knn_neighborhoods(pts, K)",
"    nb = torch.as_tensor(idx, dtype=torch.long)",
"    P = torch.as_tensor(pts, dtype=torch.float32)",
"    Nn = torch.as_tensor(nrm, dtype=torch.float32)",
"    Nn = Nn / Nn.norm(dim=1, keepdim=True).clamp_min(1e-9)",
"    q, phi, _ = sample_queries(shape, N_POINTS, N_POINTS, BOUND, rng)  # gt = CLEAN surface",
"    return dict(P=P.to(device), Nn=Nn.to(device), nbr_pos=P[nb].to(device),",
"               nbr_nrm=Nn[nb].to(device), q=torch.as_tensor(q).to(device),",
"               phi=torch.as_tensor(phi).to(device))",
"",
"# quick look at one example",
"_ex = make_example(np.random.default_rng(0), 'supertoroid')",
"print({k: tuple(v.shape) for k, v in _ex.items()})",
))

cells.append(md(
"## 2b. Real data with noise — ModelNet40 (>= 10,000 models)",
"",
"To train on real geometry, download **ModelNet40** (12,311 CAD models) and stream **noisy**",
"point clouds from it with exact mesh ground-truth distance, via `pat.datasets`. This is the",
"setting the paper uses (ABC + procedural shapes); ModelNet40 is a convenient public stand-in",
"that satisfies the >=10,000-model requirement. Set `DATA_MODE='modelnet'` to use it; the",
"synthetic path above (now also noisy) is the fast default.",
))
cells.append(code(
"DATA_MODE = 'synthetic'      # 'synthetic' (fast) or 'modelnet' (>=10k real models)",
"",
"PATHS = []",
"if DATA_MODE == 'modelnet':",
"    import os, urllib.request, zipfile",
"    import pat.datasets as D",
"    root = 'data'; os.makedirs(root, exist_ok=True)",
"    if not D.modelnet_index(root):",
"        zp = os.path.join(root, 'ModelNet40.zip')",
"        if not os.path.exists(zp):",
"            print('downloading ModelNet40 (~2GB)...')",
"            urllib.request.urlretrieve('http://modelnet.cs.princeton.edu/ModelNet40.zip', zp)",
"        with zipfile.ZipFile(zp) as z: z.extractall(root)",
"        offs = [os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(root, 'ModelNet40'))",
"                for f in fs if f.endswith('.off')]",
"        open(os.path.join(root, 'modelnet40_index.txt'), 'w').write('\\n'.join(sorted(offs)))",
"    PATHS = D.modelnet_index(root)",
"    print(f'ModelNet40 training set: {len(PATHS)} real models (>= 10000)')",
"",
"def training_example(rng, family=None):",
"    'Unified example source: noisy ModelNet mesh, or noisy synthetic shape.'",
"    if DATA_MODE == 'modelnet':",
"        import pat.datasets as D",
"        path = PATHS[rng.integers(len(PATHS))]",
"        ex = D.make_training_example(path, rng, n_points=N_POINTS, k=K,",
"                                     n_query=N_POINTS, noise_std=NOISE_STD)",
"        return {k: v.to(device) for k, v in ex.items()}",
"    return make_example(rng, family)",
))

cells.append(md(
"## 3. Model and loss",
"",
"`CoeffNet(supertoroid=True)` predicts the six polynomial coefficients **and** two squareness",
"logits per point. It is initialized to a tangent sphere with circular cross-sections, i.e.",
"the exact paper torus — so training starts from the published model and only specializes.",
))
cells.append(code(
"net = CoeffNet(d_embed=128, n_layers=8, n_heads=8, d_ff=512, supertoroid=True).to(device)",
"print('parameters:', sum(p.numel() for p in net.parameters()))",
"",
"def loss_on(example, eik=0.1):",
"    coeffs, _, sq = net(example['nbr_pos'], example['nbr_nrm'])",
"    st = (sq[:, 0], sq[:, 1]) if sq is not None else None",
"    return pat_loss(example['P'], example['Nn'], coeffs, example['q'], example['phi'],",
"                    eikonal_weight=eik, supertoroid=st)",
))

cells.append(md(
"## 4. Curriculum training",
"",
"A lightweight version of the paper's curriculum: start on easy round shapes, then mix in",
"boxy supertoroids. Increase `PHASES`/steps for a stronger model. On Colab CPU this is",
"minutes; on GPU, seconds per hundred steps.",
))
cells.append(code(
"from itertools import count",
"opt = torch.optim.Adam(net.parameters(), lr=1e-3)",
"rng = np.random.default_rng(0)",
"",
"# (family-mix, steps, lr) per phase — small by default so the notebook finishes quickly.",
"PHASES = [",
"    (['sphere', 'plane'],                      300, 1e-3),",
"    (['sphere', 'torus'],                      400, 5e-4),",
"    (['torus', 'supertoroid'],                 600, 3e-4),",
"    (['sphere','torus','supertoroid','rbox','plane'], 800, 1e-4),",
"]",
"",
"history = []",
"for pi, (fams, steps, lr) in enumerate(PHASES):",
"    for g in opt.param_groups: g['lr'] = lr",
"    for s in range(steps):",
"        fam = fams[rng.integers(len(fams))]",
"        ex = training_example(rng, fam)   # noisy ModelNet or noisy synthetic",
"        loss, ld, le = loss_on(ex)",
"        opt.zero_grad(); loss.backward()",
"        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)",
"        opt.step()",
"        history.append(float(loss.detach()))",
"        if s % 100 == 0:",
"            print(f'phase {pi} step {s:4d}  loss {float(loss):.4f}  '",
"                  f'dist {float(ld):.4f}  eik {float(le):.4f}')",
"print('done. final avg loss:', np.mean(history[-50:]))",
))
cells.append(code(
"plt.figure(figsize=(7,3))",
"plt.plot(np.convolve(history, np.ones(25)/25, 'valid'))",
"plt.xlabel('step'); plt.ylabel('loss (smoothed)'); plt.title('PAT supertoroid training'); plt.show()",
))

cells.append(md("## 5. Save the checkpoint"))
cells.append(code(
"ckpt = {'state_dict': net.state_dict(),",
"        'config': dict(d_embed=128, n_layers=8, n_heads=8, d_ff=512, supertoroid=True)}",
"torch.save(ckpt, 'pat_supertoroid.pt')",
"print('saved pat_supertoroid.pt')",
"try:",
"    from google.colab import files; files.download('pat_supertoroid.pt')",
"except Exception:",
"    pass",
))

cells.append(md(
"## 6. Plug the trained model into the library",
"",
"The checkpoint is a plain `state_dict` for `pat.model.CoeffNet`, so it loads straight into",
"`pat.PAT(model=...)` and every test/visualization path. Below we reconstruct a supertoroid",
"and compare the learned supertoroid SDF against an ordinary torus fit.",
))
cells.append(code(
"from pat import PAT",
"ckpt = torch.load('pat_supertoroid.pt', map_location='cpu')",
"model = CoeffNet(**ckpt['config']); model.load_state_dict(ckpt['state_dict']); model.eval()",
"",
"rng = np.random.default_rng(7)",
"shape = shapes.SuperToroid(R=0.6, r=0.28, p_tube=4.0, p_ring=2.0)",
"pts, nrm = shape.sample_surface(2048, rng)",
"",
"pat_learned = PAT(pts, nrm, model=model)          # supertoroid model -> supertoroids",
"pat_torus   = PAT(pts, nrm, k=24)                  # training-free torus baseline",
"",
"grid = rng.uniform(-1, 1, (4000, 3)); gt = shape.sdf(grid)",
"err_learned = np.mean(np.abs(pat_learned.sdf(grid, neighbors=64) - gt))",
"err_torus   = np.mean(np.abs(pat_torus.sdf(grid, neighbors=64) - gt))",
"print(f'learned supertoroid err: {err_learned:.4f}')",
"print(f'baseline torus      err: {err_torus:.4f}')",
"print('mean learned p_tube:', float(pat_learned.p_tube.mean()),",
"      ' p_ring:', float(pat_learned.p_ring.mean()))",
))
cells.append(md(
"To visualize locally (not on Colab), use polyscope:",
"```python",
"from pat import viz",
"viz.init()",
"viz.register_point_cloud('input', pts, nrm)",
"viz.register_reconstruction('recon', pat_learned)",
"viz.register_sdf_slice('slice', pat_learned)",
"viz.show()",
"```",
))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}

out = os.path.join(os.path.dirname(__file__), "train_pat_colab.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, "with", len(cells), "cells")
