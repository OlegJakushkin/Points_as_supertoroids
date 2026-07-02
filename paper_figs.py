"""paper_figs.py -- ONE consolidated GPU script regenerating every model-derived PNG figure used by
paper/paper2.tex from the RELEASED region-composed checkpoint (assets/waveshape.pt, PerceiverWaveNet,
mixed base, flexible 128-token split, far-field clamp + wavelet edge refiner).

Figures (all written under paper/figs/):
  principle.png       -- closed solid vs open shell under the UDF / SDF / mixed bases (2x3 grid).
                         PORT of _archive/gen_paper2_figs.py::fig_principle.  ADAPTATION: the single-base
                         UDF/SDF checkpoints no longer exist (only the unified mixed model is released),
                         so the UDF and SDF columns show the DIRECT single-base point fields
                         (WV.tsdf_from_clouds mode='unsigned'/'signed' -- the anchor family the nets are
                         trained on); the pathology is a property of the BASE, not of a particular net.
                         The mixed column is the released model.
  sdf_slices.png      -- 2D field slices: (a) teapot signed SDF, (b) thin chair signed = the OVER-FILL,
                         (c) thin chair unsigned band = the FIX.  PORT of _archive/sdf_slices.py.
                         ADAPTATION: (a) and (c) come from the released mixed net (closed teapot -> its
                         field IS a signed SDF; all-thin chair -> its field IS the unsigned band); (b)
                         uses the direct signed field (the retired signed-net checkpoint is gone).
  usdf_gate.png       -- per-point base-selection gate on cube / bunny / chair (blue=SDF, green=UDF).
                         PORT of _archive/gen_usdf_gate.py, now via the canonical WV.point_thinness.
  flexsplit.png       -- the flexible token budget swept over n_ctx in {16,40,64,88,104} on
                         bunny/teapot/chair.  PORT of _archive/gen_paper2_figs.py::fig_flexsplit.
                         Runs at res 64 (the archived figure's lattice, via net.set_res) so the
                         face-count labels stay comparable; everything else is meshed at 128^3.
  superres_demo.png   -- context+dense super-resolution on the knurl (same 48^3 output lattice spent on
                         the whole shape vs on one box-normalised region).  PORT of
                         _archive/superres_demo.py (res 48 via net.set_res, as in the original).
  compare_models.png  -- GT | tori (CoeffNet blend) | mixed (ours) on six shapes with Chamfer + #parts.
                         PORT of _archive/gen_paper2_figs.py::fig_compare; tori loaded exactly as in
                         compare/core.py::tori_net.
  cube_cloud.png, cube_coarse.png, cube_wsn.png -- the Figure-1 cube pipeline panels: input cloud,
                         coarse-LLL-only body (detail bands zeroed before idwt), full reconstruction.
                         PORT of _archive/gen_cubefig.py + _archive/rerender_paper.py (cube_coarse came
                         from rerender_paper.py; the old cube_splat panel is retired -- the current model
                         has no splat-grid input).
  wsn_favourites.png  -- canonical favourites (bunny/teapot/sphere/torus/cube/knurl), GT row + ours row
                         with IoU labels; render_suite.py's favourites-panel conventions at 128^3.

Model facts honoured throughout: the checkpoint is loaded ONCE via WV.load_at_res(ck, res=128,
bound=1.1) and re-latticed with net.set_res only where a figure genuinely needs another lattice
(flexsplit 64, superres 48 -- always restored to 128 afterwards); ALL meshing goes through the canonical
WV.mesh_field(raw, field_mode, bound=1.1, trunc=0.1).  CUDA only (asserted); matplotlib Agg.

Usage:  python paper_figs.py                    # all figures
        python paper_figs.py --only principle,flexsplit
"""
import os, sys, glob, argparse

import numpy as np
import torch
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)                                        # all repo-relative paths (assets/, data/, paper/)
sys.path.insert(0, os.path.join(ROOT, "tori"))        # tori package (pat.model / pat.pat), as compare/core.py

from waveshape import wavelet as WV, eval3d as E, shapes as S
from waveshape import render3d as R3
from waveshape.bunny import load_bunny
from waveshape.datasets import load_mesh_normalized

DEV = "cuda"
BOUND, TRUNC = 1.1, 0.1
RES = 128                                             # ALWAYS eval/mesh the resolution-free net at 128^3
CKPT = os.environ.get("CKPT", "assets/waveshape.pt")  # the released region-composed checkpoint
FIGS = os.path.join(ROOT, "paper", "figs")

LIGHT = np.array([0.4, -0.6, 0.72]); LIGHT /= np.linalg.norm(LIGHT)
BASE_C = np.array([0.42, 0.55, 0.66])

# ---------------------------------------------------------------------------- models (each loaded ONCE)
_net, _ck = None, None
FM = "mixed"                                          # field mode, read from the checkpoint on first load


def wsn():
    """The released PerceiverWaveNet, loaded ONCE at the canonical eval lattice (128^3)."""
    global _net, _ck, FM
    if _net is None:
        _ck = torch.load(CKPT, weights_only=False)
        FM = _ck.get("field_mode") or ("unsigned" if _ck.get("unsigned") else "signed")
        _net = WV.load_at_res(_ck, res=RES, bound=BOUND).cuda().eval()
        print(f"loaded {CKPT} | {_ck.get('model')} train-res {_ck.get('res')} -> eval {RES} | "
              f"field {FM} epoch {_ck.get('epoch')} val {_ck.get('val_sdferr')}", flush=True)
    return _net


_tori = None


def tori_net():
    """The original tori CoeffNet baseline, loaded exactly as compare/core.py::tori_net."""
    global _tori
    if _tori is None:
        from pat.model import CoeffNet                # tori/pat (sys.path has "tori")
        tck = torch.load("tori/assets/pat_torus.pt", weights_only=False)
        cfg = tck.get("config", {})
        ctor = {k: cfg[k] for k in ("d_embed", "n_layers", "n_heads", "d_ff", "supertoroid", "p_max") if k in cfg}
        net = CoeffNet(**ctor).to(DEV)
        net.load_state_dict(tck["state_dict"]); net.eval(); _tori = net
        print("loaded tori/assets/pat_torus.pt (CoeffNet)", flush=True)
    return _tori


def tori_recon(P, N):
    """PAT reconstruction as in compare/core.py::recon_tori (unscaled cloud, its own normalisation)."""
    from pat.pat import PAT
    return PAT(P, N, model=tori_net(), k=16, C=64.0, device=DEV).reconstruct(res=96, bound=BOUND, neighbors=64)


# ---------------------------------------------------------------------------- shapes & clouds
def _mn40_mesh(c):
    """First ModelNet40 test mesh of class ``c`` (falls back to the released assets/<c>.off copy)."""
    files = sorted(glob.glob(f"data/ModelNet40/{c}/test/*.off"))
    if files:
        return load_mesh_normalized(files[0], max_faces=200000)
    local = os.path.join("assets", f"{c}.off")
    if os.path.exists(local):
        return load_mesh_normalized(local, max_faces=200000)
    raise FileNotFoundError(f"no mesh for '{c}': need data/ModelNet40/{c}/test/*.off or assets/{c}.off")


def get_mesh(c):
    if c == "cube":   return S.normalize_to_unit_cube(trimesh.creation.box(extents=[1, 1, 1]))
    if c == "sphere": return S.normalize_to_unit_cube(trimesh.creation.uv_sphere(radius=0.7, count=[48, 48]))
    if c == "torus":  return S.normalize_to_unit_cube(trimesh.creation.torus(major_radius=0.5, minor_radius=0.2))
    if c == "knurl":  return S.normalize_to_unit_cube(E._knurl_mesh())
    if c == "bunny":  return S.normalize_to_unit_cube(load_bunny(normalize=True))
    if c == "teapot":
        tp = trimesh.load("assets/teapot.obj", force="mesh")
        tp.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0]))
        return S.normalize_to_unit_cube(tp)
    return _mn40_mesh(c)


def cloud(c, n=8000, seed=0):
    """(mesh, P_scaled, N, sc): cloud in the [-1,1] training frame (as gen_paper2_figs / compare/core)."""
    m = get_mesh(c)
    if c not in ("cube", "sphere", "knurl", "torus", "teapot"):
        m.fix_normals()
    sc = 1.0 / max(np.abs(m.vertices).max(), 1e-6)
    P, N = E.sample_cloud(m, n=n, noise=0.0, seed=seed)
    return m, P * sc, N.astype(np.float64), sc


# ---------------------------------------------------------------------------- recon (canonical meshing)
def net_field(P, N, n_ctx=None, ctx_P=None, ctx_N=None, center=None, half=None):
    """Raw field of the released net (distance units) on its CURRENT lattice."""
    net = wsn()
    Pt = torch.tensor(np.asarray(P)[None]).float().to(DEV)
    Nt = torch.tensor(np.asarray(N)[None]).float().to(DEV)
    kw = {}
    if n_ctx is not None:
        kw["n_ctx"] = int(n_ctx)
    if center is not None:
        kw.update(ctx_P=Pt if ctx_P is None else torch.tensor(np.asarray(ctx_P)[None]).float().to(DEV),
                  ctx_N=Nt if ctx_N is None else torch.tensor(np.asarray(ctx_N)[None]).float().to(DEV),
                  center=torch.tensor(np.asarray(center)[None, None]).float().to(DEV), half=float(half))
    with torch.no_grad():
        raw = net(Pt, Nt, **kw)[0][0, 0].float().cpu().numpy() * TRUNC
    del Pt, Nt
    torch.cuda.empty_cache()
    return raw


def recon(P, N, n_ctx=None, fm=None):
    """Released-model reconstruction through the CANONICAL mode-aware mesher."""
    raw = net_field(P, N, n_ctx=n_ctx)
    v, f = WV.mesh_field(raw, fm or FM, bound=BOUND, trunc=TRUNC)
    return v, f, raw


def direct_field(P, N, mode):
    """DIRECT single-base point field (the anchor family): mode in {'signed','unsigned','mixed'}."""
    with torch.no_grad():
        g = WV.tsdf_from_clouds(torch.tensor(np.asarray(P)[None]).float().to(DEV),
                                torch.tensor(np.asarray(N)[None]).float().to(DEV),
                                RES, TRUNC, BOUND, DEV, mode=mode)[0, 0].cpu().numpy()
    return g


# ---------------------------------------------------------------------------- drawing & metrics helpers
def draw(ax, v, f, alpha=1.0, view=(20, -55)):
    if v is not None and f is not None and len(f):
        tri = v[f]; fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        fn /= np.clip(np.linalg.norm(fn, axis=1, keepdims=True), 1e-9, None)
        sh = np.clip(np.abs(fn @ LIGHT) * 0.5 + 0.5, 0.32, 1.0)
        ax.add_collection3d(Poly3DCollection(tri, facecolors=np.clip(BASE_C[None] * sh[:, None], 0, 1),
                                             edgecolor="none", alpha=alpha))
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_axis_off(); ax.set_box_aspect((1, 1, 1)); ax.view_init(*view)


def chamfer(v, f, gt, n=30000):
    if v is None or f is None or not len(f):
        return float("nan")
    a = trimesh.Trimesh(v, f, process=False).sample(n); b, _ = trimesh.sample.sample_surface(gt, n)
    da, _ = cKDTree(b).query(a); db, _ = cKDTree(a).query(b)
    return (da.mean() + db.mean()) / 2 * 100


def ncomp(v, f):
    """Connected components (exposes tori's fragmentation on open shells)."""
    if v is None or f is None or not len(f):
        return 0
    try:
        return int(trimesh.Trimesh(v, f, process=False).body_count)
    except Exception:
        return -1


def nfaces(f):
    return 0 if f is None else len(f)


def _save(fig, name, **kw):
    os.makedirs(FIGS, exist_ok=True)
    path = os.path.join(FIGS, name)
    fig.savefig(path, **kw); plt.close(fig)
    print("wrote", path, flush=True)
    return path


# ============================================================================ principle.png
def fig_principle():
    """rows: CLOSED solid (cube) / OPEN shell (chair); cols: UDF base | SDF base | MIXED base.
    bad UDF = closed solid under unsigned (holey hollow shell); bad SDF = open shell under signed
    (over-filled blob).  UDF/SDF columns = the DIRECT single-base fields (the retired single-base
    checkpoints are gone); the mixed column = the released model (see module docstring)."""
    rows = ["cube", "chair"]
    fig = plt.figure(figsize=(6.6, 4.6))
    for i, c in enumerate(rows):
        _, P, N, _ = cloud(c, n=8000)
        vu, fu = WV.mesh_field(direct_field(P, N, "unsigned"), "unsigned", bound=BOUND, trunc=TRUNC)
        vs, fs = WV.mesh_field(direct_field(P, N, "signed"), "signed", bound=BOUND, trunc=TRUNC)
        vm, fm, _ = recon(P, N)
        cells = [("UDF", vu, fu), ("SDF", vs, fs), ("mixed", vm, fm)]
        for j, (name, v, f) in enumerate(cells):
            ax = fig.add_subplot(2, 3, i * 3 + j + 1, projection="3d")
            bad = (c == "cube" and name == "UDF") or (c == "chair" and name == "SDF")
            draw(ax, v, f, alpha=0.55 if bad else 1.0)
            nf = nfaces(f)
            if bad:
                tag = "holey hollow shell" if c == "cube" else "over-filled blob"
                ax.set_title(f"{name} ✗\n{tag}\n{nf:,} mesh faces", fontsize=9, color="#b02020")
            elif name == "mixed":
                ax.set_title(f"{name} ✓\n{nf:,} mesh faces", fontsize=9, color="#207020")
            else:
                ax.set_title(f"{name}\n{nf:,} mesh faces", fontsize=9, color="#555")
            if j == 0:
                ax.text2D(-0.12, 0.5, "closed solid" if c == "cube" else "open shell",
                          transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=10)
            print(f"principle {c}/{name}: {nf}f", flush=True)
    fig.tight_layout()
    _save(fig, "principle.png", dpi=140, bbox_inches="tight")


# ============================================================================ sdf_slices.png
def _best_slice(g, want_neg=True):
    """axis-2 slice index with the most surface/interior structure."""
    score = (g < 0).sum((0, 1)) if want_neg else (g < 0.04).sum((0, 1))
    return int(np.argmax(score))


def fig_sdf_slices():
    """FIELD plots (not meshes): (a) solid teapot -> signed SDF (released net; teapot is closed so the
    mixed field IS signed there); (b) thin chair under the SIGNED base -> the over-fill (direct signed
    field -- the signed checkpoint is retired); (c) thin chair, released net -> the unsigned band = the fix."""
    _, Pt_, Nt_, _ = cloud("teapot", n=4096)
    _, Pc_, Nc_, _ = cloud("chair", n=4096)
    g_tea = WV._smooth_grid(net_field(Pt_, Nt_), 0.5)
    g_ch_s = WV._smooth_grid(direct_field(Pc_, Nc_, "signed"), 0.5)
    g_ch_u = WV._smooth_grid(net_field(Pc_, Nc_), 0.5)

    ext = [-BOUND, BOUND, -BOUND, BOUND]
    lin = np.linspace(-BOUND, BOUND, RES)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))

    # (a) teapot signed
    k = _best_slice(g_tea); sl = g_tea[:, :, k].T
    im = axes[0].imshow(sl, extent=ext, origin="lower", cmap="RdBu", vmin=-TRUNC, vmax=TRUNC)
    axes[0].contour(lin, lin, sl, levels=[0.0], colors="k", linewidths=1.3)
    axes[0].set_title("(a) teapot — signed SDF\nnegative inside · 0 at surface · positive outside", fontsize=11)
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    # (b) chair under the signed base (over-fill)
    k = _best_slice(g_ch_s); sl = g_ch_s[:, :, k].T
    im = axes[1].imshow(sl, extent=ext, origin="lower", cmap="RdBu", vmin=-TRUNC, vmax=TRUNC)
    axes[1].contour(lin, lin, sl, levels=[0.0], colors="k", linewidths=1.3)
    axes[1].set_title("(b) thin chair — signed SDF\nlarge spurious 'inside' (blue) = the OVER-FILL", fontsize=11)
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # (c) chair, released mixed net (all-thin -> the unsigned band = the fix)
    k = _best_slice(g_ch_u, want_neg=False); sl = g_ch_u[:, :, k].T
    im = axes[2].imshow(sl, extent=ext, origin="lower", cmap="viridis", vmin=0, vmax=TRUNC)
    axes[2].contour(lin, lin, sl, levels=[0.0], colors="w", linewidths=1.2)
    axes[2].set_title("(c) thin chair — UDF (unsigned)\ndistance valley hugs the surface = the FIX", fontsize=11)
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    for a in axes:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    _save(fig, "sdf_slices.png", dpi=130)
    print(f"teapot signed [{g_tea.min():.3f},{g_tea.max():.3f}] | chair signed neg-frac "
          f"{(g_ch_s < 0).mean():.3f} | chair mixed min {g_ch_u.min():.3f}", flush=True)


# ============================================================================ usdf_gate.png
def fig_usdf_gate():
    """Per-point base selection on real clouds: BLUE = SDF (closed), GREEN = UDF (thin/open); via the
    canonical WV.point_thinness gate (same thresholds as the archived local copy)."""
    cmap = LinearSegmentedColormap.from_list("usdf", ["#185FA5", "#3B6D11"])   # blue=SDF -> green=UDF
    cats = ["cube", "bunny", "chair"]
    fig = plt.figure(figsize=(10, 3.6))
    for i, c in enumerate(cats):
        _, P, N, _ = cloud(c, n=6000)
        g = WV.point_thinness(P, N)[0].numpy()
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=g, cmap=cmap, vmin=0, vmax=1, s=4, depthshade=True)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_axis_off(); ax.set_box_aspect((1, 1, 1)); ax.view_init(18, -60)
        ax.set_title(f"{c}\n{(1 - g.mean()) * 100:.0f}% SDF / {g.mean() * 100:.0f}% UDF", fontsize=11)
        print(f"{c:7s}: UDF-frac {g.mean():.2f}", flush=True)
    fig.suptitle("Per-point base selection  —  blue = SDF (closed)   green = UDF (thin / open)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "usdf_gate.png", dpi=140, bbox_inches="tight")


# ============================================================================ flexsplit.png
def fig_flexsplit():
    """The flexible 128-token [ctx | SEP | main] budget swept across splits, released model.  Runs on the
    archived figure's 64^3 lattice (net.set_res) so face-count labels stay comparable; restored to 128."""
    cats = ["bunny", "teapot", "chair"]; SPLITS = [16, 40, 64, 88, 104]; nc = len(SPLITS)
    net = wsn()
    net.set_res(64)
    try:
        fig = plt.figure(figsize=(2.5 * nc, 2.7 * len(cats)))
        for i, c in enumerate(cats):
            _, P, N, _ = cloud(c, n=4096)
            for j, s in enumerate(SPLITS):
                v, f, _ = recon(P, N, n_ctx=s)
                ax = fig.add_subplot(len(cats), nc, i * nc + j + 1, projection="3d"); draw(ax, v, f)
                ttl = f"ctx {s} | main {127 - s}\n{nfaces(f)} faces"
                ax.set_title((f"{c}\n" if j == 0 else "") + ttl, fontsize=9)
            print(f"flexsplit {c}: done", flush=True)
        fig.tight_layout()
        _save(fig, "flexsplit.png", dpi=130)
    finally:
        net.set_res(RES)


# ============================================================================ superres_demo.png
def _crop_faces(v, f, center, half):
    """keep only faces whose centroid is inside the box (drop the box-edge skirt)."""
    if v is None or f is None:
        return None, None
    cen = v[f].mean(1); keep = np.all(np.abs(cen - center) < half, 1)
    fi = f[keep]
    if not len(fi):
        return None, None
    used = np.unique(fi); remap = {o: n for n, o in enumerate(used)}
    return v[used], np.vectorize(remap.get)(fi)


def _keep_big(v, f, frac=0.05):
    """drop small disconnected components (box-edge shards) -> the clean region surface only."""
    if v is None or f is None or not len(f):
        return v, f
    comps = trimesh.Trimesh(v, f, process=False).split(only_watertight=False)
    if len(comps) <= 1:
        return v, f
    big = max(len(c.faces) for c in comps)
    comps = [c for c in comps if len(c.faces) >= max(40, frac * big)]
    out = trimesh.util.concatenate(comps)
    return np.asarray(out.vertices), np.asarray(out.faces)


def fig_superres():
    """Context+dense super-resolution: the SAME 48^3 output lattice spent on the whole knurl vs on one
    box-normalised region (whole shape as context).  48^3 as in the archived figure, via net.set_res."""
    RES_SR = 48
    net = wsn()
    net.set_res(RES_SR)
    try:
        m = get_mesh("knurl")                          # fine diamond texture, ideal for super-res
        P, N = E.sample_cloud(m, n=8000, noise=0.0, seed=0)
        raw_full = net_field(P, N)
        vf, ff = WV.mesh_field(raw_full, FM, bound=BOUND, trunc=TRUNC)

        cam = np.array([1., 1., 1.]) / np.sqrt(3)      # box on the camera-facing surface patch
        center = P[int(np.argmax(N @ cam))].astype(np.float64); half = 0.45
        raw_box = net_field(P, N, ctx_P=P, ctx_N=N, center=center, half=half)
        vb, fb = WV.mesh_field(raw_box, FM, bound=BOUND, trunc=TRUNC)   # box-LOCAL frame (fills the view)
        vb, fb = _crop_faces(vb, fb, np.zeros(3), BOUND * 0.9)          # trim the box-edge skirt
        vb, fb = _keep_big(vb, fb)                                      # drop shards

        vc, fc = _crop_faces(vf, ff, center, half)     # SAME region from the full pass...
        if vc is not None:
            vc = (vc - center[None]) / half * BOUND    # ...renormalised to the box-local frame

        nfc, nfb = nfaces(fc), nfaces(fb)
        panels = [(f"full shape\none {RES_SR}$^3$ grid over the whole cylinder", vf, ff, False),
                  (f"this region, in the full pass\n{nfc:,} faces (coarse)", vc, fc, True),
                  (f"same region, super-resolved\n{nfb:,} faces ({nfb / max(nfc, 1):.0f}$\\times$ finer)",
                   vb, fb, True)]

        fig = plt.figure(figsize=(10.5, 3.9))
        for i, (lab, v, f, face_cam) in enumerate(panels):
            ax = fig.add_subplot(1, 3, i + 1, projection="3d")
            if v is not None and f is not None and len(f):
                tris = v[f]; fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
                area = np.linalg.norm(fn, axis=1, keepdims=True); fn = fn / np.clip(area, 1e-9, None)
                sh = np.clip(np.abs(fn @ LIGHT) * 0.5 + 0.5, 0.32, 1.0)
                ax.add_collection3d(Poly3DCollection(tris, facecolors=np.clip(BASE_C[None] * sh[:, None], 0, 1),
                                                     edgecolor="none"))
                if face_cam:                           # view face-on to the patch -> texture readable
                    mn = (fn * area).sum(0); mn = mn / (np.linalg.norm(mn) + 1e-9)
                    ax.view_init(np.degrees(np.arcsin(np.clip(mn[2], -1, 1))),
                                 np.degrees(np.arctan2(mn[1], mn[0])))
                else:
                    ax.view_init(20, -55)
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
            ax.set_axis_off(); ax.set_box_aspect((1, 1, 1)); ax.set_title(lab, fontsize=10)
        fig.suptitle(f"Context+dense super-resolution: one {RES_SR}$^3$ output lattice spent on the whole "
                     f"shape vs. on one region", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        _save(fig, "superres_demo.png", dpi=130)
        print(f"full {nfaces(ff)}f | region@full {nfc}f | region SUPER-RES {nfb}f", flush=True)
    finally:
        net.set_res(RES)


# ============================================================================ compare_models.png
def fig_compare():
    """GT | tori (CoeffNet blend) | mixed (ours), same shapes.  Tori takes the UNSCALED 4096-pt cloud
    (its own internal normalisation, as compare/core.py); ours the scaled 8000-pt cloud at 128^3."""
    cats = ["teapot", "bunny", "knurl", "chair", "guitar", "table"]
    fig = plt.figure(figsize=(9, 3.0 * len(cats)))
    for i, c in enumerate(cats):
        m, P, N, sc = cloud(c, n=8000)                 # ours: 8000 pts (thin shells need density)
        gt = trimesh.Trimesh(m.vertices * sc, m.faces, process=False)
        Pm, Nm = E.sample_cloud(m, n=4096, noise=0.0, seed=0)
        vt, ft = tori_recon(Pm, Nm)
        vx, fx, _ = recon(P, N)
        a0 = fig.add_subplot(len(cats), 3, i * 3 + 1, projection="3d"); draw(a0, gt.vertices, gt.faces)
        a1 = fig.add_subplot(len(cats), 3, i * 3 + 2, projection="3d"); draw(a1, vt, ft)
        a2 = fig.add_subplot(len(cats), 3, i * 3 + 3, projection="3d"); draw(a2, vx, fx)
        a0.set_title(f"{c}: ground truth", fontsize=10)
        a1.set_title(f"tori (CoeffNet)\nChamfer {chamfer(vt, ft, gt):.1f} | {ncomp(vt, ft)} parts",
                     fontsize=9.5, color="#a02020")
        a2.set_title(f"mixed (ours)\nChamfer {chamfer(vx, fx, gt):.1f} | {ncomp(vx, fx)} parts",
                     fontsize=9.5, color="#207020")
        print(f"compare {c}: tori {nfaces(ft)}f/{ncomp(vt, ft)}cc | mixed {nfaces(fx)}f/{ncomp(vx, fx)}cc",
              flush=True)
    fig.tight_layout()
    _save(fig, "compare_models.png", dpi=125)


# ============================================================================ cube_{cloud,coarse,wsn}.png
def fig_cube():
    """Figure-1 cube pipeline panels: (1) input cloud, (2) COARSE body -- the LLL band only (the 7 detail
    bands of c_clean zeroed before idwt: 'anchor + latents'), (3) the full reconstruction.  Panels share
    the isometric camera (render_meshes for the two meshes, matching scatter for the cloud)."""
    net = wsn()
    cube = get_mesh("cube")
    P, N = E.sample_cloud(cube, n=4096, noise=0.0, seed=0)
    os.makedirs(FIGS, exist_ok=True)

    # (1) input cloud (600-pt subsample, as the archived panel)
    sub = P[np.random.default_rng(0).choice(len(P), 600, replace=False)]
    fig = plt.figure(figsize=(4, 4)); ax = fig.add_subplot(111, projection="3d")
    ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], c="#2E6FB0", s=11, depthshade=True, edgecolors="none")
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_axis_off(); ax.set_box_aspect((1, 1, 1)); ax.view_init(20, -55)
    fig.tight_layout()
    _save(fig, "cube_cloud.png", dpi=130)

    # (2)+(3) coarse-only and full fields from ONE forward pass
    Pt = torch.tensor(P[None]).float().to(DEV); Nt = torch.tensor(N[None]).float().to(DEV)
    with torch.no_grad():
        out, c_anchor, c_clean, seg = net(Pt, Nt)
        cc = c_clean.clone(); cc[:, 1:] = 0                       # keep ONLY the coarse LLL band
        g_coarse = WV.idwt3d(cc.float(), net.haar)[0, 0].cpu().numpy() * TRUNC
        g_full = out[0, 0].float().cpu().numpy() * TRUNC
    del Pt, Nt, out, c_anchor, c_clean, seg, cc
    torch.cuda.empty_cache()

    vc, fc = WV.mesh_field(g_coarse, "signed", bound=BOUND, trunc=TRUNC)   # rounded connected body
    vw, fw = WV.mesh_field(g_full, FM, bound=BOUND, trunc=TRUNC)
    p2 = R3.render_meshes([("", vc, fc)], os.path.join(FIGS, "cube_coarse.png"), title="", size=(440, 440))
    print("wrote", p2, flush=True)
    p3 = R3.render_meshes([("", vw, fw)], os.path.join(FIGS, "cube_wsn.png"), title="", size=(440, 440))
    print("wrote", p3, flush=True)
    print(f"cube panels done | coarse {nfaces(fc)}f | full {nfaces(fw)}f", flush=True)


# ============================================================================ wsn_favourites.png
def _clean_occ(P, N):
    """Ground-truth inside-mask = sign of the clean direct field at the eval resolution (render_suite)."""
    return direct_field(P, N, FM) < 0


def _iou(gt_in, g):
    ri = g < 0
    return float((gt_in & ri).sum()) / max(float((gt_in | ri).sum()), 1)


def fig_wsn_favourites():
    """Canonical favourites panel (render_suite.py conventions): GT row + ours row, IoU + face labels,
    released checkpoint meshed at 128^3."""
    names = ["bunny", "teapot", "sphere", "torus", "cube", "knurl"]
    cols = len(names); fig = plt.figure(figsize=(2.1 * cols, 4.4))
    for j, nm in enumerate(names):
        m, P, N, sc = cloud(nm, n=4096)
        gt = trimesh.Trimesh(m.vertices * sc, m.faces, process=False)
        v, f, raw = recon(P, N)
        i = _iou(_clean_occ(P, N), WV._smooth_grid(raw, 0.5))
        ax = fig.add_subplot(2, cols, j + 1, projection="3d"); draw(ax, gt.vertices, gt.faces)
        ax.set_title(nm, fontsize=11, weight="bold")
        if j == 0:
            ax.text2D(-0.08, 0.5, "GT", transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=10)
        ax2 = fig.add_subplot(2, cols, cols + j + 1, projection="3d"); draw(ax2, v, f)
        ax2.text2D(0.5, -0.04, ("fail" if v is None else f"IoU {i:.2f} ({len(f)}f)"),
                   transform=ax2.transAxes, ha="center", fontsize=8.5, color="#207020")
        if j == 0:
            ax2.text2D(-0.08, 0.5, "ours", transform=ax2.transAxes, rotation=90, va="center", ha="center", fontsize=10)
        print(f"  {nm}: IoU {i:.3f} | {nfaces(f)}f", flush=True)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.90, bottom=0.05, wspace=0.0, hspace=0.06)
    _save(fig, "wsn_favourites.png", dpi=130)


# ============================================================================ CLI
FIG_FNS = {                                            # canonical order = run order
    "principle": fig_principle,
    "sdf_slices": fig_sdf_slices,
    "usdf_gate": fig_usdf_gate,
    "flexsplit": fig_flexsplit,
    "superres_demo": fig_superres,
    "compare_models": fig_compare,
    "cube": fig_cube,                                  # -> cube_cloud.png + cube_coarse.png + cube_wsn.png
    "wsn_favourites": fig_wsn_favourites,
}
ALIASES = {"cube_cloud": "cube", "cube_coarse": "cube", "cube_wsn": "cube",
           "superres": "superres_demo", "compare": "compare_models", "favourites": "wsn_favourites"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=None,
                    help="comma-separated subset of: " + ",".join(FIG_FNS) +
                         " (aliases: " + ",".join(ALIASES) + ")")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "GPU required (never run this on CPU)"

    if args.only:
        want, seen = [], set()
        for raw_name in args.only.split(","):
            nm = ALIASES.get(raw_name.strip(), raw_name.strip())
            if nm not in FIG_FNS:
                ap.error(f"unknown figure '{raw_name}'; choose from {list(FIG_FNS)} or aliases {list(ALIASES)}")
            if nm not in seen:
                want.append(nm); seen.add(nm)
    else:
        want = list(FIG_FNS)

    for nm in want:
        print(f"== {nm} ==", flush=True)
        FIG_FNS[nm]()
    print("ALL requested paper figures written", flush=True)


if __name__ == "__main__":
    main()
