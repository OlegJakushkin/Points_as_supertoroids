"""Generate notebooks/train_teacher_student_colab.ipynb (generator -> no hand-edited JSON).

A copy of train_pat_colab_processed.ipynb's setup (Drive, repo, Objaverse++ mesh cache) followed by the
TEACHER -> STUDENT amortized-splat pipeline, every stage cached to Drive, presence-checked, regenerated
ONLY if its artifact is missing.  Built for a Colab **G4 (Tesla T4, 16 GB)**.

Stages:
  1  mesh cache (Objaverse++ subset)        -> DRIVE/mesh_cache.pt          (reused if present)
  2  TEACHER per-mesh optimize (sharded)    -> DRIVE/teacher/shard_*/*.pt   (per-mesh skip, resumable)
  2b teacher QA stats                        -> DRIVE/teacher/stats.png
  3  GroupNet train                          -> DRIVE/student/groupnet.pt   (skip if present)
  4  FitNet train                            -> DRIVE/student/fitnet.pt      (skip if present)
  5  amortized eval (held-out)               -> DRIVE/eval/metrics.json + recon_*.png
"""
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
"# Points as **Supertoroids** — teacher→student amortized splat optimizer (Colab **G4 / T4 16 GB**)",
"",
"Trains on the **complete ModelNet40 train split** (all ~9843 meshes across 40 categories).",
"",
"**Stage A (teacher, slow, per-mesh):** for each ModelNet40 mesh, optimize the MINIMAL set of",
"supertoroid + cut-out-box splats whose **Minkowski filled-volume distance** to the mesh is ≤ `MD_TARGET`",
"(default 0.001), **respecting holes**.  Each mesh's optimized splats + the point→splat grouping are",
"cached to Drive.",
"",
"**Stage B (student, amortized, fast):** after caching the teacher targets, train two networks — a",
"**GroupNet** that decides *how many input points to group per output supertori point*, and a separate",
"**FitNet** that best-fits each group into a single splat.  At inference they reconstruct a mesh in one",
"forward pass (no per-mesh optimization).",
"",
"Every stage's result is **cached to Drive, checked for presence, and regenerated only if missing** —",
"so an interrupted Colab session resumes where it left off.",
))

# ----------------------------------------------------------------------------- 1. setup
cells.append(md("## 1 · Setup — Google Drive, repo, deps"))
cells.append(code(
"import os, sys, subprocess",
"from google.colab import drive",
"drive.mount('/content/drive')",
"DRIVE_DIR = '/content/drive/MyDrive/points_as_supertoroids'",
"os.makedirs(DRIVE_DIR, exist_ok=True)",
"print('outputs will be saved under:', DRIVE_DIR)",
"",
"REPO_URL    = 'https://github.com/OlegJakushkin/Points_as_supertoroids.git'",
"REPO_BRANCH = 'main'   # branch holding pat/teacher.py + pat/student.py",
"REPO_DIR    = 'Points_as_supertoroids'",
"subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'trimesh', 'scikit-image', 'scipy', 'pyvista', 'rtree'], check=False)",
"subprocess.run('apt-get install -y -qq xvfb libgl1-mesa-glx >/dev/null 2>&1', shell=True, check=False)   # headless 3D rendering",
"if not any(os.path.isdir(os.path.join(c, 'pat')) for c in [REPO_DIR, '.', '..']):",
"    subprocess.run(['git', 'clone', '--depth', '1', '--branch', REPO_BRANCH, REPO_URL, REPO_DIR], check=True)",
"for cand in [REPO_DIR, '.', '..']:",
"    if os.path.isdir(os.path.join(cand, 'pat')):",
"        os.chdir(cand); break",
"sys.path.insert(0, os.getcwd())",
"import torch, pat",
"assert torch.cuda.is_available(), 'Set the Colab runtime to a GPU (T4 / G4).'",
"print('cwd', os.getcwd(), '| pat ready | GPU', torch.cuda.get_device_name(0))",
))

# ----------------------------------------------------------------------------- 2. config
cells.append(md("## 2 · Config — every knob commented"))
cells.append(code(
"# ---- dataset (Stage 1) — the COMPLETE ModelNet40 TRAIN split (all ~9843 meshes) ----",
"MN40_URL      = 'http://modelnet.cs.princeton.edu/ModelNet40.zip'   # official Princeton OFF meshes (~2GB)",
"DENSE         = 1536   # dense surface points cached per mesh (the teacher densifies these to 50k).",
"NQUERY        = 160    # GT query points per mesh (only used by the legacy trainer; teacher ignores).",
"MAXFACES      = 200000 # skip pathologically heavy meshes.",
"",
"# ---- teacher (Stage A) — THE EXPENSIVE STAGE; sharded + resumable; BATCHED across meshes ----",
"TEACHER_SUBSET = None  # None = run the teacher on the COMPLETE ModelNet40 train set (every cached mesh).",
"                       #   It is sharded + resumable, so a disconnect just continues. Set an int to test",
"                       #   on fewer first (e.g. 500). At ~3-7 s/mesh the full ~9843 set is several GPU-hours.",
"BATCH_MESHES = 'auto'  # meshes optimized IN PARALLEL on the GPU per batch — the key throughput knob.",
"                       #   'auto' probes VRAM (peak at B=1,2) and picks the largest SAFE batch, cleaning",
"                       #   up after; or set an int (raise = faster until VRAM-bound; lower on OOM).",
"MD_TARGET   = 0.001    # target Minkowski filled-volume distance = FRACTION of the cube volume that the",
"                       #   reconstruction and GT solids disagree on, vol(A xor B)/vol(cube). 0.001 = 0.1%",
"                       #   disagreement (a near-perfect fit; e.g. a clean torus fit scores ~0.0009).",
"IOU_OK      = 0.6      # QUALITY GATE: a mesh counts as a usable teacher example if IoU >= this (or MD <=",
"                       #   MD_TARGET). Detailed meshes rarely hit the tight MD bar; IoU is the scale-free",
"                       #   gate. The student trains ONLY on meshes >= this IoU (lower for more/rougher data).",
"RES         = 64       # MD grid resolution (64^3, 4 antithetic offsets). 64 is ~8x faster than 128 and",
"                       #   adequate for the prune decisions; raise to 96/128 for a finer MD (much slower).",
"M_INIT      = 40       # splats alive at the START (the warm-fit fits these).",
"M_MAX       = 128      # capacity/mesh; GROW activates dormant splats up to this where a mesh misses",
"                       #   MD_TARGET. Meshes still short at M_MAX are marked 'hard'. (Also bounds VRAM.)",
"GROW_ADD    = 16       # dormant splats activated per grow round, at the worst-fit regions.",
"MAX_GROW    = 4        # max grow rounds before the speculative prune.",
"STEPS_WARM  = 300      # batched warm-fit steps (all meshes in the batch optimized together, on-GPU).",
"STEPS_REFIT = 70       # batched refit steps per grow/prune round.",
"MIN_KEEP    = 8        # never prune below this many splats.",
"",
"# ---- student (Stage B) ----",
"K_NBR        = 24      # kNN neighborhood size fed to the GroupNet/FitNet transformer trunk.",
"GROUP_EPOCHS = 4       # GroupNet training epochs over the cached teacher shards.",
"FIT_EPOCHS   = 4       # FitNet training epochs.",
"D_EMBED      = 128     # transformer width (paper CoeffNet = 128).",
"N_LAYERS     = 8       # transformer depth.",
"N_EVAL       = 8       # held-out meshes to reconstruct + score at the end.",
"",
"FORCE_TEACHER = False  # set True to re-run the teacher even if shards exist.",
"FORCE_STUDENT = False  # set True to re-train the student even if weights exist.",
"",
"MESH_CACHE  = os.path.join(DRIVE_DIR, 'mesh_cache.pt')",
"TEACHER_DIR = os.path.join(DRIVE_DIR, 'teacher')",
"STUDENT_DIR = os.path.join(DRIVE_DIR, 'student'); os.makedirs(STUDENT_DIR, exist_ok=True)",
"EVAL_DIR    = os.path.join(DRIVE_DIR, 'eval');    os.makedirs(EVAL_DIR, exist_ok=True)",
"print('config set | teacher subset', TEACHER_SUBSET, '| MD target', MD_TARGET)",
))

# ----------------------------------------------------------------------------- 3. ModelNet40 cache
cells.append(md(
"## 3 · Dataset — the COMPLETE ModelNet40 TRAIN split (download once, cache reused/resumed)",
"",
"Downloads ModelNet40 (40 categories of OFF meshes), then caches **every mesh in the train split** as",
"`{P,N,Q,PHI}` (dense surface cloud + normals + GT queries).  The cache is built in batches and saved",
"incrementally, so a Colab disconnect just resumes.  The held-out **test** split is used for the eval cell.",
))
cells.append(code(
"import json, zipfile, urllib.request, glob",
"PROG = os.path.join(DRIVE_DIR, 'mn40_progress.json')",
"MN40_DIR = os.path.join(DRIVE_DIR, 'ModelNet40'); MN40_ZIP = os.path.join(DRIVE_DIR, 'ModelNet40.zip')",
"if not os.path.isdir(MN40_DIR):                   # download + extract once (~2GB)",
"    if not os.path.exists(MN40_ZIP):",
"        print('downloading ModelNet40 (~2GB, one-time)...', flush=True)",
"        urllib.request.urlretrieve(MN40_URL, MN40_ZIP)",
"    print('extracting ModelNet40...', flush=True)",
"    with zipfile.ZipFile(MN40_ZIP) as z: z.extractall(DRIVE_DIR)",
"train_paths = sorted(glob.glob(os.path.join(MN40_DIR, '*', 'train', '*.off')))",
"test_paths  = sorted(glob.glob(os.path.join(MN40_DIR, '*', 'test',  '*.off')))",
"assert train_paths, f'no ModelNet40 train .off meshes under {MN40_DIR} (download/extract failed?)'",
"ncat = len({os.path.basename(os.path.dirname(os.path.dirname(p))) for p in train_paths})",
"print(f'ModelNet40: {len(train_paths)} train / {len(test_paths)} test meshes, {ncat} categories')",
"# build the cache from ALL train meshes (batched, resumable)",
"from pat.datasets import build_mesh_cache",
"parts = [torch.load(MESH_CACHE, weights_only=False)] if os.path.exists(MESH_CACHE) else []",
"cached = parts[0]['P'].shape[0] if parts else 0",
"i = json.load(open(PROG))['idx'] if (parts and os.path.exists(PROG)) else 0",
"BATCH = 400",
"while i < len(train_paths):",
"    d = build_mesh_cache(train_paths[i:i+BATCH], DENSE, NQUERY, max_faces=MAXFACES, shuffle=False)",
"    i += BATCH",
"    if d is not None:",
"        parts.append(d); cached += d['P'].shape[0]",
"        torch.save({k: torch.cat([p[k] for p in parts], 0) for k in ('P','N','Q','PHI')}, MESH_CACHE)",
"        json.dump({'idx': i, 'target': len(train_paths)}, open(PROG, 'w'))",
"    print(f'  cached {cached} meshes ({min(i,len(train_paths))}/{len(train_paths)} paths)', flush=True)",
"cache = torch.load(MESH_CACHE, weights_only=False)",
"print('dataset ready:', cache['P'].shape[0], 'ModelNet40 train meshes')",
))

# ----------------------------------------------------------------------------- 4. teacher
cells.append(md(
"## 4 · Stage A — TEACHER: minimal-splat optimize, **BATCHED across meshes** (sharded, resumable)",
"",
"`fit_and_cache_batch` optimizes **`BATCH_MESHES` meshes in parallel on the GPU**: it over-provisions",
"`M_INIT` splats/mesh, batched warm-fits, then runs a **speculative prune** — for a descending keep-",
"schedule it keeps the top-ownership splats per mesh, refits ALL meshes together, scores MD per mesh, and",
"remembers each mesh's smallest field that still met `MD_TARGET` (independently per mesh).  Ground-truth",
"occupancy is built **hole-respecting from the cached cloud P+N** (no mesh re-download).  Each mesh writes",
"one shard atomically; **current-version** shards are skipped (resumes after a disconnect), but shards from",
"an OLDER teacher pipeline are **automatically regenerated** -- so just re-running this cell upgrades a",
"stale cache (e.g. shards built before the k-NN-GT / MD-fraction fixes) without deleting anything.",
))
cells.append(code(
"from pat import teacher_batch as TB",
"import glob, time, torch",
"from tqdm.auto import tqdm",
"Pall, Nall = cache['P'], cache['N']",
"TOT = Pall.shape[0] if TEACHER_SUBSET is None else min(TEACHER_SUBSET, Pall.shape[0])   # None = all train",
"os.makedirs(TEACHER_DIR, exist_ok=True)",
"if BATCH_MESHES == 'auto':                       # probe VRAM -> largest safe batch so the FIT fills VRAM",
"    BATCH_MESHES = TB.auto_batch_size(Pall[0].numpy(), Nall[0].numpy(), m_max=M_MAX, res=RES, device='cuda')",
"    print('auto-detected safe BATCH_MESHES =', BATCH_MESHES)",
"have = len(glob.glob(os.path.join(TEACHER_DIR, 'shard_*', 'mesh_*.pt')))",
"stale = TB._T.count_stale_shards(TEACHER_DIR, list(range(TOT)))   # missing OR old-pipeline -> will regen",
"print(f'teacher: {have} shards on disk; {stale}/{TOT} missing-or-stale -> will (re)generate '",
"      f'(pipeline v{TB._T.TEACHER_VERSION}); batch {BATCH_MESHES}')",
"if stale and have: print('  NOTE: re-generating shards from an older teacher pipeline (k-NN GT + MD-fraction fixes).')",
"ok = hard = ran = 0; t0 = time.time()",
"for s in tqdm(range(0, TOT, BATCH_MESHES), desc='teacher (batched)'):",
"    chunk = list(range(s, min(s + BATCH_MESHES, TOT)))",
"    rows = TB.fit_and_cache_batch(",
"        [Pall[g].numpy() for g in chunk], [Nall[g].numpy() for g in chunk], chunk, TEACHER_DIR,",
"        force=FORCE_TEACHER, m_init=M_INIT, m_max=M_MAX, grow_add=GROW_ADD, max_grow=MAX_GROW,",
"        md_target=MD_TARGET, iou_ok=IOU_OK, res=RES, steps_warm=STEPS_WARM, steps_refit=STEPS_REFIT,",
"        min_keep=MIN_KEEP, device='cuda')",
"    for g, status, *rest in rows:",
"        ran += status != 'cached'; ok += status == 'ok'; hard += status == 'hard'",
"    if torch.cuda.is_available(): torch.cuda.empty_cache()   # clean up between batches",
"    if ran:",
"        print(f'  {min(s+BATCH_MESHES,TOT)}/{TOT} | ran {ran} | ok {ok} hard {hard} | {(time.time()-t0)/max(ran,1):.1f}s/mesh', flush=True)",
"n_shards = len(glob.glob(os.path.join(TEACHER_DIR, 'shard_*', 'mesh_*.pt')))",
"print(f'teacher cache now holds {n_shards} meshes ({ok} ok / {hard} hard this session)')",
))

cells.append(md("## 4b · Teacher QA — splat-count & MD/IoU distribution, USABLE (IoU≥IOU_OK) fraction"))
cells.append(code(
"import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt",
"from pat import student as ST",
"rows=[{'M':a['M'],'md':a['md'],'iou':a['iou'],'status':a['status']} for a in ST.iter_shards(TEACHER_DIR)]",
"Ms=[r['M'] for r in rows]; mds=[r['md'] for r in rows]; ious=[r['iou'] for r in rows]",
"usable=int(np.sum([i>=IOU_OK for i in ious])); frac=usable/max(len(rows),1)",
"json.dump(rows, open(os.path.join(TEACHER_DIR,'manifest.json'),'w'))",
"fig,ax=plt.subplots(1,3,figsize=(13,3.4))",
"ax[0].hist(Ms,bins=30,color='C0'); ax[0].set_title('# splats per mesh'); ax[0].set_xlabel('M')",
"ax[1].hist(mds,bins=30,color='C2'); ax[1].axvline(MD_TARGET,ls=':',c='r'); ax[1].set_title('MD = vol(A xor B)/vol(cube)'); ax[1].set_xlabel('MD')",
"ax[2].hist(ious,bins=30,color='C3'); ax[2].axvline(IOU_OK,ls=':',c='b'); ax[2].set_title('IoU'); ax[2].set_xlabel('IoU')",
"fig.suptitle(f'teacher QA — {len(rows)} meshes | median M={int(np.median(Ms)) if Ms else 0} | usable IoU>={IOU_OK}: {usable} ({frac:.0%})')",
"fig.tight_layout(); fig.savefig(os.path.join(TEACHER_DIR,'stats.png'),dpi=130)",
"from IPython.display import Image, display; display(Image(os.path.join(TEACHER_DIR,'stats.png')))",
"print(f'median splats/mesh {int(np.median(Ms)) if Ms else 0} | median IoU {np.median(ious):.2f} | USABLE (IoU>={IOU_OK}): {usable}/{len(rows)} ({frac:.0%}) -> these train the student')",
"if usable<32: print('WARNING: very few usable meshes. Lower IOU_OK, or run the DIAGNOSTIC cell to see if it is the GT (non-watertight) or the fit.')",
))

cells.append(md(
"## 4c · DIAGNOSTIC — is a low IoU the GT or the fit?",
"",
"For a few meshes: **surface error** = mean `|blend SDF|` on the input cloud (≈0 means the reconstruction",
"surface sits ON the points → a GOOD fit; if this is small but IoU is low, the *ground-truth occupancy* is",
"unreliable — typically a non-watertight mesh whose inside/outside is ill-defined from points+normals).  A",
"large surface error means the *fit* itself didn't converge (raise STEPS_WARM / M_MAX).  Renders show GT vs",
"reconstruction side by side.",
))
cells.append(code(
"from pat import teacher as TCH, render3d as R3",
"import numpy as np, torch, glob as _g",
"paths=sorted(_g.glob(os.path.join(TEACHER_DIR,'shard_*','mesh_*.pt')))",
"pick=paths[:: max(1,len(paths)//6)][:6]",
"print('%-8s %6s %8s %8s %8s %9s'%('gid','M','surf_err','MD','IoU','GT_vol%'))",
"for p in pick:",
"    a=TCH.load_teacher(p); sp=a['splat'].cuda(); P=a['P'].float().numpy(); N=a['N'].float().numpy()",
"    surf=float(sp.sdf_torch(torch.as_tensor(P,dtype=torch.float32,device='cuda')).abs().mean())",
"    cs=TCH.CloudShape(P,N); occ=TCH.gt_occupancy(cs,res=RES); volpct=100*float(occ.mean())",
"    md,iou=TCH.md_filled_volume(sp,occ,res=RES,device='cuda',return_iou=True)   # fresh (new MD scale)",
"    print('%-8d %6d %8.4f %8.4f %8.3f %9.1f'%(a['gid'],a['M'],surf,md,iou,volpct))",
"    try: R3.render_comparison(cs, sp, os.path.join(TEACHER_DIR,f'diag_{a[\"gid\"]:06d}.png'), title=f'gid {a[\"gid\"]} IoU {a[\"iou\"]:.2f}')",
"    except Exception as e: print('  render skip', e)",
"from IPython.display import Image, display",
"for p in sorted(_g.glob(os.path.join(TEACHER_DIR,'diag_*.png')))[:4]: display(Image(p))",
"print('READ: small surf_err + low IoU => GT/occupancy problem (non-watertight). large surf_err => fit problem.')",
))

# ----------------------------------------------------------------------------- 5. GroupNet
cells.append(md(
"## 5 · Stage B1 — GroupNet: learn *how many points to group per supertori point*",
"",
"Per-point (position-aware) seed-ness + metric embedding on the cached teacher `owner` labels, trained ONLY",
"on **usable meshes (IoU≥IOU_OK)** — bad teacher examples are gated out, so they can't poison the student.",
"At inference, NMS over the seeds gives K groups and every point joins its nearest seed (spatially coherent).",
))
cells.append(code(
"os.makedirs(STUDENT_DIR, exist_ok=True)   # ensure the output dir exists (independent of cell run order)",
"GN_PATH=os.path.join(STUDENT_DIR,'groupnet.pt')",
"if os.path.exists(GN_PATH) and not FORCE_STUDENT:",
"    ck=torch.load(GN_PATH, weights_only=False)",
"    gnet=ST.GroupNet(d_embed=D_EMBED, n_layers=N_LAYERS, d_g=ck.get('d_g',32)).cuda()",
"    gnet.load_state_dict(ck['state']); print('reused groupnet.pt')",
"else:",
"    gnet, gh = ST.train_groupnet(TEACHER_DIR, epochs=GROUP_EPOCHS, k=K_NBR, device='cuda', iou_min=IOU_OK,",
"        net=ST.GroupNet(d_embed=D_EMBED, n_layers=N_LAYERS, d_g=32).cuda())",
"    torch.save({'state':gnet.state_dict(),'d_g':32}, GN_PATH)",
"    json.dump(gh, open(os.path.join(STUDENT_DIR,'groupnet_log.json'),'w'), indent=1)",
"    if gh: print('GroupNet trained; loss', round(gh[0]['loss'],3), '->', round(gh[-1]['loss'],3))",
))

# ----------------------------------------------------------------------------- 6. FitNet
cells.append(md(
"## 6 · Stage B2 — FitNet: best-fit each group into a single supertoroid splat",
"",
"A separate permutation-invariant set encoder maps one point-group → one splat's parameters, supervised",
"by the teacher's per-splat params via a **geometry-first** loss on the induced single-splat SDF.",
))
cells.append(code(
"os.makedirs(STUDENT_DIR, exist_ok=True)",
"FN_PATH=os.path.join(STUDENT_DIR,'fitnet.pt')",
"if os.path.exists(FN_PATH) and not FORCE_STUDENT:",
"    ck=torch.load(FN_PATH, weights_only=False)",
"    fnet=ST.FitNet(d_embed=D_EMBED, n_layers=max(6,N_LAYERS-2)).cuda(); fnet.load_state_dict(ck['state'])",
"    print('reused fitnet.pt')",
"else:",
"    fnet, fh = ST.train_fitnet(TEACHER_DIR, epochs=FIT_EPOCHS, device='cuda', iou_min=IOU_OK,",
"        net=ST.FitNet(d_embed=D_EMBED, n_layers=max(6,N_LAYERS-2)).cuda())",
"    torch.save({'state':fnet.state_dict()}, FN_PATH)",
"    json.dump(fh, open(os.path.join(STUDENT_DIR,'fitnet_log.json'),'w'), indent=1)",
"    if fh: print('FitNet trained; loss', round(fh[0]['loss'],3), '->', round(fh[-1]['loss'],3))",
))

# ----------------------------------------------------------------------------- 7. eval (ModelNet40 test)
cells.append(md(
"## 7 · Amortized eval on the held-out ModelNet40 TEST split (voxel-free)",
"",
"GroupNet+FitNet reconstruct **test meshes the teacher never saw** in one forward pass (no per-mesh",
"optimization).  Quality is the **voxel-free** Monte-Carlo IoU* — our continuous `splat.sdf<0` vs the exact",
"mesh occupancy (`MeshShape`), no occupancy grid on either side — so it is a true generalization metric.",
))
cells.append(code(
"from pat import eval3d as E, render3d as R3",
"from pat.datasets import load_mesh_normalized",
"import numpy as np",
"os.makedirs(EVAL_DIR, exist_ok=True)",
"sel = test_paths[:: max(1, len(test_paths)//N_EVAL)][:N_EVAL]   # spread across categories",
"metrics=[]",
"for p in sel:",
"    try: mesh = load_mesh_normalized(p, max_faces=MAXFACES)",
"    except Exception as e: print('skip', os.path.basename(p), e); continue",
"    P,N = E.sample_cloud(mesh, n=DENSE, seed=0)",
"    sp,K = ST.reconstruct_amortized(P, N, gnet, fnet, k=K_NBR, device='cuda')",
"    if sp is None: continue",
"    gt = E.mesh_gt(mesh); pm = E.proper_metrics(gt, sp.cuda(), n=40000)",
"    name = os.path.splitext(os.path.basename(p))[0]",
"    metrics.append({'name':name,'K':int(K),'iou':pm['iou'],'vol_err':pm['vol_err']})",
"    try: E.gallery_render(name, gt.mesh, sp.cuda(), os.path.join(EVAL_DIR,f'recon_{name}.png'), res=128, iou=pm['iou'])",
"    except Exception as e: print('render skip', name, e)",
"json.dump(metrics, open(os.path.join(EVAL_DIR,'metrics.json'),'w'), indent=1)",
"if metrics:",
"    print('amortized on ModelNet40 TEST: mean IoU* %.3f | mean vol_err %.3f | mean K %.1f'%(",
"        np.mean([m['iou'] for m in metrics]), np.mean([m['vol_err'] for m in metrics]), np.mean([m['K'] for m in metrics])))",
"    from IPython.display import Image, display",
"    import glob as _g",
"    for q in sorted(_g.glob(os.path.join(EVAL_DIR,'recon_*.png')))[:4]: display(Image(q))",
"print('done — teacher shards, student weights, and eval all cached under', DRIVE_DIR)",
))

# ----------------------------------------------------------------------------- 8. canonical gallery
cells.append(md(
"## 8 · Canonical test-shape gallery — model performance on named shapes",
"",
"Fits the supertoroid-splat **teacher** (per-mesh optimizer; the student amortizes it) to five canonical",
"shapes — **teapot, bunny, hole+bolts plate, cube+cylinder (noisy sampling), diamond-knurl (sharp corners",
"& texture)** — and renders the **exact ground-truth mesh vs reconstruction** (the GT mesh is drawn",
"directly, NOT a voxel grid).  `IoU*` is the **voxel-free** Monte-Carlo IoU: continuous on BOTH sides —",
"an exact analytic SDF / winding-number reference vs our `splat.sdf < 0` — so it has no voxelization bias.",
))
cells.append(code(
"from pat import eval3d as E, teacher_batch as TB",
"from IPython.display import Image, display",
"GAL_DIR=os.path.join(DRIVE_DIR,'gallery'); os.makedirs(GAL_DIR, exist_ok=True)",
"shapes=E.canonical_shapes(bunny=True)",
"clouds=[E.sample_cloud(m, n=1536, noise=getattr(gt,'noise',0.0), seed=0) for _,gt,m in shapes]",
"fits=TB.fit_teacher_batch([c[0] for c in clouds],[c[1] for c in clouds], m_init=M_INIT, m_max=M_MAX,",
"    grow_add=GROW_ADD, max_grow=MAX_GROW, md_target=MD_TARGET, iou_ok=IOU_OK, res=RES,",
"    steps_warm=STEPS_WARM, steps_refit=STEPS_REFIT, min_keep=MIN_KEEP, device='cuda')",
"for (name,gt,mesh),(sp,md,iouv,status) in zip(shapes,fits):",
"    sp=sp.cuda(); iou=E.proper_metrics(gt,sp,n=40000)['iou']",
"    p=os.path.join(GAL_DIR, name.replace('+','_').replace(' ','_')+'.png')",
"    E.gallery_render(name, mesh, sp, p, res=128, iou=iou); print(f'{name}: IoU* {iou:.3f} ({sp.M} splats)'); display(Image(p))",
"print('gallery saved to', GAL_DIR)",
))

# ----------------------------------------------------------------------------- 9. quality vs properties
cells.append(md(
"## 9 · Quality vs mesh properties — voxel-free eval on a validation set",
"",
"Fits the teacher to a validation set (canonical shapes + procedural primitives spanning watertight/open,",
"genus, thinness, face-count) and scores the **voxel-free** IoU* / volume error against the exact library",
"reference (analytic SDF or winding-number `MeshShape` — *not* the training voxel grid).  The matrix",
"scatters each quality metric against each mesh property, revealing what geometry the model struggles on.",
))
cells.append(code(
"import numpy as np, json as _j",
"vshapes=E.val_shapes(bunny=True)",
"records=[]",
"for s in range(0, len(vshapes), BATCH_MESHES):",
"    chunk=vshapes[s:s+BATCH_MESHES]",
"    cl=[E.sample_cloud(m, n=1536, noise=getattr(gt,'noise',0.0), seed=0) for _,gt,m in chunk]",
"    fc=TB.fit_teacher_batch([c[0] for c in cl],[c[1] for c in cl], m_init=M_INIT, m_max=M_MAX,",
"        grow_add=GROW_ADD, max_grow=MAX_GROW, md_target=MD_TARGET, iou_ok=IOU_OK, res=RES,",
"        steps_warm=STEPS_WARM, steps_refit=STEPS_REFIT, min_keep=MIN_KEEP, device='cuda')",
"    for (name,gt,mesh),(sp,md,iouv,status) in zip(chunk,fc):",
"        sp=sp.cuda(); pm=E.proper_metrics(gt,sp,n=40000); mp=E.mesh_properties(mesh)",
"        records.append(dict(name=name, M=int(sp.M), **pm, **mp))",
"    if torch.cuda.is_available(): torch.cuda.empty_cache()",
"_j.dump(records, open(os.path.join(EVAL_DIR,'val_metrics.json'),'w'), indent=1)",
"E.plot_metrics_matrix(records, os.path.join(EVAL_DIR,'quality_vs_properties.png'))",
"from IPython.display import Image, display; display(Image(os.path.join(EVAL_DIR,'quality_vs_properties.png')))",
"print('val set: %d shapes | mean IoU* %.3f | mean vol_err %.3f'%(len(records), np.mean([r['iou'] for r in records]), np.mean([r['vol_err'] for r in records])))",
))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(__file__), "train_teacher_student_colab.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, "(%d cells)" % len(cells))
