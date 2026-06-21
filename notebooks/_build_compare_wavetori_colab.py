"""Generate notebooks/compare_tori_vs_wavetori_colab.ipynb (generator -> no hand-edited JSON).

Head-to-head on the **complete ModelNet40 train split**, validated on the held-out **test**
split, of:
  * Network A -- the ORIGINAL tori network (pat.model.CoeffNet);
  * Network B -- **WaveTori**: the frozen tori CoeffNet's per-point blend PRIOR refined by a
    wavelet denoiser (pat.wavetori).

Both train on the SAME cache.  Validation renders the **paper-style 2-row figure** per test mesh
(top: 3D reconstructions; bottom: 2D SDF-slice isolines + zero-set + points), saved to files,
plus MD / IoU* / Chamfer.  RAM is freed between every stage.  Tuned for an **A100 80 GB**.

NOTE: clones REPO_URL@main and imports pat.wavetori + pat.compare -- push those before running.
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

# ----------------------------------------------------------------------------- title
cells.append(md(
"# Original **tori network** vs. **WaveTori** — complete ModelNet40 (A100 80 GB)",
"",
"Two reconstruction models, trained on the **complete ModelNet40 train split** (~9843 meshes) and",
"validated on the held-out **test** split.",
"",
"### Network A — the original tori network",
"`pat.model.CoeffNet` predicts, per point, a local (super)torus; the tori blend into a continuous SDF",
"(Feng–Gkioulekas–Crane, *Points as Tori*).  Topology-clean and continuous, but the per-point fits leave",
"a **bumpy** surface and struggle on flats.",
"",
"### Network B — WaveTori (tori prior → wavelet refiner)",
"The **same** trained CoeffNet produces the per-point blend field; a **wavelet denoiser** then refines it",
"in the multi-scale (3-D Haar) domain.  Because the tori prior is already coherent and island-free, the",
"wavelet only removes residual surface bumps — so WaveTori keeps the tori's clean topology **and** gains",
"the wavelet's smoothness (locally it matched the best of both: lower MD, higher IoU\\*).",
"",
"### Fair comparison + outputs",
"Same cache, same noise, same noisy input cloud per test mesh.  For each test mesh we save the **paper-",
"style 2-row figure** (3D recon + 2-D SDF-slice isolines/zero-set/points) and report voxel-free **IoU\\***,",
"**Minkowski filled-volume distance (MD)** and **Chamfer**.  Every stage frees RAM (`gc` + `empty_cache`).",
"",
"> Set the runtime to **GPU (A100 80 GB)**.  Everything is cached to Drive and resumes.",
))

# ----------------------------------------------------------------------------- 1. setup
cells.append(md("## 1 · Setup — Drive, repo, deps, RAM helper"))
cells.append(code(
"import os, sys, subprocess, gc",
"from google.colab import drive",
"drive.mount('/content/drive')",
"DRIVE_DIR = '/content/drive/MyDrive/points_as_supertoroids'",
"os.makedirs(DRIVE_DIR, exist_ok=True)",
"REPO_URL='https://github.com/OlegJakushkin/Points_as_supertoroids.git'; REPO_BRANCH='main'; REPO_DIR='Points_as_supertoroids'",
"subprocess.run([sys.executable,'-m','pip','install','-q','trimesh','scikit-image','scipy','pyvista','rtree'], check=False)",
"subprocess.run('apt-get install -y -qq xvfb libgl1-mesa-glx >/dev/null 2>&1', shell=True, check=False)",
"if not any(os.path.isdir(os.path.join(c,'pat')) for c in [REPO_DIR,'.','..']):",
"    subprocess.run(['git','clone','--depth','1','--branch',REPO_BRANCH,REPO_URL,REPO_DIR], check=True)",
"for cand in [REPO_DIR,'.','..']:",
"    if os.path.isdir(os.path.join(cand,'pat')): os.chdir(cand); break",
"sys.path.insert(0, os.getcwd())",
"import torch, pat",
"from pat import compare as CMP, wavetori as WT, wavelet as WV",
"assert torch.cuda.is_available(), 'Set the runtime to a GPU (A100 80 GB).'",
"def free():",
"    gc.collect()",
"    if torch.cuda.is_available(): torch.cuda.empty_cache(); torch.cuda.synchronize()",
"print('cwd', os.getcwd(), '| GPU', torch.cuda.get_device_name(0),",
"      '| %.0f GB' % (torch.cuda.get_device_properties(0).total_memory/1e9))",
))

# ----------------------------------------------------------------------------- 2. config
cells.append(md("## 2 · Config — A100 80 GB (lower the batches on a smaller GPU)"))
cells.append(code(
"# ---- dataset: the COMPLETE ModelNet40 TRAIN split ----",
"MN40_URL='http://modelnet.cs.princeton.edu/ModelNet40.zip'",
"DENSE=1536; NQUERY=1024; MAXFACES=200000; SUBSET=None   # None = every cached train mesh",
"",
"# ---- Network A: tori CoeffNet ----",
"TORI_EPOCHS=8; TORI_BATCH=64; TORI_NPOINTS=1024; K_NBR=24; D_EMBED=128; N_LAYERS=8; TORI_LR=1e-3",
"NOISE_STD=0.015          # tori training noise",
"",
"# ---- Network B: WaveTori wavelet refiner (frozen tori prior + DYNAMIC noise) ----",
"WT_RES=64                # TSDF resolution (mult of 8). 64 = A100; drop to 48/32 on a small GPU.",
"WT_TRUNC=0.1; WT_BASE=40 # refiner width (~2.1M params, ~tori size).",
"WT_EPOCHS=8; WT_BATCH=48 # meshes per step (each builds a tori-blend prior; lower on OOM).",
"WT_NPOINTS=1024          # cloud points used for the tori-blend prior (fewer = faster blend).",
"WT_LR=2e-3; LAM_WAVE=0.3; LAM_GRAD=0.05; NOISE_LO=0.005; NOISE_HI=0.03   # dynamic noise range",
"",
"# ---- eval ----",
"EVAL_NOISE=0.015; N_EVAL=12; RES_RECON=96",
"FIG_RES=72       # marching-cubes grid for the paper FIGURES (GT mesh-SDF march is CPU-bound; 72 keeps it quick)",
"FORCE_TORI=False; FORCE_WT=False",
"",
"MESH_CACHE=os.path.join(DRIVE_DIR,'mesh_cache_q%d.pt'%NQUERY)   # NQUERY-keyed (won't clash with other notebooks)",
"PROG=os.path.join(DRIVE_DIR,'mn40_q%d_progress.json'%NQUERY)",
"MODELS_DIR=os.path.join(DRIVE_DIR,'wavetori_models'); os.makedirs(MODELS_DIR, exist_ok=True)",
"EVAL_DIR=os.path.join(DRIVE_DIR,'wavetori_eval');     os.makedirs(EVAL_DIR, exist_ok=True)",
"print('config set | wavetori res', WT_RES, '| subset', SUBSET)",
))

# ----------------------------------------------------------------------------- 3. dataset
cells.append(md(
"## 3 · Dataset — complete ModelNet40 TRAIN split (download once, cached/resumed)",
"",
"Caches every train mesh as `{P,N,Q,PHI}` (the same cache both stages use).  Incremental + resumable.",
))
cells.append(code(
"import json, zipfile, urllib.request, glob",
"MN40_DIR=os.path.join(DRIVE_DIR,'ModelNet40'); MN40_ZIP=os.path.join(DRIVE_DIR,'ModelNet40.zip')",
"if not os.path.isdir(MN40_DIR):",
"    if not os.path.exists(MN40_ZIP):",
"        print('downloading ModelNet40 (~2GB)...', flush=True); urllib.request.urlretrieve(MN40_URL, MN40_ZIP)",
"    print('extracting...', flush=True)",
"    with zipfile.ZipFile(MN40_ZIP) as z: z.extractall(DRIVE_DIR)",
"train_paths=sorted(glob.glob(os.path.join(MN40_DIR,'*','train','*.off')))",
"test_paths =sorted(glob.glob(os.path.join(MN40_DIR,'*','test','*.off')))",
"assert train_paths, f'no ModelNet40 train meshes under {MN40_DIR}'",
"print(f'ModelNet40: {len(train_paths)} train / {len(test_paths)} test')",
"from pat.datasets import build_mesh_cache",
"parts=[torch.load(MESH_CACHE, weights_only=False)] if os.path.exists(MESH_CACHE) else []",
"if parts and (parts[0]['P'].shape[1]!=DENSE or parts[0]['Q'].shape[1]!=NQUERY):",
"    raise SystemExit('existing cache shape != configured DENSE/NQUERY; delete it + the progress json.')",
"cached=parts[0]['P'].shape[0] if parts else 0",
"i=json.load(open(PROG))['idx'] if (parts and os.path.exists(PROG)) else 0",
"BATCH=400",
"while i < len(train_paths):",
"    d=build_mesh_cache(train_paths[i:i+BATCH], DENSE, NQUERY, max_faces=MAXFACES, shuffle=False); i+=BATCH",
"    if d is not None:",
"        parts.append(d); cached+=d['P'].shape[0]",
"        torch.save({k: torch.cat([p[k] for p in parts],0) for k in ('P','N','Q','PHI')}, MESH_CACHE)",
"        json.dump({'idx':i,'target':len(train_paths)}, open(PROG,'w'))",
"    print(f'  cached {cached} ({min(i,len(train_paths))}/{len(train_paths)})', flush=True)",
"cache=torch.load(MESH_CACHE, weights_only=False)",
"del parts; free()",
"print('dataset ready:', cache['P'].shape[0], 'train meshes | cloud', tuple(cache['P'].shape[1:]))",
))

# ----------------------------------------------------------------------------- 4. Stage 1 tori
cells.append(md(
"## 4 · Stage 1 — train the ORIGINAL tori CoeffNet on all of ModelNet40",
"",
"`pat.compare.train_tori_cache` (paper L1 + eikonal loss, NaN/spike guard, best-by-val).  This network",
"is **both** Network A *and* the frozen prior WaveTori builds on.",
))
cells.append(code(
"def _best(h):",
"    fin=[(r['val'],r['epoch']) for r in h if r.get('val') is not None and r['val']==r['val']]",
"    return min(fin) if fin else (None,None)",
"TORI_PATH=os.path.join(MODELS_DIR,'tori_modelnet40.pt')",
"if os.path.exists(TORI_PATH) and not FORCE_TORI:",
"    ck=torch.load(TORI_PATH, weights_only=False)",
"    tori=CMP.CoeffNet(d_embed=ck['d_embed'], n_layers=ck['n_layers']).cuda(); tori.load_state_dict(ck['state'])",
"    tori_hist=ck.get('hist',[]); print('reused', TORI_PATH, '| best val', ck.get('best_val'))",
"else:",
"    tori, tori_hist = CMP.train_tori_cache(cache, k=K_NBR, epochs=TORI_EPOCHS, batch=TORI_BATCH,",
"        n_points=TORI_NPOINTS, noise_std=NOISE_STD, lr=TORI_LR, d_embed=D_EMBED, n_layers=N_LAYERS,",
"        device='cuda', subset=SUBSET, log_every=100)",
"    bv,be=_best(tori_hist)",
"    torch.save({'state':tori.state_dict(),'hist':tori_hist,'d_embed':D_EMBED,'n_layers':N_LAYERS,",
"               'best_val':bv,'best_epoch':be}, TORI_PATH)",
"    print(f'saved BEST tori (epoch {be}, val {bv})')",
"tori.eval(); free()",
"print('tori params:', sum(p.numel() for p in tori.parameters()))",
))

# ----------------------------------------------------------------------------- 5. Stage 2 wavetori
cells.append(md(
"## 5 · Stage 2 — train the WaveTori refiner (frozen tori prior, DYNAMIC noise)",
"",
"`pat.wavetori.train_wavetori`: each step builds the frozen tori's per-point blend prior of a freshly,",
"randomly-noised cloud, and the wavelet refiner learns to map it to the clean TSDF (Huber + wavelet-",
"coefficient + gradient loss; best-by-val; RAM freed periodically).",
))
cells.append(code(
"WT_PATH=os.path.join(MODELS_DIR,'wavetori_refiner.pt')",
"if os.path.exists(WT_PATH) and not FORCE_WT:",
"    ck=torch.load(WT_PATH, weights_only=False)",
"    wave=WV.WaveletDenoiser(base=ck['base']).cuda(); wave.load_state_dict(ck['state'])",
"    wt_hist=ck.get('hist',[]); WT_RES,WT_TRUNC=ck['res'],ck['trunc']",
"    print('reused', WT_PATH, '| res', WT_RES, '| best val', ck.get('best_val'))",
"else:",
"    wave, wt_hist = WT.train_wavetori(cache, tori, res=WT_RES, trunc=WT_TRUNC, epochs=WT_EPOCHS,",
"        batch=WT_BATCH, n_points=WT_NPOINTS, noise_lo=NOISE_LO, noise_hi=NOISE_HI, lr=WT_LR,",
"        lam_wave=LAM_WAVE, lam_grad=LAM_GRAD, base=WT_BASE, k=K_NBR, device='cuda', subset=SUBSET,",
"        log_every=100)",
"    bv,be=_best(wt_hist)",
"    torch.save({'state':wave.state_dict(),'hist':wt_hist,'base':WT_BASE,'res':WT_RES,'trunc':WT_TRUNC,",
"               'best_val':bv,'best_epoch':be}, WT_PATH)",
"    print(f'saved BEST wavetori refiner (epoch {be}, val {bv})')",
"wave.eval(); free()",
"print('wavetori refiner params:', wave.count_params())",
))

# ----------------------------------------------------------------------------- 6. curves
cells.append(md("## 6 · Training curves (train loss + held-out val; ★ = saved best epoch)"))
cells.append(code(
"import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt",
"def _curve(ax,h,title,c):",
"    ep=[r['epoch'] for r in h]; ax.plot(ep,[r['loss'] for r in h],'-o',c=c,label='train loss')",
"    vals=[r.get('val',float('nan')) for r in h]",
"    if any(v==v for v in vals):",
"        ax.plot(ep,vals,'--s',c='k',label='val err'); bv,be=_best(h)",
"        if be is not None: ax.scatter([be],[bv],s=150,marker='*',c='gold',edgecolor='k',zorder=5)",
"    ax.set_title(title); ax.set_xlabel('epoch'); ax.legend(fontsize=8)",
"fig,ax=plt.subplots(1,2,figsize=(11,3.8))",
"if tori_hist: _curve(ax[0],tori_hist,'A: tori (L1+eikonal)','C0')",
"if wt_hist: _curve(ax[1],wt_hist,'B: WaveTori refiner','C3')",
"fig.tight_layout(); fig.savefig(os.path.join(EVAL_DIR,'training_curves.png'),dpi=130)",
"from IPython.display import Image, display; display(Image(os.path.join(EVAL_DIR,'training_curves.png')))",
))

# ----------------------------------------------------------------------------- 7. validate on test split
cells.append(md(
"## 7 · Validate BOTH on the held-out ModelNet40 TEST split — paper-style 2-row figures",
"",
"Per test mesh: the **same** noisy cloud is reconstructed by the tori net and by WaveTori.",
"`pat.render.render_comparison` saves the **paper layout** — top row 3D reconstructions, bottom row the",
"**2-D SDF slice** (red distance isolines, blue zero-set, black input points) — to `EVAL_DIR`.  We also",
"score voxel-free MD / IoU\\* / Chamfer.  RAM is freed after every mesh.",
))
cells.append(code(
"from pat.datasets import load_mesh_normalized",
"from pat.pat import PAT",
"from pat import eval3d as E, render as PR",
"import numpy as np",
"sel=test_paths[:: max(1,len(test_paths)//N_EVAL)][:N_EVAL]",
"records=[]",
"for p in sel:",
"    try: mesh=load_mesh_normalized(p, max_faces=MAXFACES)",
"    except Exception as e: print('skip', os.path.basename(p), e); continue",
"    name=os.path.splitext(os.path.basename(p))[0]",
"    P,N=E.sample_cloud(mesh, n=DENSE, noise=EVAL_NOISE, seed=0)",
"    gt=E.mesh_gt(mesh); gv,gf=gt.mesh.vertices, gt.mesh.faces",
"    pat=PAT(P,N,model=tori,k=K_NBR,C=64.0,device='cuda')",
"    wr =WT.WaveToriReconstruction(P,N,tori,wave,res=WT_RES,trunc=WT_TRUNC,device='cuda',k=K_NBR)",
"    m_t=E.proper_metrics(gt, CMP._SdfAdapter(lambda q: pat.sdf(q,neighbors=64)), n=40000)",
"    m_w=E.proper_metrics(gt, wr, n=40000)",
"    vt,ft=pat.reconstruct(res=RES_RECON,bound=1.1,neighbors=64); vw,fw=wr.reconstruct()",
"    m_t['chamfer']=CMP.chamfer(gv,gf,vt,ft); m_w['chamfer']=CMP.chamfer(gv,gf,vw,fw)",
"    records.append({'name':name,'tori':m_t,'wavetori':m_w})",
"    print(f\"{name:20s} | MD tori {m_t['md']:.3f} wavetori {m_w['md']:.3f} | IoU* {m_t['iou']:.3f}/{m_w['iou']:.3f}\")",
"    try:",
"        PR.render_comparison(gt, {'original tori':pat, 'WaveTori':wr}, P,",
"            os.path.join(EVAL_DIR, f'fig_{name}.png'), recon_res=FIG_RES, recon_bound=1.1,",
"            suptitle=name, npoints_label=DENSE)",
"    except Exception as e: print('  fig skip', name, e)",
"    del pat, wr, vt, ft, vw, fw; free()",
"json.dump(records, open(os.path.join(EVAL_DIR,'metrics.json'),'w'), indent=1)",
"from IPython.display import Image, display; import glob as _g",
"for q in sorted(_g.glob(os.path.join(EVAL_DIR,'fig_*.png')))[:6]: display(Image(q))",
))

# ----------------------------------------------------------------------------- 8. summary
cells.append(md("## 8 · Summary — WaveTori vs original tori"))
cells.append(code(
"import numpy as np, matplotlib.pyplot as plt",
"col=lambda s,k: np.array([r[s][k] for r in records], float)",
"rows=[('MD (↓)','md'),('IoU* (↑)','iou'),('Chamfer (↓)','chamfer')]",
"print(f\"{'metric':14s} {'tori':>12s} {'WaveTori':>12s}   winner\")",
"for lab,key in rows:",
"    t,w=np.nanmean(col('tori',key)),np.nanmean(col('wavetori',key))",
"    better=(w<t) if key in ('md','chamfer') else (w>t)",
"    print(f'{lab:14s} {t:12.4f} {w:12.4f}   {\"WaveTori\" if better else \"tori\"}')",
"wins=int((col('wavetori','md')<col('tori','md')).sum())",
"print(f'\\nMD: WaveTori beats tori on {wins}/{len(records)} test meshes')",
"fig,ax=plt.subplots(1,2,figsize=(11,4)); x=np.arange(len(rows)); w=0.36",
"ax[0].bar(x-w/2,[np.nanmean(col('tori',k)) for _,k in rows],w,label='tori',color='C0')",
"ax[0].bar(x+w/2,[np.nanmean(col('wavetori',k)) for _,k in rows],w,label='WaveTori',color='C3')",
"ax[0].set_xticks(x); ax[0].set_xticklabels([l for l,_ in rows]); ax[0].legend(); ax[0].set_title('mean metrics')",
"mt,mw=col('tori','md'),col('wavetori','md'); lim=[0,max(0.01,np.nanmax([mt.max(),mw.max()]))]",
"ax[1].scatter(mt,mw,s=28,c='C2'); ax[1].plot(lim,lim,':',c='gray')",
"ax[1].set_xlabel('tori MD'); ax[1].set_ylabel('WaveTori MD'); ax[1].set_title('per-mesh MD (below line = WaveTori wins)')",
"fig.tight_layout(); fig.savefig(os.path.join(EVAL_DIR,'summary.png'),dpi=130)",
"from IPython.display import Image, display; display(Image(os.path.join(EVAL_DIR,'summary.png')))",
"free(); print('done — models + figures under', DRIVE_DIR)",
))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(__file__), "compare_tori_vs_wavetori_colab.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, "(%d cells)" % len(cells))
