"""Train the wavelet residual to ADD sub-region DETAIL the analytic composition cannot reach -- the fork that
makes the network earn its keep.  TWO changes vs train.py:

  (1) NON-ANCHOR, GT-LIKE TARGET.  The residual is trained toward the TRUE DENSE field -- the direct TSDF of a
      DENSE resampling of each mesh (default 8192 pts) -- which carries the fine relief that the SPARSE (1536-pt)
      region-composition smears away (e.g. the knurl diamonds region-growing cannot segment).  The identity-start
      ANCHOR is unchanged: the sparse region-composed field (sharp edges for free).  So the network's job is
      exactly "put back the sub-region detail the composition lost", from the raw points + global context.

  (2) GT-BASED SELECTION + VALUE TRACKING.  Checkpoints are kept by GROUND-TRUTH reconstruction F-score (meshed
      output vs the GT surface), NOT by SDF-error against the anchor.  Every epoch we print the full-model vs
      ANCHOR-ONLY gap on GT metrics for a detail-heavy val set (canonical shapes incl. the KNURL, plus held-out
      dataset shapes) -- so we SEE whether the residual adds value.  A run only "succeeds" if full > anchor.

GPU only.  Mirrors train.py's recipe (mixed base, context+region, bf16 + 8-bit Adam, region cache).
  python train_detail.py --epochs 20 --batch 8 --dense data/se_clouds_dense.pt --out waveshape_detail
"""
import argparse, json, os, time
import numpy as np, torch, trimesh
from scipy.spatial import cKDTree
from waveshape import wavelet as WV, eval3d as E
from waveshape.shapes import normalize_to_unit_cube
from waveshape.bunny import load_bunny

dev = "cuda"; bound = 1.1; RES = 64; TRUNC = 0.1; DENSE = 1536
EVAL_RES = 128; NOISE_LO, NOISE_HI = 0.0, 0.20; DRAWS = 2; LR = 2e-3; SEED = 0
CTX_MIN, CTX_MAX = 16, 104
LAM_WAVE, LAM_GRAD, LAM_SEG, LAM_SMOOTH, LAM_SIGN, LAM_CONN, LAM_CORNER, LAM_GEO = 0.4, 0.05, 0.05, 0.05, 0.30, 0.4, 0.5, 0.2
TAU = 0.03                                    # GT F-score threshold: TIGHT so sub-region detail actually counts


def light_fscore(v, f, gt, tau=TAU, n=30000, seed=0):
    """Symmetric F-score (%) between a reconstructed mesh and a GT surface point set -- ground truth, not anchor."""
    if v is None or f is None or not len(f):
        return 0.0
    a = trimesh.Trimesh(v, f, process=False).sample(n)
    da, _ = cKDTree(gt).query(a); db, _ = cKDTree(a).query(gt)
    p = float((da < tau).mean()); r = float((db < tau).mean())
    return 200.0 * p * r / (p + r + 1e-9)


def canon_meshes():
    m = {"cube": trimesh.creation.box(extents=[1, 1, 1]),
         "sphere": trimesh.creation.uv_sphere(radius=0.7, count=[48, 48]),
         "torus": trimesh.creation.torus(major_radius=0.5, minor_radius=0.2),
         "bunny": load_bunny(normalize=True), "knurl": E._knurl_mesh()}
    tp = trimesh.load("assets/teapot.obj", force="mesh")
    tp.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])); m["teapot"] = tp
    return {k: normalize_to_unit_cube(v) for k, v in m.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--cap", type=int, default=0)
    ap.add_argument("--sparse", default="data/se_clouds.pt", help="sparse INPUT/anchor cloud cache")
    ap.add_argument("--dense", default="data/se_clouds_dense.pt", help="aligned DENSE cloud cache (target source)")
    ap.add_argument("--target", choices=["dense", "anchor"], default="dense",
                    help="dense = GT-like detailed field (earn-its-keep); anchor = old circular target (baseline)")
    ap.add_argument("--out", default="waveshape_detail")
    ap.add_argument("--resume", default="")
    ap.add_argument("--res", type=int, default=RES, help="train lattice (raise to 96 if 64 is too coarse to hold the relief)")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "GPU only"
    base = "mixed"; res, eval_res, batch = a.res, EVAL_RES, a.batch
    if a.smoke:
        res, eval_res, batch, a.epochs, a.cap = 16, 32, 4, 2, 24
    out_best, out_latest = f"assets/{a.out}.pt", f"assets/{a.out}_latest.pt"
    os.makedirs("renders", exist_ok=True)
    haar = WV.haar_filters_3d(dev); t0 = time.time()

    # ---- aligned SPARSE (input/anchor) + DENSE (target) pools -----------------------------------------------
    if a.smoke:
        from train import make_solids
        P, N = make_solids(24); Pd, Nd = P, N                      # smoke: dense==sparse (aligned); only exercises plumbing
    else:
        blob = torch.load(a.sparse, weights_only=False)
        P, N = blob["P"], blob["N"]; assert P.shape[1] == DENSE
        assert os.path.exists(a.dense), f"dense cloud cache {a.dense} missing (build it in the notebook, aligned to se_clouds.pt order)"
        db = torch.load(a.dense, weights_only=False)
        Pd, Nd = db["P"], db["N"]
        assert Pd.shape[0] == P.shape[0], f"dense/sparse count mismatch {Pd.shape[0]} vs {P.shape[0]} (must be built from the SAME mesh list, same order)"
        sh = torch.randperm(P.shape[0], generator=torch.Generator().manual_seed(SEED))
        P, N, Pd, Nd = P[sh], N[sh], Pd[sh], Nd[sh]                 # SAME shuffle keeps sparse[i] <-> dense[i] aligned
        if a.cap: P, N, Pd, Nd = P[:a.cap], N[:a.cap], Pd[:a.cap], Nd[:a.cap]
    M = P.shape[0]; print(f"pool: {M} shapes | sparse {P.shape[1]}pts | dense {Pd.shape[1]}pts | target={a.target}", flush=True)
    perm = torch.randperm(M, generator=torch.Generator().manual_seed(1))
    n_val = min(16, M // 6); val_idx = perm[:n_val].tolist(); train_idx = perm[n_val:]

    # ---- per-shape region cache for the SPARSE composed anchor ----------------------------------------------
    rp_path = f"assets/region_pool_{M}.pt"
    if os.path.exists(rp_path):
        region_pool = torch.load(rp_path, weights_only=False)
        print(f"region pool: loaded {len(region_pool)} from {rp_path}", flush=True)
    else:
        region_pool = []; t_rp = time.time()
        for i in range(M):
            Pg, Ng = P[i].to(dev), N[i].to(dev)
            lab = WV.region_labels(Pg, Ng); ops = WV.region_pair_ops(Pg, Ng, lab)
            thin = WV.point_thinness(Pg[None], Ng[None])[0].cpu()
            region_pool.append((lab.astype(np.int16), ops, thin))
            if (i + 1) % 500 == 0: print(f"  region pool {i+1}/{M} | {time.time()-t_rp:.0f}s", flush=True)
        torch.save(region_pool, rp_path)
        print(f"region pool: built {M} in {time.time()-t_rp:.0f}s", flush=True)

    def composed_batch(Pb, Nb, regs, res_):
        return torch.cat([WV.tsdf_composed(Pb[b], Nb[b], regs[b][0], res_, TRUNC, bound, dev,
                                           ops=regs[b][1], thin=regs[b][2]) for b in range(Pb.shape[0])], 0) / TRUNC

    def dense_target(Pb_dense, Nb_dense, res_):
        return WV.tsdf_from_clouds(Pb_dense, Nb_dense, res_, TRUNC, bound, dev, mode=base) / TRUNC

    # ---- val set: canonical meshes (incl. KNURL) + held-out dataset shapes, all scored vs GROUND TRUTH ------
    cm = canon_meshes(); val_canon = []
    for name, mesh in cm.items():
        sc = 1.0 / max(np.abs(mesh.vertices).max(), 1e-6)
        Pp, Nn = E.sample_cloud(mesh, n=DENSE, noise=0.0, seed=0)
        Pp = (Pp * sc).astype(np.float32); gt = trimesh.Trimesh(mesh.vertices * sc, mesh.faces, process=False)
        gtpts, _ = trimesh.sample.sample_surface(gt, 40000)
        Pt = torch.tensor(Pp[None]).to(dev); Nt = torch.tensor(Nn[None]).float().to(dev)
        lab = WV.region_labels(Pt[0], Nt[0]); ops = WV.region_pair_ops(Pt[0], Nt[0], lab)
        thin = WV.point_thinness(Pt, Nt)[0].cpu()
        val_canon.append((name, Pt, Nt, [(lab.astype(np.int16), ops, thin)], np.asarray(gtpts)))

    # ---- model ----------------------------------------------------------------------------------------------
    torch.manual_seed(SEED)
    net = WV.PerceiverWaveNet(with_seg=True, res=res, trunc=TRUNC, bound=bound, field_mode=base).to(dev)
    best = -1.0; start_ep = 0
    if a.resume and os.path.exists(a.resume):
        ck = torch.load(a.resume, weights_only=False)
        if ck.get("detail"):
            net.load_state_dict({k: v for k, v in ck["state"].items() if k != "qpos"}, strict=False)
            best = ck.get("val_f", -1.0); start_ep = int(ck.get("epoch", 0))
            print(f"resumed {a.resume}: epoch {start_ep+1}, best val-F {best:.2f}", flush=True)
    bf16 = torch.cuda.is_bf16_supported()
    try:
        import bitsandbytes as bnb; opt = bnb.optim.Adam8bit(net.parameters(), lr=LR); on = "Adam8bit"
    except Exception:
        opt = torch.optim.Adam(net.parameters(), lr=LR); on = "Adam"
    from contextlib import nullcontext
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if bf16 else nullcontext
    print(f"{net.count_params():,} params | TARGET={a.target} ({'DENSE GT-like' if a.target=='dense' else 'anchor (circular)'}) "
          f"| GT-based selection (F@{TAU}) | {'bf16' if bf16 else 'fp32'}+{on}", flush=True)

    def _meta(ep_, f_):
        return {"state": net.state_dict(), "base": base, "res": res, "eval_res": eval_res, "trunc": TRUNC,
                "with_seg": True, "model": "PerceiverWaveNet", "epoch": ep_, "val_f": f_,
                "field_mode": base, "composed": True, "detail": True, "target_mode": a.target}

    def _save(meta, path):
        tmp = path + ".tmp"; torch.save(meta, tmp); os.replace(tmp, path)

    def gt_eval():
        """Mesh FULL and ANCHOR-ONLY at eval_res for every val shape; return mean GT F-score for each + the
        per-canonical breakdown.  This is the earn-its-keep signal: full must beat anchor."""
        net.eval(); net.set_res(eval_res)
        rows = []
        with torch.no_grad():
            items = [(nm, Pt, Nt, rg, gp) for (nm, Pt, Nt, rg, gp) in val_canon]
            for ii in val_idx:                                   # held-out dataset shapes: GT = their DENSE cloud
                Pt = P[ii:ii+1].to(dev); Nt = N[ii:ii+1].to(dev)
                items.append((f"data{ii}", Pt, Nt, [region_pool[ii]], Pd[ii].numpy()))
            for nm, Pt, Nt, rg, gp in items:
                out, c_anchor, _, _ = net(Pt, Nt, regions=rg)
                full = out[0, 0].float().cpu().numpy() * TRUNC
                anch = net._postprocess(WV.idwt3d(c_anchor, net.haar), Pt)[0, 0].float().cpu().numpy() * TRUNC
                vf, ff = WV.mesh_field(full, base, bound=bound, trunc=TRUNC)
                va, fa = WV.mesh_field(anch, base, bound=bound, trunc=TRUNC)
                rows.append((nm, light_fscore(vf, ff, gp), light_fscore(va, fa, gp)))
        net.set_res(res); net.train()
        return rows

    g = torch.Generator().manual_seed(2); hist = []
    for ep in range(start_ep, a.epochs):
        tr = train_idx[torch.randperm(len(train_idx), generator=g)]; run = nb = 0
        for s in range(0, len(tr), batch):
            ii = tr[s:s + batch]
            Pc = P[ii].repeat(DRAWS, 1, 1).to(dev); Nc = N[ii].repeat(DRAWS, 1, 1).to(dev)
            Pdc = Pd[ii].repeat(DRAWS, 1, 1).to(dev); Ndc = Nd[ii].repeat(DRAWS, 1, 1).to(dev)
            regs = [region_pool[gi] for gi in ii.tolist()] * DRAWS
            with torch.no_grad():
                Bc = Pc.shape[0]
                si = torch.randint(0, Pc.shape[1], (Bc,), device=dev)
                center = Pc[torch.arange(Bc, device=dev), si].unsqueeze(1)
                whole = torch.rand(Bc, 1, 1, device=dev) < 0.5
                center = torch.where(whole, torch.zeros_like(center), center)
                half = torch.where(whole, torch.full((Bc, 1, 1), bound, device=dev),
                                   torch.empty(Bc, 1, 1, device=dev).uniform_(0.3, bound))
                sc = bound / half
                Pt_dense = (Pdc - center) * sc                    # dense cloud -> box frame (same transform as anchor)
                # TARGET: dense GT-like field (detail the sparse composition can't reach), or the old anchor target
                if a.target == "dense":
                    clean = dense_target(Pt_dense, Ndc, res)
                else:
                    clean = composed_batch((Pc - center) * sc, Nc, regs, res)
                tc = WV.dwt3d(clean, haar)
                ns = torch.empty((Bc, 1, 1), device=dev).uniform_(NOISE_LO, NOISE_HI); ns[:len(ii)] = 0.0
                Pn = Pc + torch.randn(Pc.shape, device=dev) * ns
                seg_label = WV.wavelet_side_labels(tc)
            nctx = int(torch.randint(CTX_MIN, CTX_MAX + 1, (1,), generator=g).item())
            with amp():
                pred, c_anchor, c_clean, seg = net(Pn, Nc, ctx_P=Pn, ctx_N=Nc, center=center, half=half,
                                                   n_ctx=nctx, regions=regs)   # anchor = SPARSE composed (identity start)
                loss = WV.wavelet_surface_loss(pred, clean, c_clean, tc, seg, seg_label,
                                               LAM_WAVE, LAM_GRAD, LAM_SEG, LAM_SMOOTH, LAM_SIGN, LAM_CONN, LAM_GEO, LAM_CORNER)
            opt.zero_grad()
            if torch.isfinite(loss) and (nb < 5 or loss < 3 * (run / max(nb, 1))):
                loss.backward()
                for p in net.parameters():
                    if p.grad is not None: torch.nan_to_num_(p.grad, 0., 0., 0.)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                run += float(loss.detach()); nb += 1
            del clean, tc, Pn, pred, c_anchor, c_clean, seg, loss
            if nb == 1 or nb % 25 == 0:
                print(f"  ep{ep+1} {min(s+batch,len(tr))}/{len(tr)} step{nb} loss {run/max(nb,1):.4f} "
                      f"| GPU {torch.cuda.max_memory_allocated()/1e9:.1f}GB | {time.time()-t0:.0f}s", flush=True)
        torch.cuda.empty_cache()
        rows = gt_eval()
        full_m = float(np.mean([r[1] for r in rows])); anch_m = float(np.mean([r[2] for r in rows]))
        kn = next((r for r in rows if r[0] == "knurl"), None)
        improved = full_m > best
        hist.append({"epoch": ep + 1, "train": run / max(nb, 1), "val_f_full": full_m, "val_f_anchor": anch_m,
                     "delta": full_m - anch_m, "per_shape": {r[0]: [round(r[1], 1), round(r[2], 1)] for r in rows}})
        _save(_meta(ep + 1, full_m), out_latest)
        if improved:
            best = full_m; _save(_meta(ep + 1, full_m), out_best)
        knmsg = f" | KNURL full {kn[1]:.1f} vs anchor {kn[2]:.1f} (d{kn[1]-kn[2]:+.1f})" if kn else ""
        print(f"epoch {ep+1}/{a.epochs}: loss {run/max(nb,1):.4f} | GT-F full {full_m:.2f} vs ANCHOR {anch_m:.2f} "
              f"(d{full_m-anch_m:+.2f}){'  *SAVED*' if improved else ''}{knmsg} | {time.time()-t0:.0f}s", flush=True)
        json.dump(hist, open("renders/detail_train_hist.json", "w"), indent=1)
        torch.cuda.empty_cache()
    print(f"DONE {time.time()-t0:.0f}s | best GT-F(full) {best:.2f} | {out_best}", flush=True)
    print("VERDICT: the network earns its keep iff full > anchor above (esp. on the knurl).", flush=True)


if __name__ == "__main__":
    main()
