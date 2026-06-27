#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dockerised batch adapter for CAP-UDF (https://github.com/junshengzhou/CAP-UDF).

CAP-UDF is a PER-SHAPE optimisation method with NO pretrained weights: for every
input point cloud it optimises a fresh MLP (an unsigned distance field) from
scratch by the consistency-aware "pull" objective, then extracts a mesh from the
gradient field of the learned UDF. This adapter reuses the repo's OWN building
blocks (CAPUDFNetwork, the CUDA ChamferDistanceL1 loss, and the official
gradient-aware surface_extraction meshing) and replicates the official training
loop from the repo's run.py / confs/base.conf, fitting one field per cloud.

Shared I/O contract:
  INPUT : /in/<id>.npz  -> float32 arrays points (N,3), normals (N,3),
          already normalised into [-1,1]^3 (N ~ 4096).
  OUTPUT: /out/<id>.ply -> one triangle mesh in the SAME [-1,1] frame.

Usage: python /app/run.py /in /out
"""

import os
import sys
import glob
import math
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

# ---- repo imports (we run with CWD = /app/CAPUDF so these resolve) ----------
from models.fields import CAPUDFNetwork
from extensions.chamfer_dist import ChamferDistanceL1
from tools.surface_extraction import surface_extraction


# ---------------------------------------------------------------------------
# Config (env-overridable). Defaults mirror confs/base.conf, with iteration
# budgets kept at the official values. MCUBE_RES is lowered from the repo's 256
# to 128 because tools/surface_extraction.py walks the grid in a pure-Python
# triple loop; 128 keeps per-shape meshing time and the 8 GB budget sane.
# ---------------------------------------------------------------------------
def _envi(name, default):
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def _envf(name, default):
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


STEP1_MAXITER = _envi("CAPUDF_STEP1_MAXITER", 40000)   # confs/base.conf
STEP2_MAXITER = _envi("CAPUDF_STEP2_MAXITER", 60000)   # confs/base.conf
WARMUP_END = _envf("CAPUDF_WARMUP_END", 1000.0)
LR = _envf("CAPUDF_LR", 0.001)
BATCH = _envi("CAPUDF_BATCH", 5000)
BATCH2 = _envi("CAPUDF_BATCH2", 20000)
DF_FILTER = _envf("CAPUDF_DF_FILTER", 0.01)
LOW_RANGE = _envf("CAPUDF_LOW_RANGE", 1.1)
EXTRA_RATE = _envf("CAPUDF_EXTRA_POINTS_RATE", 1.0)
MCUBE_RES = _envi("CAPUDF_MCUBE_RES", 128)
EVAL_NUM_POINTS = _envi("CAPUDF_EVAL_NUM_POINTS", 1000000)

DEVICE = torch.device("cuda")


# ---------------------------------------------------------------------------
# Query-data preparation. This is models.dataset.process_data adapted to take
# an in-memory point cloud (no disk round-trip). It returns the normalised
# point cloud plus the per-query nearest-surface points exactly as the repo
# does. We ALSO return the (center, scale) used so the output mesh can be
# mapped back into the original [-1,1] frame.
# ---------------------------------------------------------------------------
def search_nearest_point(point_batch, point_gt):
    # identical logic to models/dataset.py:search_nearest_point
    num_b, num_g = point_batch.shape[0], point_gt.shape[0]
    pb = point_batch.unsqueeze(1).repeat(1, num_g, 1)
    pg = point_gt.unsqueeze(0).repeat(num_b, 1, 1)
    distances = torch.sqrt(torch.sum((pb - pg) ** 2, axis=-1) + 1e-12)
    return torch.argmin(distances, axis=1).detach().cpu().numpy()


def build_query_data(raw_pc):
    """Port of models.dataset.process_data (the .npz writer) to memory."""
    pointcloud = np.asarray(raw_pc, dtype=np.float64)

    shape_scale = np.max([
        pointcloud[:, 0].max() - pointcloud[:, 0].min(),
        pointcloud[:, 1].max() - pointcloud[:, 1].min(),
        pointcloud[:, 2].max() - pointcloud[:, 2].min(),
    ])
    shape_center = np.array([
        (pointcloud[:, 0].max() + pointcloud[:, 0].min()) / 2,
        (pointcloud[:, 1].max() + pointcloud[:, 1].min()) / 2,
        (pointcloud[:, 2].max() + pointcloud[:, 2].min()) / 2,
    ])
    pointcloud = (pointcloud - shape_center) / shape_scale

    POINT_NUM = pointcloud.shape[0] // 60
    POINT_NUM_GT = pointcloud.shape[0] // 60 * 60
    if POINT_NUM < 1:
        raise RuntimeError("too few points (<60) for CAP-UDF query sampling")
    QUERY_EACH = 1000000 // POINT_NUM_GT

    idx = np.random.choice(pointcloud.shape[0], POINT_NUM_GT, replace=False)
    pointcloud = pointcloud[idx, :]

    ptree = cKDTree(pointcloud)
    sigmas = []
    for p in np.array_split(pointcloud, 100, axis=0):
        d = ptree.query(p, 51)
        sigmas.append(d[0][:, -1])
    sigmas = np.concatenate(sigmas)

    sample, sample_near = [], []
    for i in range(QUERY_EACH):
        scale = 0.25 if 0.25 * np.sqrt(POINT_NUM_GT / 20000) < 0.25 \
            else 0.25 * np.sqrt(POINT_NUM_GT / 20000)
        tt = pointcloud + scale * np.expand_dims(sigmas, -1) * \
            np.random.normal(0.0, 1.0, size=pointcloud.shape)
        sample.append(tt)
        tt = tt.reshape(-1, POINT_NUM, 3)
        near_tmp = []
        for j in range(tt.shape[0]):
            nidx = search_nearest_point(
                torch.tensor(tt[j]).float().cuda(),
                torch.tensor(pointcloud).float().cuda())
            near_tmp.append(np.asarray(pointcloud[nidx]).reshape(-1, 3))
        near_tmp = np.asarray(near_tmp).reshape(-1, 3)
        sample_near.append(near_tmp)

    sample = np.asarray(sample).reshape(-1, 3)
    sample_near = np.asarray(sample_near).reshape(-1, 3)
    return pointcloud, sample, sample_near, POINT_NUM, shape_center, shape_scale


# ---------------------------------------------------------------------------
# In-memory analogue of models.dataset.Dataset (same batching scheme).
# ---------------------------------------------------------------------------
class MemDataset:
    def __init__(self, raw_pc):
        pc, sample, sample_near, point_num, center, scale = build_query_data(raw_pc)
        self.point_num = point_num
        self.center = center
        self.scale = scale

        self.point = torch.from_numpy(sample_near).to(DEVICE).float()
        self.sample = torch.from_numpy(sample).to(DEVICE).float()
        self.point_gt = torch.from_numpy(pc).to(DEVICE).float()
        self.sample_points_num = self.sample.shape[0] - 1

        p = sample_near
        self.object_bbox_min = np.array(
            [p[:, 0].min(), p[:, 1].min(), p[:, 2].min()]) - 0.05
        self.object_bbox_max = np.array(
            [p[:, 0].max(), p[:, 1].max(), p[:, 2].max()]) + 0.05
        self.point_new = None

    def get_train_data(self, batch_size):
        ic = np.random.choice(10, 1)
        ifi = np.random.choice(self.sample_points_num // 10, batch_size, replace=False)
        index = ifi * 10 + ic
        return self.point[index], self.sample[index], self.point_gt

    def get_train_data_step2(self, batch_size):
        ic = np.random.choice(10, 1)
        ifi = np.random.choice(self.sample_points_num // 10, batch_size, replace=False)
        index = ifi * 10 + ic
        return self.point_new[index], self.sample[index], self.point_gt

    def gen_new_data(self, tree):
        _, index = tree.query(self.sample.detach().cpu().numpy(), 1)
        self.point_new = torch.from_numpy(tree.data[index]).to(DEVICE).float()


# ---------------------------------------------------------------------------
# Meshing (mirrors run.py:extract_fields / extract_geometry but in-memory).
# ---------------------------------------------------------------------------
def extract_fields(bound_min, bound_max, resolution, query_func, grad_func):
    N = 32
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)
    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    g = np.zeros([resolution, resolution, resolution, 3], dtype=np.float32)
    for xi, xs in enumerate(X):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                xx, yy, zz = torch.meshgrid(xs, ys, zs)
                pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1),
                                 zz.reshape(-1, 1)], dim=-1).cuda()
                grad = grad_func(pts).reshape(len(xs), len(ys), len(zs), 3) \
                    .detach().cpu().numpy()
                val = query_func(pts).reshape(len(xs), len(ys), len(zs)) \
                    .detach().cpu().numpy()
                u[xi * N: xi * N + len(xs),
                  yi * N: yi * N + len(ys),
                  zi * N: zi * N + len(zs)] = val
                g[xi * N: xi * N + len(xs),
                  yi * N: yi * N + len(ys),
                  zi * N: zi * N + len(zs)] = grad
    return u, g


def reconstruct_mesh(udf_network, dataset, resolution, out_dir):
    bound_min = torch.tensor(dataset.object_bbox_min, dtype=torch.float32)
    bound_max = torch.tensor(dataset.object_bbox_max, dtype=torch.float32)
    u, g = extract_fields(
        bound_min, bound_max, resolution,
        query_func=lambda pts: udf_network.udf(pts),
        grad_func=lambda pts: udf_network.gradient(pts))
    b_max = bound_max.detach().cpu().numpy()
    b_min = bound_min.detach().cpu().numpy()
    # Official gradient-aware marching cubes (PyMCubes under the hood).
    mesh = surface_extraction(u, g, out_dir, 0, b_max, b_min, resolution)
    return mesh


# ---------------------------------------------------------------------------
# Per-shape optimisation (faithful port of run.py:Runner.train, no logging,
# no disk checkpoints). This IS the official inference path.
# ---------------------------------------------------------------------------
def gen_extra_pointcloud(udf_network, dataset, low_range):
    res = []
    gen_nums = 0
    while gen_nums < EVAL_NUM_POINTS:
        points, samples, _ = dataset.get_train_data(5000)
        offsets = samples - points
        std = torch.std(offsets)
        extra_std = std * low_range
        rands = torch.normal(0.0, extra_std, size=points.shape)
        samples = points + rands.cuda().float()
        samples.requires_grad = True
        grad = udf_network.gradient(samples).squeeze()
        udf = udf_network.udf(samples)
        gnorm = F.normalize(grad, dim=1)
        moved = samples - gnorm * udf
        keep = (udf < DF_FILTER).squeeze(1)
        moved = moved[keep]
        gen_nums += moved.shape[0]
        res.append(moved.detach().cpu().numpy())
    return np.concatenate(res)[:EVAL_NUM_POINTS]


def fit_one(raw_pc, out_dir):
    dataset = MemDataset(raw_pc)

    # network identical to confs/base.conf model.udf_network
    udf_network = CAPUDFNetwork(
        d_in=3, d_out=1, d_hidden=256, n_layers=8, skip_in=(4,),
        multires=0, bias=0.5, scale=1.0,
        geometric_init=True, weight_norm=True).to(DEVICE)
    optimizer = torch.optim.Adam(udf_network.parameters(), lr=LR)
    chamfer_l1 = ChamferDistanceL1().cuda()

    def set_lr(it):
        if it < WARMUP_END:
            lr = it / WARMUP_END
        else:
            lr = 0.5 * (math.cos((it - WARMUP_END) /
                        (STEP2_MAXITER - WARMUP_END) * math.pi) + 1)
        lr *= LR
        for pg in optimizer.param_groups:
            pg['lr'] = lr

    for it in range(STEP2_MAXITER):
        set_lr(it)
        if it < STEP1_MAXITER:
            points, samples, point_gt = dataset.get_train_data(BATCH)
        else:
            points, samples, point_gt = dataset.get_train_data_step2(BATCH2)

        samples.requires_grad = True
        grad = udf_network.gradient(samples).squeeze()
        udf = udf_network.udf(samples)
        gnorm = F.normalize(grad, dim=1)
        moved = samples - gnorm * udf
        loss = chamfer_l1(points.unsqueeze(0), moved.unsqueeze(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cur = it + 1
        # progressive field-consistency update at end of step 1
        if cur == STEP1_MAXITER:
            gen_pc = gen_extra_pointcloud(udf_network, dataset, LOW_RANGE)
            try:
                import point_cloud_utils as pcu
                k = int(EXTRA_RATE * dataset.point_gt.shape[0])
                idx = pcu.downsample_point_cloud_poisson_disk(gen_pc, num_samples=k)
                poisson_pc = gen_pc[idx]
            except Exception:
                # fallback: random subsample if pcu poisson-disk is unavailable
                k = int(EXTRA_RATE * dataset.point_gt.shape[0])
                sel = np.random.choice(gen_pc.shape[0],
                                       min(k, gen_pc.shape[0]), replace=False)
                poisson_pc = gen_pc[sel]
            dense = np.concatenate(
                (dataset.point_gt.detach().cpu().numpy(), poisson_pc))
            ptree = cKDTree(dense)
            dataset.gen_new_data(ptree)

    mesh = reconstruct_mesh(udf_network, dataset, MCUBE_RES, out_dir)

    # Map mesh vertices from process_data's normalised frame back to the
    # ORIGINAL input frame ([-1,1]^3): x_orig = x_norm * scale + center.
    mesh.vertices = mesh.vertices * dataset.scale + dataset.center[None, :]

    del udf_network, optimizer, dataset
    torch.cuda.empty_cache()
    return mesh


def load_npz(path):
    d = np.load(path)
    pts = np.asarray(d["points"], dtype=np.float32)
    # normals exist in the contract but CAP-UDF does not use them.
    return pts.reshape(-1, 3)


def main():
    if len(sys.argv) != 3:
        print("usage: python run.py /in /out", file=sys.stderr)
        sys.exit(2)
    in_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(in_dir, "*.npz")))
    if not files:
        print("no .npz inputs found in {}".format(in_dir), file=sys.stderr)

    for f in files:
        sid = os.path.splitext(os.path.basename(f))[0]
        out_ply = os.path.join(out_dir, sid + ".ply")
        try:
            print("=== fitting {} ===".format(sid), flush=True)
            pts = load_npz(f)
            mesh = fit_one(pts, out_dir)
            mesh.export(out_ply)
            print("    wrote {} (V={}, F={})".format(
                out_ply, len(mesh.vertices), len(mesh.faces)), flush=True)
        except Exception:
            print("!!! FAILED {}: skipping".format(sid), file=sys.stderr)
            traceback.print_exc()
            torch.cuda.empty_cache()
            continue


if __name__ == "__main__":
    # repo's run.py sets this so freshly-created tensors default to CUDA.
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    main()