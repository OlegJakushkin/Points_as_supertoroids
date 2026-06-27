# -*- coding: utf-8 -*-
"""
Self-contained dockerised adapter for NeuralPull (ICML 2021).
Official repo: https://github.com/mabaorui/NeuralPull-Pytorch

NeuralPull is a PER-SHAPE optimisation method: there are NO pretrained weights.
For every input cloud we fit one MLP (the repo's NPullNetwork) from scratch by
minimising the official "pull" loss, then marching-cubes the learned SDF.

This adapter REUSES the repo's own code:
  - models.fields.NPullNetwork      (the SDF MLP + .sdf()/.gradient())
  - models.dataset.search_nearest_point (nearest surface point lookup)
and replicates the repo's training loop (run.py Runner.train) and mesh
extraction (run.py Runner.extract_fields / extract_geometry) verbatim.

Contract:
  INPUT : /in/<id>.npz  with float32 arrays points (N,3) and normals (N,3),
          already normalised into the cube frame [-1,1]^3 (N ~ 4096).
  OUTPUT: /out/<id>.ply triangle mesh in the SAME [-1,1] frame.

Crucially we DO NOT apply the repo's process_data() re-centre/re-scale step
(which would move the shape into its own ~[-0.5,0.5] frame). We build the query
samples directly in the incoming [-1,1] frame and march cubes over [-1,1]^3,
so the output mesh is directly comparable to the ground-truth mesh.
"""

import os
import sys
import glob
import math
import time
import argparse
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
import trimesh
import mcubes

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Make the cloned official repo importable and reuse its modules.
# The Dockerfile clones NeuralPull-Pytorch to /app/NeuralPull.
# ---------------------------------------------------------------------------
REPO_DIR = os.environ.get("NEURALPULL_DIR", "/app/NeuralPull")
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from models.fields import NPullNetwork            # official SDF MLP
from models.dataset import search_nearest_point   # official nearest-point helper

DEVICE = torch.device("cuda")

# Official model hyper-parameters, copied verbatim from confs/npull.conf
SDF_CFG = dict(
    d_out=1,
    d_in=3,
    d_hidden=256,
    n_layers=8,
    skip_in=[4],
    multires=0,
    bias=0.5,
    scale=1.0,
    geometric_init=True,
    weight_norm=True,
)

# Official train hyper-parameters from confs/npull.conf
LEARNING_RATE = 0.001
MAXITER = 40000
WARM_UP_END = 1000
BATCH_SIZE = 5000


# ---------------------------------------------------------------------------
# Query-sample construction. This mirrors models.dataset.process_data EXACTLY
# (same sigma = distance to the 51st neighbour, same Gaussian scale schedule,
# same nearest-surface-point lookup), but operates on the incoming points in
# their original [-1,1] frame (no re-centring / re-scaling).
# ---------------------------------------------------------------------------
def build_samples(pointcloud):
    pointcloud = np.asarray(pointcloud, dtype=np.float32)

    # Repo block-size convention: keep N a multiple of 60.
    POINT_NUM = pointcloud.shape[0] // 60
    POINT_NUM_GT = POINT_NUM * 60
    if POINT_NUM < 1:
        # Too few points to follow the //60 convention; fall back to all points
        # arranged in a single block so the rest of the pipeline still works.
        POINT_NUM = pointcloud.shape[0]
        POINT_NUM_GT = pointcloud.shape[0]
    QUERY_EACH = max(1, 1000000 // POINT_NUM_GT)

    idx = np.random.choice(pointcloud.shape[0], POINT_NUM_GT, replace=False)
    pointcloud = pointcloud[idx, :]

    ptree = cKDTree(pointcloud)
    sigmas = []
    for p in np.array_split(pointcloud, 100, axis=0):
        if p.shape[0] == 0:
            continue
        k = min(51, pointcloud.shape[0])
        d = ptree.query(p, k)
        sigmas.append(d[0][:, -1])
    sigmas = np.concatenate(sigmas)

    sample = []
    sample_near = []
    pc_t = torch.tensor(pointcloud).float().cuda()
    for _ in range(QUERY_EACH):
        scale = 0.25 * np.sqrt(POINT_NUM_GT / 20000.0)
        tt = pointcloud + scale * np.expand_dims(sigmas, -1) * np.random.normal(
            0.0, 1.0, size=pointcloud.shape
        )
        sample.append(tt)
        tt_blocks = tt.reshape(-1, POINT_NUM, 3)

        near_tmp = []
        for j in range(tt_blocks.shape[0]):
            nn_idx = search_nearest_point(
                torch.tensor(tt_blocks[j]).float().cuda(), pc_t
            )
            near_tmp.append(np.asarray(pointcloud[nn_idx]).reshape(-1, 3))
        near_tmp = np.asarray(near_tmp).reshape(-1, 3)
        sample_near.append(near_tmp)

    sample = np.asarray(sample).reshape(-1, 3)
    sample_near = np.asarray(sample_near).reshape(-1, 3)
    point_gt = pointcloud.reshape(-1, 3)
    return sample, sample_near, point_gt


# ---------------------------------------------------------------------------
# Mesh extraction over a FIXED [-1,1]^3 grid (contract frame).
# This is run.py's extract_fields / extract_geometry, copied verbatim except
# that the bounds are forced to [-1,1] instead of the per-cloud bbox.
# ---------------------------------------------------------------------------
def extract_mesh(sdf_network, resolution=256, threshold=0.0):
    bound_min = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32)
    bound_max = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)

    # query_func returns -sdf, exactly as in run.py Runner.validate_mesh.
    def query_func(pts):
        return -sdf_network.sdf(pts)

    N = 32
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat(
                        [xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
                        dim=-1,
                    )
                    val = (
                        query_func(pts)
                        .reshape(len(xs), len(ys), len(zs))
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    u[
                        xi * N : xi * N + len(xs),
                        yi * N : yi * N + len(ys),
                        zi * N : zi * N + len(zs),
                    ] = val

    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()
    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return trimesh.Trimesh(vertices, triangles)


# ---------------------------------------------------------------------------
# Per-shape optimisation loop, identical to run.py Runner.train.
# ---------------------------------------------------------------------------
def update_lr(optimizer, iter_step):
    init_lr = LEARNING_RATE
    warn_up = WARM_UP_END
    max_iter = MAXITER
    if iter_step < warn_up:
        lr = iter_step / warn_up
    else:
        lr = 0.5 * (math.cos((iter_step - warn_up) / (max_iter - warn_up) * math.pi) + 1)
    lr = lr * init_lr
    for g in optimizer.param_groups:
        g["lr"] = lr


def fit_one(points_np, maxiter=MAXITER):
    sample_np, sample_near_np, _ = build_samples(points_np)

    point = torch.from_numpy(np.asarray(sample_near_np)).to(DEVICE).float()  # nearest surface pts
    sample = torch.from_numpy(np.asarray(sample_np)).to(DEVICE).float()      # query pts
    sample_points_num = sample.shape[0] - 1

    sdf_network = NPullNetwork(**SDF_CFG).to(DEVICE)
    optimizer = torch.optim.Adam(sdf_network.parameters(), lr=LEARNING_RATE)

    batch_size = min(BATCH_SIZE, sample.shape[0])
    for it in range(maxiter):
        update_lr(optimizer, it)

        # Official batched sampler (run.py DatasetNP.np_train_data).
        index_coarse = np.random.choice(10, 1)
        index_fine = np.random.choice(max(1, sample_points_num // 10), batch_size, replace=False)
        index = index_fine * 10 + index_coarse
        index = np.clip(index, 0, sample.shape[0] - 1)

        cur_points = point[index]
        cur_samples = sample[index]

        cur_samples.requires_grad = True
        grad = sdf_network.gradient(cur_samples).squeeze()
        sdf_val = sdf_network.sdf(cur_samples)
        grad_norm = F.normalize(grad, dim=1)
        moved = cur_samples - grad_norm * sdf_val

        loss = torch.linalg.norm((cur_points - moved), ord=2, dim=-1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return sdf_network


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--maxiter", type=int, default=MAXITER)
    parser.add_argument("--resolution", type=int, default=256)
    args = parser.parse_args()

    # The repo runs with CUDA tensors as the default type.
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.in_dir, "*.npz")))
    print("Found {} input clouds".format(len(files)), flush=True)

    for fp in files:
        shape_id = os.path.splitext(os.path.basename(fp))[0]
        out_fp = os.path.join(args.out_dir, shape_id + ".ply")
        if os.path.exists(out_fp):
            print("[skip] {} already done".format(shape_id), flush=True)
            continue
        t0 = time.time()
        try:
            data = np.load(fp)
            points = np.asarray(data["points"], dtype=np.float32).reshape(-1, 3)
            # normals are present in the contract but NeuralPull does not use them.
            sdf_network = fit_one(points, maxiter=args.maxiter)
            mesh = extract_mesh(sdf_network, resolution=args.resolution, threshold=0.0)
            mesh.export(out_fp)
            print(
                "[ok] {} -> {} verts ({:.1f}s)".format(
                    shape_id, len(mesh.vertices), time.time() - t0
                ),
                flush=True,
            )
        except Exception as e:  # never crash the batch on one bad shape
            print("[fail] {}: {}".format(shape_id, repr(e)), flush=True)
            continue
        finally:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()