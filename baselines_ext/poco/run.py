#!/usr/bin/env python3
"""
POCO surface-reconstruction baseline adapter.

Implements the shared scorer contract:
  python /app/run.py /in /out
  - /in/<id>.npz  : float32 arrays  points (N,3), normals (N,3), in [-1,1]^3
  - /out/<id>.ply : ONE triangle mesh in the SAME [-1,1] frame.

Reuses POCO's OFFICIAL inference path verbatim:
  - networks.Network(3, latent, 2, "FKAConv", {InterpAttentionKHeadsNet, k})
  - net.load_state_dict(checkpoint["state_dict"])
  - net.get_latent(data, with_correction=False)
  - generate.export_mesh_and_refine_vertices_region_growing_v2(...)
which is exactly what POCO's generate.py:main() calls. We only replace POCO's
file-list dataset/dataloader plumbing with a per-npz feeder that builds the same
`data` dict the network expects.
"""
import os
import sys
import glob
import yaml
import logging
import numpy as np
import torch

# POCO repo is on PYTHONPATH at /opt/POCO (set in the Dockerfile).
import networks
# Importing generate only executes its top-level imports (its main() is guarded
# by __main__), giving us the official meshing routine to reuse.
from generate import export_mesh_and_refine_vertices_region_growing_v2

logging.basicConfig(level=logging.WARNING)

MODEL_DIR = os.environ.get(
    "POCO_MODEL_DIR",
    "/weights/ABC_3k_FKAConv_InterpAttentionKHeadsNet_None",
)
GEN_RES = int(os.environ.get("POCO_GEN_RESOLUTION", "128"))
REFINE_ITER = int(os.environ.get("POCO_REFINE_ITER", "10"))
MANIFOLD_POINTS = int(os.environ.get("POCO_MANIFOLD_POINTS", "3000"))


def load_network(model_dir, device):
    cfg_path = os.path.join(model_dir, "config.yaml")
    ckpt_path = os.path.join(model_dir, "checkpoint.pth")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    latent_size = cfg["network_latent_size"]
    n_labels = cfg["network_n_labels"]
    backbone = cfg["network_backbone"]
    decoder = {"name": cfg["network_decoder"], "k": cfg["network_decoder_k"]}
    use_normals = bool(cfg.get("normals", False))

    # in_channels is always 3 in POCO: normals (3) for the normals model,
    # or a constant ones(.,3) feature for the no-normals model.
    net = networks.Network(3, latent_size, n_labels, backbone, decoder)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    net.load_state_dict(checkpoint["state_dict"])
    net.to(device)
    net.eval()
    return net, use_normals


def build_data(points, normals, use_normals, device):
    """Build the exact `data` dict the FKAConv backbone / InterpAttention head
    expect:  pos (1,3,N) and x (1,3,N).  Mirrors POCO's transform pipeline
    (Permutation [1,0] then DataLoader batch dim)."""
    pts = torch.from_numpy(points.astype(np.float32))        # (N,3)
    if use_normals:
        feat = torch.from_numpy(normals.astype(np.float32))  # normals as features
    else:
        feat = torch.ones_like(pts)                          # ConvONet-style constant feature

    pos = pts.transpose(0, 1).unsqueeze(0).contiguous().to(device)   # (1,3,N)
    x = feat.transpose(0, 1).unsqueeze(0).contiguous().to(device)    # (1,3,N)
    return {"pos": pos, "x": x}


def subsample(points, normals, n):
    N = points.shape[0]
    if n is None or n <= 0 or n == N:
        return points, normals
    if N >= n:
        idx = np.random.choice(N, n, replace=False)
    else:  # pad by sampling with replacement (rare; N~4096 >= 3000)
        idx = np.concatenate([np.arange(N),
                              np.random.choice(N, n - N, replace=True)])
    return points[idx], normals[idx]


def reconstruct_one(net, use_normals, points, normals, device):
    points, normals = subsample(points, normals, MANIFOLD_POINTS)
    data = build_data(points, normals, use_normals, device)

    with torch.no_grad():
        # Official latent encoding (same call generate.py uses for objects).
        latent = net.get_latent(data, with_correction=False)

        input_points = data["pos"][0].cpu().numpy().transpose(1, 0)  # (N,3)
        mesh = export_mesh_and_refine_vertices_region_growing_v2(
            net, latent,
            resolution=GEN_RES,
            padding=1,
            mc_value=0,
            device=device,
            input_points=input_points,
            refine_iter=REFINE_ITER,
            out_value=1,
            step=None,
        )
    return mesh


def main(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    net, use_normals = load_network(MODEL_DIR, device)
    print(f"[poco] model={MODEL_DIR} normals={use_normals} "
          f"res={GEN_RES} refine={REFINE_ITER} pts={MANIFOLD_POINTS} "
          f"device={device}", flush=True)

    files = sorted(glob.glob(os.path.join(in_dir, "*.npz")))
    print(f"[poco] {len(files)} input clouds", flush=True)

    import open3d as o3d  # provided by POCO's requirements
    for fp in files:
        sid = os.path.splitext(os.path.basename(fp))[0]
        out_fp = os.path.join(out_dir, sid + ".ply")
        try:
            d = np.load(fp)
            points = np.asarray(d["points"], dtype=np.float32)
            normals = np.asarray(d["normals"], dtype=np.float32) \
                if "normals" in d else np.zeros_like(points)

            mesh = reconstruct_one(net, use_normals, points, normals, device)
            if mesh is None:
                print(f"[poco] {sid}: empty mesh, skipping", flush=True)
                continue
            # Vertices already in the input (=[-1,1]) frame: the routine maps
            # grid coords back via bmin/step from input_points; scale=1.
            o3d.io.write_triangle_mesh(out_fp, mesh)
            print(f"[poco] {sid}: wrote {out_fp}", flush=True)
        except Exception as e:  # never crash the batch
            print(f"[poco] {sid}: FAILED ({type(e).__name__}: {e})", flush=True)
            continue


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python /app/run.py /in /out", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])