#!/usr/bin/env python
"""
ConvONet (Convolutional Occupancy Networks, ECCV 2020) batch reconstruction adapter.

Shared I/O contract
--------------------
  python /app/run.py /in /out
  * /in/<id>.npz  -> float32 arrays `points` (N,3) and `normals` (N,3),
                     already normalised into the cube frame [-1, 1]^3.
  * /out/<id>.ply <- one triangle mesh per shape, in the SAME [-1, 1] frame.
  * A shape that fails is skipped (the batch does not crash).

What this runner does
---------------------
It reuses ConvONet's OWN inference code:
  - the `pointnet_local_pool` encoder (src.encoder.pointnet.LocalPoolPointnet),
  - the `simple_local` decoder       (src.conv_onet.models.decoder.LocalDecoder),
  - ConvolutionalOccupancyNetwork    (src.conv_onet.models),
  - Generator3D.generate_mesh        (src.conv_onet.generation),
configured exactly as in configs/pointcloud/shapenet_3plane.yaml, and loads the
OFFICIAL pretrained ShapeNet 3-plane weights (auto-downloaded, same URL the repo's
own configs/pointcloud/pretrained/shapenet_3plane.yaml uses).

We construct the model directly instead of going through src.config.get_dataset /
generate.py, because the shared contract feeds raw point-cloud .npz files rather than
the full ShapeNet dataset layout (points.npz with occupancies, watertight meshes,
metadata.yaml) that the dataset loader expects. The model/generator objects are the
repo's, unchanged.

Coordinate frames (IMPORTANT)
-----------------------------
ConvONet's ShapeNet model is trained on shapes living in the UNIT cube [-0.5, 0.5]
(padding 0.1 -> effective [-0.55, 0.55]); see src.common.normalize_coordinate /
normalize_3d_coordinate, which divide point coords by (1 + padding) and add 0.5.
The shared contract delivers points in [-1, 1]. So:
  * IN : scale input points by 0.5  ->  [-0.5, 0.5]   (model's native frame)
  * OUT: scale the generated mesh by 2.0 -> back to the [-1, 1] GT frame.
Generator3D returns vertices in box_size*(v-0.5) ~ [-0.55, 0.55]; *2 -> ~[-1.1, 1.1],
co-registered with the [-1, 1] ground truth.
"""

import os
import sys
import glob
import traceback

import numpy as np
import torch

# ConvONet repo (on PYTHONPATH via the Dockerfile) --------------------------------
from torch.utils import model_zoo
from src.encoder.pointnet import LocalPoolPointnet
from src.conv_onet.models.decoder import LocalDecoder
from src.conv_onet.models import ConvolutionalOccupancyNetwork
from src.conv_onet.generation import Generator3D

# Official pretrained ShapeNet 3-plane point-cloud model. This is the exact URL in
# configs/pointcloud/pretrained/shapenet_3plane.yaml (test.model_file).
WEIGHTS_URL = (
    "https://s3.eu-central-1.amazonaws.com/avg-projects/"
    "convolutional_occupancy_networks/models/pointcloud/shapenet_3plane.pt"
)

# Hyper-params copied verbatim from configs/pointcloud/shapenet_3plane.yaml
# (merged over configs/default.yaml).
C_DIM = 32
PADDING = 0.1
THRESHOLD = 0.2          # test.threshold in shapenet_3plane.yaml
RESOLUTION0 = 32         # generation.resolution_0 in default.yaml
UPSAMPLING_STEPS = 2     # generation.upsampling_steps in default.yaml

# Input -> model frame:  [-1, 1] -> [-0.5, 0.5]
IN_SCALE = 0.5
# Model frame -> output GT frame:  [-0.55, 0.55] -> [-1.1, 1.1] (co-registered w/ [-1,1])
OUT_SCALE = 1.0 / IN_SCALE


def build_model(device):
    """Recreate the shapenet_3plane model exactly as src.config.get_model would."""
    encoder = LocalPoolPointnet(
        dim=3,
        c_dim=C_DIM,
        hidden_dim=32,
        plane_type=["xz", "xy", "yz"],
        plane_resolution=64,
        unet=True,
        unet_kwargs=dict(depth=4, merge_mode="concat", start_filts=32),
        padding=PADDING,
    )
    decoder = LocalDecoder(
        dim=3,
        c_dim=C_DIM,
        hidden_size=32,
        sample_mode="bilinear",
        padding=PADDING,
    )
    model = ConvolutionalOccupancyNetwork(decoder, encoder, device=device)

    # Auto-download + load the official weights. The repo saves checkpoints as
    # {'model': state_dict, ...} (see src.checkpoints.CheckpointIO); mirror its
    # parse_state_dict by pulling the 'model' entry.
    state = model_zoo.load_url(WEIGHTS_URL, progress=True, map_location=device)
    if "model" in state:
        model.load_state_dict(state["model"])
    else:  # be liberal in case of a bare state_dict
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def make_generator(model, device):
    """Same Generator3D the repo builds in src.config.get_generator for
    input_type='pointcloud' (vol_bound / vol_info are None for ShapeNet objects)."""
    return Generator3D(
        model,
        device=device,
        threshold=THRESHOLD,
        resolution0=RESOLUTION0,
        upsampling_steps=UPSAMPLING_STEPS,
        sample=False,
        refinement_step=0,
        simplify_nfaces=None,
        input_type="pointcloud",
        padding=PADDING,
        vol_info=None,
        vol_bound=None,
    )


def reconstruct_one(generator, device, npz_path, out_path):
    d = np.load(npz_path)
    pts = np.asarray(d["points"], dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        raise ValueError("bad 'points' array shape: %r" % (pts.shape,))

    # [-1, 1] -> model's native [-0.5, 0.5] frame.
    pts = pts * IN_SCALE

    inputs = torch.from_numpy(pts).float().unsqueeze(0).to(device)  # (1, N, 3)

    # Official inference path: encode_inputs -> MISE marching cubes -> trimesh.
    with torch.no_grad():
        mesh = generator.generate_mesh({"inputs": inputs}, return_stats=False)

    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise RuntimeError("empty mesh (no surface crossed the threshold)")

    # Model frame -> [-1, 1] GT frame.
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) * OUT_SCALE
    mesh.export(out_path)  # trimesh writes a binary PLY by extension


def main(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[convonet] device:", device, flush=True)

    model = build_model(device)
    generator = make_generator(model, device)

    paths = sorted(glob.glob(os.path.join(in_dir, "*.npz")))
    print("[convonet] %d input shape(s)" % len(paths), flush=True)

    n_ok = 0
    for p in paths:
        sid = os.path.splitext(os.path.basename(p))[0]
        out_path = os.path.join(out_dir, sid + ".ply")
        try:
            reconstruct_one(generator, device, p, out_path)
            n_ok += 1
            print("[convonet] OK   %s" % sid, flush=True)
        except Exception as e:  # skip a bad shape, keep the batch alive
            print("[convonet] FAIL %s: %s" % (sid, e), flush=True)
            traceback.print_exc()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print("[convonet] done: %d/%d succeeded" % (n_ok, len(paths)), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python run.py <in_dir> <out_dir>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])