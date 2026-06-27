#!/usr/bin/env python3
# NKSR batch adapter implementing the shared I/O contract.
#
#   python /app/run.py /in /out
#
# For each /in/<id>.npz (float32 arrays `points` (N,3) and `normals` (N,3),
# already normalised into the cube frame [-1,1]^3, N ~ 4096) this writes one
# triangle mesh /out/<id>.ply in the SAME [-1,1] frame.
#
# It uses NKSR's OFFICIAL inference path: nksr.Reconstructor(...).reconstruct(xyz, normal)
# followed by field.extract_dual_mesh(...), exactly as documented in the repo's
# README / NKSR-USAGE.md / examples/recons_simple.py. NKSR is a feed-forward
# (generalising) reconstructor with a global linear solve per shape -- there is
# no per-shape gradient optimisation to run, so we simply call reconstruct() once
# per cloud.
#
# Weights: the default config 'ks' (kitchen-sink, CC-BY-SA 4.0) is auto-downloaded
# by the package from HuggingFace heiwang1997/nksr-checkpoints (ks.pth) via
# torch.hub on first use. It is cached under TORCH_HOME. The model is trained on a
# MIX of objects + scenes (not ModelNet), so ModelNet evaluation here is zero-shot /
# out-of-distribution. detail_level keeps it scale-robust for an object in [-1,1].

import os
import sys
import glob
import traceback

import numpy as np
import torch


def write_ply_triangle(path, verts, faces):
    """Write a minimal binary-little-endian triangle-mesh PLY (verts float32, faces int32)."""
    verts = np.ascontiguousarray(verts, dtype="<f4")
    faces = np.ascontiguousarray(faces, dtype="<i4")
    n_v = verts.shape[0]
    n_f = faces.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n_v}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {n_f}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    with open(path, "wb") as f:
        f.write(header)
        f.write(verts.tobytes())
        if n_f:
            # Each face row: uint8 count (=3) + 3 int32 indices.
            counts = np.full((n_f, 1), 3, dtype=np.uint8)
            face_bytes = np.concatenate(
                [counts.view(np.uint8), faces.view(np.uint8).reshape(n_f, 12)], axis=1
            )
            f.write(face_bytes.tobytes())


def main(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    if not torch.cuda.is_available():
        print("[FATAL] CUDA is not available; NKSR needs a GPU for the official path.", flush=True)
        sys.exit(1)

    import nksr  # imported after CUDA check so the _C extension loads cleanly

    device = torch.device("cuda:0")
    # 'ks' = kitchen-sink default (mixed object+scene training, CC-BY-SA 4.0).
    reconstructor = nksr.Reconstructor(device, config="ks")

    files = sorted(glob.glob(os.path.join(in_dir, "*.npz")))
    print(f"[INFO] Found {len(files)} input clouds in {in_dir}", flush=True)

    for fp in files:
        sid = os.path.splitext(os.path.basename(fp))[0]
        out_fp = os.path.join(out_dir, f"{sid}.ply")
        try:
            data = np.load(fp)
            xyz_np = np.asarray(data["points"], dtype=np.float32)
            nrm_np = np.asarray(data["normals"], dtype=np.float32)
            if xyz_np.ndim != 2 or xyz_np.shape[1] != 3 or xyz_np.shape[0] == 0:
                print(f"[SKIP] {sid}: bad points array {xyz_np.shape}", flush=True)
                continue

            input_xyz = torch.from_numpy(xyz_np).float().to(device)
            input_normal = torch.from_numpy(nrm_np).float().to(device)
            # Normalise normals defensively (NKSR expects unit normals as the feature).
            input_normal = input_normal / (
                torch.linalg.norm(input_normal, dim=-1, keepdim=True) + 1e-8
            )

            with torch.no_grad():
                # Official inference call (README / examples/recons_simple.py).
                # detail_level=1.0 -> finest density-adaptive reconstruction; the
                # density rescaling is reversed on output, so the mesh comes back
                # in the SAME [-1,1] input frame.
                field = reconstructor.reconstruct(
                    input_xyz, input_normal, detail_level=1.0
                )
                if field is None:
                    print(f"[SKIP] {sid}: reconstructor returned None", flush=True)
                    continue
                mesh = field.extract_dual_mesh(mise_iter=1)

            v = mesh.v.detach().cpu().numpy().astype(np.float32)
            f = mesh.f.detach().cpu().numpy().astype(np.int32)
            if v.shape[0] == 0 or f.shape[0] == 0:
                print(f"[SKIP] {sid}: empty mesh ({v.shape[0]} v, {f.shape[0]} f)", flush=True)
                continue

            write_ply_triangle(out_fp, v, f)
            print(f"[OK]   {sid}: {v.shape[0]} verts, {f.shape[0]} faces -> {out_fp}", flush=True)

        except Exception:
            print(f"[FAIL] {sid}: exception during reconstruction", flush=True)
            traceback.print_exc()
            # Do not crash the batch -- skip this shape and continue.
            torch.cuda.empty_cache()
            continue


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python /app/run.py <in_dir> <out_dir>", flush=True)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])