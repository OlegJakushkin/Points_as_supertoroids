## method

CAP-UDF (Zhou et al., NeurIPS 2022 / TPAMI 2024) — per-shape unsigned-distance-field (UDF) optimisation. For each input cloud it optimises a fresh 8-layer MLP from scratch with the consistency-aware "pull" objective (move query points down the UDF gradient by their predicted distance, minimise CUDA Chamfer-L1 to the surface), then extracts a triangle mesh from the gradient field via the repo's own gradient-aware marching cubes (PyMCubes). There are NO pretrained weights — the README "Train" command IS the inference path (load_ckpt=none). My /app/run.py reuses the repo's own building blocks (models.fields.CAPUDFNetwork, extensions.chamfer_dist.ChamferDistanceL1 CUDA op, tools.surface_extraction.surface_extraction) and faithfully replicates run.py:Runner.train (40k step-1 + 20k step-2 iters, cosine LR with 1000-iter warmup, the progressive field-consistency point-cloud update at step1, exactly as confs/base.conf), fitting one field per /in/<id>.npz and writing /out/<id>.ply.

## feasibility

runnable-with-risk

## build_cmd

docker build -t capudf-runner C:\work\Points_as_supertoroids\_capudf_runner

## run_cmd

docker run --rm --gpus all -v C:\path\to\in:/in -v C:\path\to\out:/out capudf-runner /in /out

## weights_note

NONE. CAP-UDF has no pretrained weights and the repo ships none. It is a per-shape test-time optimisation method: each cloud trains a fresh randomly-initialised MLP from scratch (confs/*.conf all set train.load_ckpt = none). Nothing is downloaded at build or run time except the source repo itself (pinned to commit 0b360a3). The Google-Drive link in the README is only processed input/query DATA for the paper's ShapeNetCars/3DScenes/SRB benchmarks, not model weights, and is not used by this runner.

## ood_note

Not applicable in the usual sense: because there are NO released/pretrained weights, there is zero train/test domain mismatch — the field is fit directly to each ModelNet input cloud at test time. So running on ModelNet is neither "zero-shot" nor "OOD"; every shape is reconstructed by optimising from scratch on that shape's own points. The only dataset-dependent knob is the query-sampling `scale` (models/dataset.py L51, replicated in the adapter as 0.25*sqrt(POINT_NUM_GT/20000) floored at 0.25); the README flags this as having strong influence on object-level quality, and the default is tuned for object-level clouds like ModelNet, so it is appropriate here.

## caveats

1) RUNTIME is the main risk: this is per-shape optimisation at the official 60k-iteration budget (40k + 20k) PLUS a pure-Python triple-nested marching-cubes loop in tools/surface_extraction.py. Expect very roughly 3-8 min/shape on a 3080 (optimisation) plus meshing; a large ModelNet batch will take many GPU-hours. Iteration counts and mesh resolution are env-overridable (CAPUDF_STEP1_MAXITER / CAPUDF_STEP2_MAXITER / CAPUDF_MCUBE_RES) to trade quality for throughput; I defaulted MCUBE_RES to 128 (repo uses 256) for batch sanity. 2) MESHING: the prompt mentioned MeshUDF, but the ACTUAL repo does NOT use MeshUDF — it uses its own gradient-sign-aware marching cubes over PyMCubes (tools/surface_extraction.py); I reuse that exact function. surface_extraction crashes via np.concatenate([]) if the field has no zero-crossing cells (degenerate fit) — this is caught per-shape so the batch continues. 3) FRAME: process_data re-normalises each cloud into its own center/max-extent frame (~[-0.5,0.5]); I capture (center, scale) and apply the inverse to the output mesh so /out/<id>.ply lands back in the input [-1,1] frame for direct GT comparison — without this the mesh would be mis-scaled. 4) REPO BUG patched: tools/utils.py has a top-level `from tkinter import Variable` (invalid symbol) that would crash run.py's imports; the Dockerfile strips it (and an unused `from random import sample`); python3-tk is also installed defensively. 5) The chamfer CUDA ext is built with TORCH_CUDA_ARCH_LIST="8.6+PTX" so it targets sm_86 (RTX 3080). 6) normals in the .npz are ignored (CAP-UDF is normal-free). 7) Clouds with <51 points (KDTree k=51) or <60 points (POINT_NUM split) will fail and be skipped — fine for the ~4096-point contract. 8) I did NOT build or run this (GPU busy, per instructions); it is statically verified against the repo only — the chamfer nvcc build under cu113 + a clean first-shape run are the unverified steps, hence feasibility "runnable-with-risk".

