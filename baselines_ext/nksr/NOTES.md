## method

NKSR (Neural Kernel Surface Reconstruction, nv-tlabs). Feed-forward neural-kernel reconstructor: a sparse-voxel U-Net encodes the oriented point cloud, then a global sparse linear (gradient-fitting) solve produces an implicit field, and a dual marching-cubes (extract_dual_mesh) extracts the mesh. Official inference path used verbatim from the repo's README / NKSR-USAGE.md / examples/recons_simple.py: nksr.Reconstructor(device, config='ks').reconstruct(xyz, normal, detail_level=1.0) then field.extract_dual_mesh(mise_iter=1). NKSR is generalising (no per-shape optimisation), so one reconstruct() call per cloud. Default 'ks' (kitchen-sink) checkpoint, voxel_size 0.1 with detail_level density-adaptive rescaling that is reversed on output, so the mesh is returned in the SAME input frame ([-1,1]) with no manual rescale.

## feasibility

runnable-with-risk

## build_cmd

docker build -t nksr-runner C:\work\Points_as_supertoroids\_nksr_runner

## run_cmd

docker run --gpus all --rm -v C:\path\to\in:/in -v C:\path\to\out:/out nksr-runner /in /out

## weights_note

Weights are auto-downloaded by the nksr package via torch.hub.load_state_dict_from_url (see package/nksr/configs.py). The default config 'ks' (kitchen-sink) pulls https://huggingface.co/heiwang1997/nksr-checkpoints/resolve/main/checkpoints/ks.pth and caches it under TORCH_HOME (/root/.cache/torch in the image). The Dockerfile pre-fetches it at build time (best-effort; falls back to first-run download). No manual download or HF token needed. The full snet-* family also exists on that HF repo (snet-n3k-wnormal/-wonormal, snet-nr3k-*, snet-p1k-*, plus p2s.pth and carla.pth) and is selectable via nksr.Reconstructor(device, config='snet'|'snet-wonormal'); 'snet-wonormal' is the normals-free variant. License on weights is CC-BY-SA 4.0 (code is the NVIDIA Source Code License).

## ood_note

The default 'ks' checkpoint is trained on a MIX of object + scene datasets (ShapeNet objects + Points2Surf/ABC + indoor/outdoor scenes), NOT on ModelNet. The selectable snet-* checkpoints are trained on ShapeNet. Either way, ModelNet evaluation here is strictly ZERO-SHOT / OUT-OF-DISTRIBUTION (no ModelNet-trained NKSR weights exist). NKSR generalises across scales, but no NKSR variant was trained on ModelNet, so reported scores reflect cross-dataset generalisation, not in-domain performance.

## caveats

(1) `pip install nksr` from PyPI does NOT work: the PyPI package is a stale 0.0.0 pure-python placeholder with no CUDA extension, and the project's old prebuilt CUDA wheels have EXPIRED (repo's 2025-09-08 news). The only supported path is compiling the CUDA `_C` extension from source via `pip install --no-build-isolation package/`, pinned by the repo to torch 2.7.0 + CUDA 12.8 -- this is what the Dockerfile does. (2) Fixed a real bug in the repo's own requirements.txt: it installs torch==2.7.0 but points the torch-scatter index at torch-2.8.0+cu128.html (ABI mismatch -> import failure). I pin torch-scatter to the matching torch-2.7.0+cu128 index (verified the pt27cu128 cp310 linux wheel exists on data.pyg.org). (3) The build runs nvcc compilation of a large CUDA/OpenVDB(nanovdb) extension and clones openvdb+eigen from git at build time -- needs network and ~10-20+ min; this is why feasibility is 'runnable-with-risk' rather than 'runnable' (cannot empirically confirm the compile succeeds without building, which was disallowed). (4) Output frame: NKSR's detail_level path divides input xyz by an internal density scale then multiplies the extracted mesh back by the same scale (BaseField.set_scale / extract_dual_mesh), so the mesh is returned in the SAME [-1,1] frame as the input -- no manual rescaling needed. mesh.v/mesh.f are GPU torch tensors (MeshingResult). (5) Memory: object-scale (~4096 pts) at voxel_size 0.1 with detail_level easily fits in 8 GB; set PYTORCH_NO_CUDA_MEMORY_CACHING=1 only if OOM appears. (6) sm_86 is fully supported by CUDA 12.8; TORCH_CUDA_ARCH_LIST is pinned to 8.6 so the extension is built specifically for the RTX 3080. (7) The PLY writer is hand-rolled (binary_little_endian, triangle faces) to avoid extra deps; per-shape failures are caught and skipped so the batch never crashes. (8) If finer object detail is wanted, switching config to 'snet' (ShapeNet, voxel_size 0.02) is a one-line change, but note ShapeNet ONet meshes were normalised to ~unit-cube extent while inputs here are extent-2 ([-1,1]); the 'ks'+detail_level choice avoids that scale assumption.

