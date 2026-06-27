## method

POCO (Point Convolution for Surface Reconstruction, CVPR 2022, valeoai/POCO). FKAConv point-convolution encoder computes a per-input-point latent; an InterpAttention K-heads decoder answers occupancy queries by learned attention over the K nearest input-point latents; a region-growing marching-cubes routine (export_mesh_and_refine_vertices_region_growing_v2) extracts and vertex-refines the iso-surface. The adapter reuses POCO's OWN functions: it builds the network exactly as generate.py:main() does (networks.Network(3, latent, 2, "FKAConv", {InterpAttentionKHeadsNet,k=64})), loads checkpoint["state_dict"], calls net.get_latent(data, with_correction=False), then the repo's export_mesh_and_refine_vertices_region_growing_v2(...). Only POCO's file-list dataset/dataloader is replaced by a per-/in/<id>.npz feeder that constructs the identical data dict (pos=(1,3,N); x=(1,3,N) = normals for the normals model, or ones for the no-normals model). Output vertices are emitted in the input point frame ([-1,1]) since no autoscale is applied (scale=1) and the routine maps grid coords back via bmin/step from the input points.

## feasibility

runnable

## build_cmd

docker build -t poco-runner C:\work\Points_as_supertoroids\compare\poco_runner

## run_cmd

docker run --rm --gpus all -v C:\path\to\in:/in -v C:\path\to\out:/out poco-runner /in /out
# To use the ShapeNet model that consumes the input NORMALS instead of the default no-normals ABC model:
#   docker run --rm --gpus all -e POCO_MODEL_DIR=/weights/ShapeNet_Normals_FKAConv_InterpAttentionKHeadsNet_None -v C:\path\to\in:/in -v C:\path\to\out:/out poco-runner /in /out

## weights_note

Weights are baked into the image at build time from the official GitHub release v0.0.0 (verified to resolve, 200 OK): ABC_3k.zip (~106 MB) -> /weights/ABC_3k_FKAConv_InterpAttentionKHeadsNet_None/{config.yaml,checkpoint.pth}, and ShapeNet_3k_normals.zip (~111 MB) -> /weights/ShapeNet_Normals_FKAConv_InterpAttentionKHeadsNet_None/. Checkpoint format is {'epoch','state_dict','optimizer'} loaded via net.load_state_dict(checkpoint['state_dict']); input conv net.cv0.cv.weight is (64,3,1,16) confirming in_channels=3. Verified configs: ABC_3k has normals=false, manifold_points=3000, latent=32, n_labels=2, decoder_k=64 (so it ignores the contract's input normals and uses a constant ones feature). ShapeNet model has normals=true and DOES consume the provided normals. Default = ABC_3k (POCO's canonical diverse-CAD object model). Other available releases (not downloaded): ShapeNet_3k.zip (no-normals), ABC_10k.zip, SyntheticRooms_10k.zip.

## ood_note

Released POCO weights are ShapeNet and ABC ONLY -- there is NO ModelNet checkpoint. Evaluating either model on ModelNet is therefore zero-shot / out-of-distribution: the ABC model was trained on ABC CAD meshes and the ShapeNet model on the 13 ShapeNet categories, neither of which is ModelNet. POCO is a learned feed-forward method (not per-shape optimization), so there is no test-time fitting that could adapt it to the new distribution; results are pure generalization.

## caveats

1) sm_86 is supported: torch 1.8.1+cu111 ships Ampere (sm_80/sm_86) kernels, so the RTX 3080 launches kernels; this is the oldest stack that both supports sm_86 and loads the released weights. 2) numpy is pinned to 1.21.6 because POCO's generate.py uses the removed np.int alias on the meshing path (lines 64, 141) -- numpy>=1.24 would crash; the adapter reuses that exact unmodified function. 3) No custom CUDA/C++ ops are on the inference path -- POCO's kNN uses scipy.spatial.KDTree and sampling uses torch_geometric.voxel_grid (pure python/torch); the only compiled extensions (triangle_hash, pykdtree) are eval-only and built per README purely for repo parity, NOT required for reconstruction. 4) The contract's normals are USED only if you select the ShapeNet normals model (POCO_MODEL_DIR=ShapeNet_Normals...); the default ABC_3k model is the no-normals variant and replaces them with a constant ones feature, matching its training. 5) Input clouds (~4096 pts) are subsampled to 3000 to match the released "3k" training regime (override via POCO_MANIFOLD_POINTS; set <=0 to use all points). 6) gen_resolution defaults to 128 (README's fast setting) which fits comfortably in 8 GB for a single object; raise to 256 for the paper-quality setting if memory allows. 7) Versions not pinned by the repo (torch-geometric 1.7.2, open3d 0.15.2, scikit-image 0.19.3, pandas 1.3.5, trimesh 3.22.4) were chosen for cp37 / torch-1.8 compatibility and the skimage.measure.marching_cubes API POCO calls; these are the main empirical risk and the reason this is rated runnable rather than guaranteed. 8) Per instructions, nothing was built or executed -- everything is statically verified against repo commit 7f85ace3f353cedb63d77e9aafaf6342af4710a3 (run.py parses; Network and export_mesh_and_refine_vertices_region_growing_v2 signatures match the calls). 9) Output PLY vertices are in the input [-1,1] frame because no autoscale is applied (scale=1) and the routine maps grid coords back via bmin/step derived from the input points.

