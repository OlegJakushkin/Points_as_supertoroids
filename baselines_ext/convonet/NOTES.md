## method

ConvONet (Convolutional Occupancy Networks, ECCV 2020) — official ShapeNet 3-plane pretrained model, run via the repo's own encoder/decoder/Generator3D inference path. Files written to C:\work\Points_as_supertoroids\compare\convonet_runner\ (Dockerfile + run.py).

GROUNDED IN THE ACTUAL REPO (pinned commit 838bea5b2f1314f2edbb68d05ebb0db49f1f3bd2):
- Inference entry: generate.py -> generator.generate_mesh(data) where data['inputs'] is a (1,N,3) point cloud, c = model.encode_inputs(inputs), then MISE marching-cubes (src/conv_onet/generation.py Generator3D).
- Model build (src/config.py get_model + configs/pointcloud/shapenet_3plane.yaml): encoder 'pointnet_local_pool' = LocalPoolPointnet(c_dim=32, hidden_dim=32, plane_type=['xz','xy','yz'], plane_resolution=64, unet=True, unet_kwargs={depth:4, merge_mode:concat, start_filts:32}, padding=0.1); decoder 'simple_local' = LocalDecoder(c_dim=32, hidden_size=32, sample_mode=bilinear, padding=0.1). Generator params: threshold=0.2 (shapenet_3plane.yaml test.threshold), resolution_0=32, upsampling_steps=2 (default.yaml). input_type='pointcloud' so vol_bound/vol_info=None (no sliding window).
- My run.py reuses these exact repo classes (LocalPoolPointnet, LocalDecoder, ConvolutionalOccupancyNetwork, Generator3D) rather than generate.py's dataset loader, because the shared contract feeds raw point-cloud .npz, not the full ShapeNet dataset layout (points.npz+occupancies, watertight meshes, metadata.yaml) the loader requires. The model and the inference call are unchanged from upstream.

COORDINATE FRAME (critical, verified in src/common.py): the ShapeNet model is trained on shapes in the UNIT cube [-0.5,0.5] (normalize_coordinate/normalize_3d_coordinate divide by 1+padding and add 0.5). The contract delivers [-1,1]. run.py scales input points by 0.5 (-> [-0.5,0.5]) before encoding, and scales output mesh vertices by 2.0 to map the generated [-0.55,0.55] mesh back into the [-1,1] GT frame.

WEIGHTS: auto-downloaded by torch model_zoo.load_url from the exact URL in configs/pointcloud/pretrained/shapenet_3plane.yaml: https://s3.eu-central-1.amazonaws.com/avg-projects/convolutional_occupancy_networks/models/pointcloud/shapenet_3plane.pt . Checkpoint is a dict {'model': state_dict, ...} (src/checkpoints.py); run.py loads the 'model' entry. Cached in TORCH_HOME (/root/.cache/torch); first run needs internet.

## feasibility

runnable-with-risk

## build_cmd

docker build -t convonet-runner C:\work\Points_as_supertoroids\compare\convonet_runner

## run_cmd

docker run --rm --gpus all -v C:\path\to\in:/in -v C:\path\to\out:/out convonet-runner /in /out

## weights_note

Auto-downloaded at runtime by torch.utils.model_zoo.load_url from the official URL embedded in the repo's own pretrained config (configs/pointcloud/pretrained/shapenet_3plane.yaml): https://s3.eu-central-1.amazonaws.com/avg-projects/convolutional_occupancy_networks/models/pointcloud/shapenet_3plane.pt . No manual download or login. The .pt is a dict {'model': state_dict, ...}; run.py loads the 'model' entry into the freshly built model. Cached under TORCH_HOME=/root/.cache/torch — first container run needs outbound internet to S3; mount/persist that dir (or pre-bake the file) to avoid re-downloading. Weights are ~a few MB (tiny: c_dim=32 model).

## ood_note

The released weights are trained on ShapeNet (subset of 13 categories, object-level, point clouds with sigma=0.005 noise). They are NOT trained on ABC and NOT on ModelNet. Evaluating on ModelNet40 (this project's benchmark) is therefore ZERO-SHOT / OUT-OF-DISTRIBUTION: ModelNet categories and mesh statistics differ from ShapeNet's training set, so reconstruction quality will be a lower bound on the method's in-domain capability and should be reported as zero-shot generalization, not in-domain ConvONet performance. The repo also ships scene-level (synthetic room / Matterport) models, but those are for multi-object scenes in metric units and are the wrong frame for single normalized objects; shapenet_3plane is the correct object-level checkpoint.

## caveats

1) AMPERE/CUDA RE-HOSTING IS THE MAIN RISK SOURCE: upstream pins torch 1.4.0/cu101 which cannot run on sm_86. I deliberately re-host the unchanged official code + official weights on torch 1.13.1/cu117 (sm_86-capable). The pretrained .pt is a plain state_dict of standard nn layers (Linear/Conv2d/ConvTranspose2d) loaded by exact key, and the inference path uses only stable ops (grid_sample, scatter_mean/scatter_max, marching cubes), so numerical results are expected to match the paper's model. This is verified statically (every imported symbol/signature checked against the repo) but NOT executed (GPU busy) — hence 'runnable-with-risk'. If load_state_dict raised a key/shape mismatch, it would surface immediately on first shape; the kwargs were taken verbatim from shapenet_3plane.yaml so a mismatch is not expected.
2) NORMALS UNUSED: ConvONet's point-cloud encoder consumes only xyz (LocalPoolPointnet.forward takes (B,N,3)); the contract's `normals` array is ignored. This matches upstream — ConvONet does not use input normals.
3) FRAME RESCALE is essential (input *0.5, output *2.0). If the scorer ever changes the GT frame, update IN_SCALE/OUT_SCALE accordingly. Output vertices land in ~[-1.1,1.1] (the 0.1 padding), co-registered with [-1,1] GT — correct for Chamfer/SDF comparison.
4) numpy pinned <1.24 (np.float/np.int aliases) and Cython 0.29.x so the repo's .pyx extensions build cleanly; trimesh 3.x is used (repo originally used 2.37 but 3.x exports PLY fine and is more robust). If a future trimesh removes implicit binary-PLY-by-extension, pass file_type='ply'.
5) 8 GB FIT: tiny model (c_dim=32) + N~4096 points + 32^3 grid with 2 MISE upsampling steps (effective 128^3) easily fits in 8 GB for single-object generation. No OOM expected.
6) FIRST-RUN INTERNET: needed for the S3 weight download (and, at build time, for pip wheels + git clone). Air-gapped hosts must pre-bake the .pt into TORCH_HOME.
7) Not executed/verified at runtime per instructions (GPU busy); build and inference were verified only statically against the cloned repo.

