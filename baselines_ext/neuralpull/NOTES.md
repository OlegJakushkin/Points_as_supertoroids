## method

NeuralPull (ICML 2021), official PyTorch repo https://github.com/mabaorui/NeuralPull-Pytorch. Per-shape optimisation: one MLP fit per cloud, no pretrained weights. Adapter reuses the repo's own models.fields.NPullNetwork and models.dataset.search_nearest_point, and replicates run.py's pull-loss training loop + extract_fields/marching-cubes and confs/npull.conf hyperparameters verbatim.

## feasibility

runnable

## build_cmd

docker build -t neuralpull-runner C:\work\Points_as_supertoroids\compare\neuralpull_runner

## run_cmd

docker run --rm --gpus all -v C:\path\to\in:/in -v C:\path\to\out:/out neuralpull-runner /in /out

## weights_note

NO weights exist or are needed. NeuralPull is per-shape optimisation: it fits one MLP per cloud from scratch (confirmed by grep over the repo -- no .pth shipped, no download URL, no gdown/wget anywhere; the only torch.save is the per-shape checkpoint written DURING a fit). The repo ships only source + a demo cloud (data/gargoyle.ply/.npz). The Dockerfile therefore has no weight-download step; the only network access at build time is `git clone` of the repo and pip wheel downloads.

## ood_note

Not applicable in the usual pretrained sense: there is no released checkpoint, so nothing is ShapeNet/ABC-pretrained. Every shape (ModelNet or otherwise) is optimised from scratch directly on its own input points, so there is no train/test domain gap and no zero-shot/OOD concern -- the method is dataset-agnostic by construction.

## caveats

1) COST: per-shape optimisation runs maxiter=40000 Adam iterations per cloud (repo default) -- on the order of a few minutes per shape on a 3080; a large ModelNet-val batch is many GPU-hours. Lower via --maxiter (e.g. 10000-15000) for a faster, slightly coarser fit. 2) FRAME: the repo's process_data re-centres+re-scales each cloud into its own ~[-0.5,0.5] frame; that would break the contract, so the adapter deliberately bypasses process_data, builds query samples directly in the incoming [-1,1] frame, and forces marching cubes over a fixed [-1,1]^3 grid -- output is directly comparable to the GT mesh. 3) The repo's argparse default --conf ./confs/np_srb.conf does not exist in the repo (only confs/npull.conf is present); the adapter hardcodes the npull.conf model+train hyperparameters instead of parsing a conf file, so this missing-file bug is sidestepped. 4) NORMALS are loaded from the npz but unused -- NeuralPull is normals-free by design. 5) NeuralPull targets watertight surface reconstruction; on noisy/open ModelNet shells it can over-close thin structures (inherent to the method, not the adapter). 6) MEMORY: per-shape data (~1M query pts), 256^3 grid (~67MB) and the 8-layer/256 MLP all fit well within 8GB; --gpus all required. 7) WHEEL PIN: torch==1.11.0+cu113 / torchvision==0.12.0+cu113 are installed via --extra-index-url https://download.pytorch.org/whl/cu113 (the canonical channel the README's cudatoolkit=11.3 conda line resolves to); CUDA 11.3 supports sm_86 so kernels launch on the RTX 3080. Not built/run here per instruction (GPU busy); verified statically: README/setup, run.py, models/{fields,dataset,utils}.py and confs/npull.conf read directly, imported symbols (NPullNetwork, search_nearest_point) confirmed to exist, and run.py AST-parsed clean.

