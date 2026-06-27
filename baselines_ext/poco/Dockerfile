# POCO: Point Convolution for Surface Reconstruction (CVPR 2022)
# https://github.com/valeoai/POCO
#
# Stack matches the repo's official environment: Python 3.7.10, CUDA 11.1,
# PyTorch 1.8.1+cu111, torch-geometric companion wheels (scatter/sparse/
# cluster/spline) built for torch-1.8.0+cu111.
#
# sm_86 note: CUDA 11.1 / torch 1.8.1+cu111 ship Ampere kernels (sm_80/sm_86),
# so an RTX 3080 (sm_86) launches kernels fine. This is the OLDEST stack that
# still loads POCO's released weights AND supports sm_86 -- exactly on the
# CUDA>=11.0 boundary required by the host.
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.6"

# ---- APT deps (POCO README) + build toolchain + miniconda fetch ------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget ca-certificates build-essential gcc g++ \
        libgl1-mesa-glx libopenblas-dev libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# ---- Miniconda + Python 3.7.10 (matches environment.yml exactly) -----------
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh -O /tmp/mc.sh && \
    bash /tmp/mc.sh -b -p $CONDA_DIR && rm /tmp/mc.sh
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda install -y python=3.7.10 && conda clean -afy

# ---- POCO source (pinned commit) -------------------------------------------
RUN git clone https://github.com/valeoai/POCO /opt/POCO && \
    cd /opt/POCO && git checkout 7f85ace3f353cedb63d77e9aafaf6342af4710a3

WORKDIR /opt/POCO

# ---- Python deps -----------------------------------------------------------
# numpy<1.24 is REQUIRED: POCO's generate.py uses removed aliases (np.int).
RUN pip install --no-cache-dir --upgrade "pip==23.*" "setuptools<60" wheel && \
    pip install --no-cache-dir \
        "numpy==1.21.6" "scipy==1.7.3" "Cython==0.29.36" "pyyaml" && \
    pip install --no-cache-dir \
        torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 \
        -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir \
        https://data.pyg.org/whl/torch-1.8.0%2Bcu111/torch_scatter-2.0.8-cp37-cp37m-linux_x86_64.whl \
        https://data.pyg.org/whl/torch-1.8.0%2Bcu111/torch_sparse-0.6.12-cp37-cp37m-linux_x86_64.whl \
        https://data.pyg.org/whl/torch-1.8.0%2Bcu111/torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl \
        https://data.pyg.org/whl/torch-1.8.0%2Bcu111/torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl && \
    pip install --no-cache-dir "torch-geometric==1.7.2" && \
    pip install --no-cache-dir \
        "scikit-image==0.19.3" "scikit-learn==1.0.2" \
        "tqdm" "tensorboard" "pandas==1.3.5" "h5py" "pykdtree==1.3.5" \
        "open3d==0.15.2" "plyfile" "trimesh==3.22.4"

# ---- Build POCO CPU extensions (triangle_hash + pykdtree), per README ------
RUN CC=gcc CXX=gcc python setup_c.py build_ext --inplace && \
    python setup.py build_ext --inplace

# ---- Pretrained weights: ABC_3k (default) + ShapeNet normals (optional) ----
# Released weights are ShapeNet and ABC ONLY (no ModelNet) -> ModelNet is OOD.
RUN mkdir -p /weights && cd /weights && \
    wget -q https://github.com/valeoai/POCO/releases/download/v0.0.0/ABC_3k.zip && \
    python -c "import zipfile; zipfile.ZipFile('ABC_3k.zip').extractall('.')" && rm ABC_3k.zip && \
    wget -q https://github.com/valeoai/POCO/releases/download/v0.0.0/ShapeNet_3k_normals.zip && \
    python -c "import zipfile; zipfile.ZipFile('ShapeNet_3k_normals.zip').extractall('.')" && rm ShapeNet_3k_normals.zip

ENV PYTHONPATH=/opt/POCO

# ---- Adapter ---------------------------------------------------------------
COPY run.py /app/run.py

ENV POCO_MODEL_DIR=/weights/ABC_3k_FKAConv_InterpAttentionKHeadsNet_None \
    POCO_GEN_RESOLUTION=128 \
    POCO_REFINE_ITER=10 \
    POCO_MANIFOLD_POINTS=3000

ENTRYPOINT ["python", "/app/run.py"]
CMD ["/in", "/out"]