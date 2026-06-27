# NKSR (Neural Kernel Surface Reconstruction) dockerised runner.
#
# The PyPI `nksr` package is a stale placeholder (0.0.0, pure-python, no CUDA
# extension) and the project's old prebuilt wheels have EXPIRED (see the repo's
# 2025-09-08 news entry). The supported install is therefore to COMPILE the CUDA
# extension from source via `pip install --no-build-isolation package/`, which the
# repo pins to PyTorch 2.7.0 + CUDA 12.8 (requirements.txt / environment.yml).
#
# CUDA 12.8 fully supports sm_86 (RTX 3080, Ampere), so this runs on the target GPU.
# We use the -devel image because building the extension needs nvcc + the cuSPARSE/
# cuBLAS/cuSOLVER dev libraries that setup.py links against.

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# Build only for the host GPU's arch (Ampere sm_86) -> faster build, exact match.
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV FORCE_CUDA=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
# Cache the auto-downloaded HF checkpoint inside the image layer.
ENV TORCH_HOME=/root/.cache/torch
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        git ninja-build build-essential \
        libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

RUN python -m pip install --upgrade pip setuptools wheel

# --- PyTorch 2.7.0 + cu128 (matches NKSR requirements.txt) ---
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        "torch==2.7.0+cu128" torchvision

# --- torch-scatter built against the same torch/CUDA ABI ---
RUN pip install --no-cache-dir torch-scatter \
        -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

# --- remaining python deps from the repo's requirements.txt ---
RUN pip install --no-cache-dir \
        "python-pycg>=1.0.1" \
        numpy matplotlib ninja GitPython pyntcloud plyfile \
        omegaconf tqdm requests

# --- clone NKSR and compile the CUDA extension from source ---
# Pinned to a known-good commit so the build is reproducible.
RUN git clone https://github.com/nv-tlabs/NKSR /opt/NKSR \
    && cd /opt/NKSR \
    && git checkout 14d75673b1506ffb39fc9c067ed581a0b0a3777d

# setup.py clones openvdb + eigen into package/external during the build and then
# compiles nksr._C. --no-build-isolation lets it see the torch we already installed.
RUN cd /opt/NKSR/package && pip install --no-build-isolation .

# --- pre-fetch the default 'ks' checkpoint so first run is offline-ready ---
# (Best-effort: if the build host has no network, the first container run will
#  fetch it instead.)
RUN python -c "import torch; torch.hub.load_state_dict_from_url('https://huggingface.co/heiwang1997/nksr-checkpoints/resolve/main/checkpoints/ks.pth')" || \
    echo "[warn] checkpoint pre-fetch skipped (no network at build); will download at runtime"

COPY run.py /app/run.py
WORKDIR /app

ENTRYPOINT ["python", "/app/run.py"]
CMD ["/in", "/out"]