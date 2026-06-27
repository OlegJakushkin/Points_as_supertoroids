# CAP-UDF batch runner (https://github.com/junshengzhou/CAP-UDF)
# Per-shape UDF optimisation, NO pretrained weights.
# Official stack: Python 3.8, PyTorch 1.11.0, CUDA 11.3 (cu113) -> supports
# sm_86 (RTX 3080 Ampere), so it runs on the target host.
#
# We need the -devel image (nvcc) to compile the chamfer_dist CUDA extension.
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# Build the chamfer CUDA ext for Ampere (sm_86); +PTX for forward-compat.
ENV TORCH_CUDA_ARCH_LIST="8.6+PTX"
# Make CUDA headers/libs from the base image visible to the build.
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ninja-build \
        python3.8 python3.8-dev python3-pip python3-tk \
        libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.8 /usr/bin/python \
    && ln -sf /usr/bin/python3.8 /usr/bin/python3

RUN python -m pip install --no-cache-dir --upgrade "pip==23.*" "setuptools<70" wheel

# PyTorch 1.11.0 + cu113 (sm_86-capable) from the official cu113 index.
RUN python -m pip install --no-cache-dir \
        torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
        --extra-index-url https://download.pytorch.org/whl/cu113

# README pip dependencies (exact pins where the README pins them) + numpy/cython
# that PyMCubes / point_cloud_utils build against.
RUN python -m pip install --no-cache-dir \
        "numpy==1.23.5" cython \
        tqdm "pyhocon==0.3.57" trimesh PyMCubes scipy \
        "point_cloud_utils==0.29.7"

# ---- the CAP-UDF source, pinned to the verified commit ----
WORKDIR /app
RUN git clone https://github.com/junshengzhou/CAP-UDF /app/CAPUDF \
    && cd /app/CAPUDF && git checkout 0b360a3

# Build the one CUDA extension (chamfer distance) -> installs module `chamfer`.
RUN cd /app/CAPUDF/extensions/chamfer_dist && python setup.py install

# tools/utils.py has a broken top-level `from tkinter import Variable` (Variable
# is not a tkinter symbol) and an unused `from random import sample`. run.py
# imports remove_far/remove_outlier from this module, so the bad import would
# crash at import time. Strip both lines; nothing in the codebase uses them.
RUN sed -i '/from tkinter import Variable/d; /from random import sample/d' \
        /app/CAPUDF/tools/utils.py

# Drop in our batch adapter.
COPY run.py /app/run.py

# Run from inside the repo so `models`, `extensions`, `tools` import cleanly.
WORKDIR /app/CAPUDF
ENV PYTHONPATH=/app/CAPUDF

ENTRYPOINT ["python", "/app/run.py"]