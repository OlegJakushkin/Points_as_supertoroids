# NeuralPull (ICML 2021) per-shape surface reconstruction runner.
# Official repo: https://github.com/mabaorui/NeuralPull-Pytorch
#
# Stack matches the repo README (Python 3.8, PyTorch 1.11.0, CUDA 11.3).
# CUDA 11.3 supports Ampere sm_86, so this launches kernels on an RTX 3080.
# There are NO pretrained weights: NeuralPull fits one MLP per cloud, so the
# image needs no weight download step.

FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Python 3.8 + build toolchain (PyMCubes builds a small C++ extension).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.8 python3.8-dev python3-pip \
        git build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/bin/python3.8 /usr/bin/python3 && \
    python -m pip install --no-cache-dir --upgrade "pip<24" setuptools wheel

# PyTorch 1.11.0 + cu113 (Ampere-capable) from the official index.
RUN python -m pip install --no-cache-dir \
        torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
        --extra-index-url https://download.pytorch.org/whl/cu113

# Repo Python deps (versions from the README), plus numpy/scipy pinned to
# versions compatible with Python 3.8 + torch 1.11.
RUN python -m pip install --no-cache-dir \
        "numpy==1.22.4" \
        "scipy==1.8.1" \
        tqdm \
        "pyhocon==0.3.57" \
        trimesh \
        PyMCubes

# Clone the official repo so we can import its own modules (models.fields,
# models.dataset). Pin nothing extra: it is plain-torch source, no build step.
RUN git clone --depth 1 https://github.com/mabaorui/NeuralPull-Pytorch /app/NeuralPull
ENV NEURALPULL_DIR=/app/NeuralPull

# Our contract adapter.
COPY run.py /app/run.py
WORKDIR /app

ENTRYPOINT ["python", "/app/run.py"]
CMD ["/in", "/out"]