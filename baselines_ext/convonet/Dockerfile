# ConvONet (Convolutional Occupancy Networks, ECCV 2020) surface-reconstruction runner.
#
# Upstream pins torch 1.4.0 / CUDA 10.1, which CANNOT launch kernels on an Ampere
# RTX 3080 (sm_86 requires CUDA >= 11). We therefore run the *same official code +
# the same official ShapeNet 3-plane weights* on a newer, sm_86-capable stack
# (CUDA 11.7 + PyTorch 1.13.1 + torch-scatter 2.1.1). The pretrained .pt is a plain
# state_dict of nn.Linear / nn.Conv2d / nn.ConvTranspose2d layers, which loads
# unchanged across this torch jump; the only torch-version-fragile helper in the repo
# (b_inv -> torch.gesv) is never called on the point-cloud inference path.
#
# The repo's C/Cython mesh extensions (libmcubes / libmise / libsimplify / etc.) are
# pure numpy/C and version-independent; we build them with `setup.py build_ext`.

FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # torch hub cache for the auto-downloaded pretrained .pt
    TORCH_HOME=/root/.cache/torch

# ---- system + python 3.8 + build toolchain (for Cython/C++ extensions) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.8 python3.8-dev python3-pip \
        build-essential g++ git ca-certificates wget \
        libgomp1 libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.8 /usr/bin/python && \
    python -m pip install --upgrade "pip<24" "setuptools<60" wheel

# ---- PyTorch 1.13.1 + cu117 (supports sm_86) ----
RUN python -m pip install \
        torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
        --extra-index-url https://download.pytorch.org/whl/cu117

# ---- torch-scatter matching pt1.13 / cu117 (prebuilt wheel, no compile) ----
RUN python -m pip install torch-scatter==2.1.1 \
        -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

# ---- remaining python deps used by the repo's inference + I/O path ----
# (Cython needed to build the mesh extensions; numpy<1.24 keeps np.float/np.int
#  aliases that some of the .pyx C glue and trimesh 2.x rely on.)
RUN python -m pip install \
        "Cython==0.29.36" \
        "numpy==1.23.5" \
        "scipy==1.9.3" \
        "scikit-image==0.19.3" \
        "trimesh==3.22.4" \
        "plyfile==0.7.4" \
        "PyYAML==6.0" \
        "tqdm==4.65.0" \
        "pandas==1.4.4" \
        "tensorboardX==2.6"

# ---- clone the official repo at a pinned commit and build C/Cython extensions ----
ARG CONVONET_SHA=838bea5b2f1314f2edbb68d05ebb0db49f1f3bd2
RUN git clone https://github.com/autonomousvision/convolutional_occupancy_networks /opt/convonet && \
    cd /opt/convonet && git checkout ${CONVONET_SHA} && \
    python setup.py build_ext --inplace

# repo on PYTHONPATH so `import src...` resolves
ENV PYTHONPATH=/opt/convonet:${PYTHONPATH}

# ---- our adapter ----
COPY run.py /app/run.py

WORKDIR /opt/convonet
ENTRYPOINT ["python", "/app/run.py"]
CMD ["/in", "/out"]