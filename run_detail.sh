#!/bin/bash
# 4090 detail run: build an aligned sparse+dense cache, then train the wavelet residual toward the DENSE
# GT-like target with GT-based selection for 3 epochs.  Logs to /work/detail_run.log.
exec >>/work/detail_run.log 2>&1
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /work
echo "=== $(date) launching detail run ==="
conda run -n fvdb pip install -q scipy scikit-image trimesh rtree bitsandbytes 2>&1 | tail -1
if [ ! -f data/se_clouds_dense.pt ]; then
    conda run --no-capture-output -n fvdb python -u build_dense_cache.py --cap 3000 || { echo "CACHE BUILD FAILED"; exit 1; }
fi
conda run --no-capture-output -n fvdb python -u train_detail.py \
    --sparse data/se_clouds_aln.pt --dense data/se_clouds_dense.pt \
    --epochs 3 --batch 4 --out waveshape_detail
echo "=== $(date) detail run exited ==="
