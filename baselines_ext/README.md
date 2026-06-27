# Learned reconstruction baselines (dockerised, honest)

Each `<method>/` holds a self-contained `Dockerfile` + `run.py` that reconstructs a mesh from each input cloud,
implementing one shared contract so a single scorer folds them into our comparison. **Nothing is reimplemented**
— every adapter reuses the method's official code + (where applicable) official weights.

## Contract
- **Input** `/in/<id>.npz`: float32 `points (N,3)`, `normals (N,3)` in the cube frame `[-1,1]^3`.
- **Output** `/out/<id>.ply`: one triangle mesh in the same frame.
- **Entry** `python run.py /in /out`.

## Honest scope
| method | kind | weights | on our shapes |
|---|---|---|---|
| Neural-Pull | per-shape optim | none (fits per cloud) | **fair** (no domain gap) |
| CAP-UDF | per-shape optim | none | **fair** |
| SAP (optim mode) | per-shape optim | none | **fair** |
| POCO | feed-forward | ShapeNet/ABC | **zero-shot / OOD** |
| NKSR | feed-forward | ShapeNet/scenes | **zero-shot / OOD** |
| ConvONet | feed-forward | ShapeNet | **zero-shot / OOD** (re-hosted on CUDA 11.7 for sm_86) |

Feed-forward methods have **no ModelNet checkpoint**, so their numbers are cross-dataset generalization, reported
as such. The per-shape optimizers are the apples-to-apples comparison.

## Run one
```powershell
# 1. export the clouds once (CPU)
docker run --rm -v "C:\work\Points_as_supertoroids:/workspace" -w /workspace -e PYTHONPATH=/workspace `
  waveshape-compare python baselines_ext/export_clouds.py
# 2. build + run a method (GPU)
docker build -t wsn-bl-neuralpull baselines_ext/neuralpull
docker run --rm --gpus all `
  -v "C:\work\Points_as_supertoroids\baselines_ext\clouds:/in" `
  -v "C:\work\Points_as_supertoroids\baselines_ext\out\neuralpull:/out" wsn-bl-neuralpull /in /out
# 3. score everything in baselines_ext/out/* against GT (CPU)
docker run --rm -v "C:\work\Points_as_supertoroids:/workspace" -w /workspace -e PYTHONPATH=/workspace `
  waveshape-compare python baselines_ext/score.py
```
Results land in `baselines_ext/learned_metrics.json` (per shape, per method: F-score, SDF-error,
normal-consistency, chamfer, components, watertightness).
