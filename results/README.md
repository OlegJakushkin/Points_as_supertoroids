# Measurement runs — compose local + Colab with `compose.py`

Each `results/<run>/` holds one measurement run, **same schema** for host and Colab, so they merge:

| file | producer | contents |
|---|---|---|
| `metrics_val.json` | `compare/bench_val.py` | per-shape per-method benchmark (F-score, SDF-error, normal-consistency, mesh-defects, parts, time) |
| `gate_sweep.json` | `compare/gate_sweep.py` | gate accuracy vs density / normal-noise / thresholds |
| `learned_metrics.json` or `learned_vs_ours.json` | `baselines_ext/score.py` / `compare/fig_learned.py` | dockerised learned baselines, if run |
| `check_v2.txt` | `compare/check_v2.py` | v1 vs v2 before/after |
| `meta.json` | (written per run) | `{run, model, date, val_k}` |

**Runs**
- `local/` — produced on this host, current checkpoint (`waveshape_mixed_v2_latest.pt`).
- `colab/` — download your Colab `DRIVE_WS/results` folder and drop it in here as `results/colab/`.

**Compose:** `python compose.py` → `results/comparison.md` — the reviewer quantitative table, with the fixed
public baselines once and **ours** per run, so you can read off what the longer Colab training bought.
