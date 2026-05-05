---
language:
  - en
license: mit
tags:
  - supply-chain
  - risk-prediction
  - graph-neural-network
  - zero-shot-learning
  - self-supervised-learning
  - lstm
  - pytorch
  - time-series
library_name: pytorch
pipeline_tag: time-series-forecasting
---

# GHOST

**Zero-shot supply chain disruption forecasting — no labeled data required.**

[![Paper](https://img.shields.io/badge/paper-ResearchGate-00CCBB?logo=researchgate&logoColor=white)](https://doi.org/10.13140/RG.2.2.27961.94567)
[![GitHub](https://img.shields.io/badge/github-GHOST-181717?logo=github)](https://github.com/rxbinsingh/GHOST)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rxbinsingh/GHOST/blob/main/LICENSE)

---

GHOST (Graph-based Hierarchical On-the-fly Self-correcting Threat detector) predicts supply chain disruptions using only standard operational metrics — no historical disruption labels, no pretraining, no external dataset. It runs in ~15 minutes on a T4 GPU across 180k+ orders.

The system is a closed-loop self-distillation framework: it synthesizes its own disruption scenarios, trains its own risk models, and corrects its own drift — entirely from operational data.

---

## How it works

```
Operational data (orders, lead times, costs)
    │
    ▼
Zero-shot risk inference
  Statistical anomaly detection per node per timestep
  Sigmoid-normalized risk scores — no labels needed
    │
    ▼
Bootstrap synthetic scenario generation
  8 disruption types: port strikes, natural disasters,
  cyber attacks, demand surges, geopolitical conflicts...
  Diversity-constrained candidate selection
    │
    ▼
Bidirectional LSTM with attention
  128 hidden units per direction
  Weighted MSE loss (4x penalty on high-risk misses)
  Bootstrap self-distillation with stability preservation
    │
    ▼
Dual-level GNN risk propagation
  Global: system-wide risk injection to all nodes
  Local: edge-conditioned multi-head attention (8 heads)
  Captures cascading failure patterns
    │
    ▼
Probe-based drift correction
  Embedding centroid monitoring
  Adaptive fine-tuning when drift > 0.015
  No manual retraining or new labels needed
    │
    ▼
Risk predictions + mitigation report
  Per-node risk scores, critical node identification,
  actionable mitigation strategies
```

---

## Results

### Overall performance — DataCo Smart Supply Chain (180,519 orders)

| Metric | Value |
|---|---|
| Prediction MSE | 0.008067 |
| High-risk samples identified | 30,943 (17.2%) |
| Bootstrap stability | 3/3 iterations preserved |
| Drift correction | 0.000014 → 0.000 |
| Zero-shot scenario mapping | 5/5 (100%) |
| Runtime (T4 GPU) | 15.3 minutes |
| Peak memory | 12.4 GB |

### Ablation

| Configuration | MSE | High-Risk ID | Runtime |
|---|---|---|---|
| LSTM only | 0.0089 | 28,456 | 8.2 min |
| + Multi-anchor ensemble | 0.0084 | 29,234 | 9.1 min |
| + Bootstrap (no stability) | 0.0112 | 31,567 | 12.8 min |
| + Stability mechanism | 0.0081 | 30,943 | 13.2 min |
| + Drift correction | 0.0081 | 30,943 | 14.1 min |
| **Full GHOST system** | **0.0081** | **30,943** | **15.3 min** |

### Comparison with baselines

| Method | MSE | High-Risk ID | Runtime |
|---|---|---|---|
| Statistical Threshold | 0.0156 | 18,052 | 0.8 min |
| Isolation Forest | 0.0134 | 22,341 | 3.2 min |
| Random Forest | 0.0098 | 27,123 | 5.7 min |
| LSTM Only | 0.0089 | 28,456 | 8.2 min |
| Graph Attention Network | 0.0091 | 29,012 | 11.4 min |
| **GHOST (Full)** | **0.0081** | **30,943** | **15.3 min** |

---

## Quick start

```bash
git clone https://github.com/rxbinsingh/GHOST
cd GHOST
pip install -r requirements.txt
```

```python
from src.ghost_complete import GHOST

ghost = GHOST()
data  = ghost.load_data('path/to/supply_chain_data.csv')
results = ghost.run_pipeline(data)

risk_scores      = results['risk_predictions']
high_risk_nodes  = results['high_risk_nodes']
```

---

## Repository structure

| Path | Description |
|---|---|
| `src/ghost_complete.py` | Full GHOST pipeline — risk inference, LSTM, GNN, drift correction |
| `src/core/` | Core algorithms — bootstrap, drift detection, stability |
| `src/models/` | LSTM and GNN model definitions |
| `src/data/` | Data loading and preprocessing utilities |
| `src/decision/` | Decision support and mitigation report generation |
| `notebooks/GHOST_Demo.py` | End-to-end demo |
| `docs/` | API reference, installation guide, mathematical formulations |
| `requirements.txt` | Python dependencies |

---

## Dataset

Evaluated on the [DataCo Smart Supply Chain dataset](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) — 180,519 real supply chain orders, 53 operational features, no ground-truth disruption labels. Ideal for zero-shot evaluation.

---

## Requirements

- Python 3.8+
- PyTorch 1.12+
- NetworkX 2.8+
- NVIDIA GPU recommended (T4 or better); CPU supported but slow

---

## Paper

> **GHOST: Self-Bootstrapping Supply Chain Disruption Forecasting via Multi-Scale Risk Propagation and Adaptive Drift Correction**
> Robin Singh, 2025
> [https://doi.org/10.13140/RG.2.2.27961.94567](https://doi.org/10.13140/RG.2.2.27961.94567)

```bibtex
@article{singh2025ghost,
  title   = {GHOST: Self-Bootstrapping Supply Chain Disruption Forecasting
             via Multi-Scale Risk Propagation and Adaptive Drift Correction},
  author  = {Singh, Robin},
  year    = {2025},
  doi     = {10.13140/RG.2.2.27961.94567},
  url     = {https://doi.org/10.13140/RG.2.2.27961.94567}
}
```

---

## License

[MIT](https://github.com/rxbinsingh/GHOST/blob/main/LICENSE) © 2025 Robin Singh
