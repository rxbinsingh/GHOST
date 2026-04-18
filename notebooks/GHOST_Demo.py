# ============================================================
# GHOST Demo — Google Colab Notebook
# Run each cell in order
# ============================================================

# CELL 1 — Install dependencies
# !pip install torch networkx scikit-learn pandas numpy matplotlib scipy -q

# CELL 2 — Run GHOST (synthetic data, no download needed)
# Paste the contents of src/ghost_complete.py here and run

# CELL 3 — Run GHOST on real DataCo dataset
# 1. Download DataCoSupplyChainDataset.csv from:
#    https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis
# 2. Upload to Colab
# 3. Uncomment and run:
#
# report = main('/content/DataCoSupplyChainDataset.csv')

# ============================================================
# Quick usage example (modular API)
# ============================================================

import sys
sys.path.insert(0, '..')

from src.data.generator import generate_supply_chain_graph, generate_time_series, create_features
from src.models.lstm_predictor import GHOSTLSTMPredictor, GHOSTTrainer
from src.models.gnn_simulator import GHOSTGNNSimulator, graph_to_tensors
from src.models.disruption_gen import GHOSTDisruptionGenerator
from src.core.bootstrap_loop import GHOSTBootstrapLoop
from src.core.drift_correction import GHOSTDriftCorrector
from src.decision.support import generate_report, print_report
import numpy as np

DEVICE = "cpu"  # change to "cuda" on GPU

# 1. Generate data
G  = generate_supply_chain_graph(n_nodes=20)
df = generate_time_series(n_days=365, n_nodes=20)
X, y, _, _ = create_features(df, seq_len=30)
split = int(0.8 * len(X))

# 2. Train LSTM
model   = GHOSTLSTMPredictor(input_dim=8)
trainer = GHOSTTrainer(model, device=DEVICE)
trainer.train(X[:split], y[:split], X[split:], y[split:], epochs=20)

# 3. Bootstrap loop
gen  = GHOSTDisruptionGenerator(n_nodes=20, device=DEVICE)
loop = GHOSTBootstrapLoop(trainer, gen)
boot_res = loop.run(X[:split], y[:split], X[split:], y[split:], n_iterations=2)

# 4. Drift correction
dc = GHOSTDriftCorrector(model, device=DEVICE)
dc.set_reference(X[split:][:50], trainer.scaler)
dc.correct(X[split:][:50], y[split:][:50], trainer.scaler)

# 5. GNN simulation
preds, _ = trainer.predict(X[split:])
global_risk = float(np.mean(preds))
nf, adj, er, gr = graph_to_tensors(G, global_risk, DEVICE)
import torch
gnn = GHOSTGNNSimulator().to(DEVICE)
with torch.no_grad():
    impact, _ = gnn(nf, adj, er, gr)
impact_np = impact.cpu().numpy()

# 6. Zero-shot scenarios
scenarios = [gen.zero_shot_from_text(t) for t in [
    "dock workers have walked off the job",
    "earthquake has struck the manufacturing region",
    "hackers encrypted all warehouse systems",
]]

# 7. Decision report
report = generate_report(scenarios, impact_np, G)
print_report(report)
