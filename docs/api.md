# API Reference

## Data Generation

### `generate_supply_chain_graph(n_nodes=20)`
Returns a `nx.DiGraph` with layered supply chain topology.

### `generate_time_series(n_days=365, n_nodes=20, disruption_prob=0.05)`
Returns a `pd.DataFrame` with synthetic supply chain time series.

### `create_features(df, seq_len=30)`
Returns `(X, y, node_ids, dates)` — sliding window sequences.

---

## Models

### `GHOSTLSTMPredictor(input_dim=8, hidden_dim=128, num_layers=2, dropout=0.3)`
Bidirectional LSTM with temporal attention. Returns `(risk_score, embedding)`.

### `GHOSTTrainer(model, lr=1e-3, device='cuda')`
- `.train(X_tr, y_tr, X_va, y_va, epochs=50)` — trains the model
- `.predict(X)` — returns `(predictions, embeddings)`

### `GHOSTGNNSimulator(node_feat_dim=6, edge_feat_dim=4, risk_dim=16, hidden_dim=64, n_layers=3)`
Dual-level risk injection GNN. Returns `(impact_scores, node_embeddings)`.

### `graph_to_tensors(G, global_risk=0.5, device='cpu')`
Converts a NetworkX graph to model-ready tensors.

---

## Core Algorithms

### `GHOSTDisruptionGenerator(n_nodes=20, device='cpu')`
- `.generate(n_scenarios=16, risk_model=None, top_k=8)` — bootstrapped scenario generation
- `.zero_shot_from_text(text)` — zero-shot text → scenario mapping
- `.scenarios_to_training_data(scenarios, seq_len=30)` — convert to training arrays

### `GHOSTBootstrapLoop(trainer, generator)`
- `.run(X_real, y_real, X_val, y_val, n_iterations=3, ...)` — self-distillation loop

### `GHOSTDriftCorrector(model, device='cuda', lr=5e-4)`
- `.set_reference(X_probe, scaler)` — establish reference embedding
- `.correct(X_probe, y_probe, scaler, n_rounds=2)` — probe-based drift correction

### `GHOSTZeroShotGenerator(G, n_nodes=20)`
- `.generate_from_text(text)` — graph-aware zero-shot scenario generation
- `.batch_generate(texts)` — batch processing

---

## Decision Support

### `generate_report(scenarios, impact_scores, G)`
Returns a structured risk assessment report dict.

### `print_report(report)`
Pretty-prints the threat assessment report.
