# Mathematical Formulations

## Zero-Shot Risk Inference

Risk score inferred from operational anomalies (no labels needed):

```
r_i^(t) = σ( (||z_i^(t)||₂ + λ·δ_i^(t)) / (1 + ||z_i^(t)||₂ + λ·δ_i^(t)) )
```

where `z_i^(t)` is the z-score vector of operational metrics and `δ_i^(t)` is the delivery delay.

## Bidirectional LSTM with Temporal Attention

```
h_t = BiLSTM(x_t, h_{t-1})
α_t = softmax(W_a · tanh(W_h · h_t))
c   = Σ_t α_t · h_t
ŷ   = σ(W_o · c)
```

## Dual-Level GNN Risk Injection

Global injection:
```
h_i^(l) = h_i^(l) + γ · W_g · r_global + β
```

Local edge-attention:
```
e_{ij} = (Q_i · K_j^T / √d) · gate(r_{ij})
h_i^(l+1) = W_o · Σ_j softmax(e_{ij}) · V_j
```

## Bootstrap Self-Distillation

Scenario score:
```
s(sc) = 0.4·sev + 0.3·(1-delay) + 0.3·min(cost/4, 1)
```

Blended training:
```
D_blend = α·D_real + (1-α)·D_synthetic,  α = 0.55
```

## Probe-Based Drift Correction

Drift metric:
```
drift = 1 - cos(e_current, e_reference)
```

Embedding update when drift > threshold:
```
e_ref ← 0.65·e_ref + 0.35·e_current
e_ref ← e_ref / ||e_ref||
```
