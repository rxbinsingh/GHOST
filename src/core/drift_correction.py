"""
GHOST - Probe-Based Drift Correction
Novel: runs a probe inference after each cycle, measures embedding drift,
       and self-corrects by blending + fine-tuning when drift exceeds threshold.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional

from src.models.lstm_predictor import GHOSTLSTMPredictor, SupplyChainDataset


class GHOSTDriftCorrector:
    """
    Probe-Based Drift Correction.

    After each prediction cycle:
      1. Run a probe inference on a small probe set
      2. Compute embedding drift = 1 - cosine_similarity(current, reference)
      3. If drift > threshold: blend embeddings and fine-tune
    """

    DRIFT_THRESHOLD = 0.015

    def __init__(self, model: GHOSTLSTMPredictor,
                 device: str = "cuda",
                 lr: float = 5e-4):
        self.model     = model
        self.device    = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.reference_embedding: Optional[np.ndarray] = None
        self.drift_history = []

    def set_reference(self, X_probe: np.ndarray, scaler) -> np.ndarray:
        self.model.eval()
        B, T, F = X_probe.shape
        Xs = scaler.transform(X_probe.reshape(-1, F)).reshape(B, T, F)
        loader = DataLoader(SupplyChainDataset(Xs, np.zeros(B)), batch_size=32)
        embeds = []
        with torch.no_grad():
            for Xb, _ in loader:
                _, e = self.model(Xb.to(self.device))
                embeds.append(e.cpu().numpy())
        self.reference_embedding = np.mean(np.concatenate(embeds), axis=0)
        return self.reference_embedding

    def _compute_drift(self, current_embed: np.ndarray) -> float:
        if self.reference_embedding is None:
            return 0.0
        ref = self.reference_embedding
        cos_sim = (np.dot(current_embed, ref) /
                   (np.linalg.norm(current_embed) * np.linalg.norm(ref) + 1e-8))
        return float(1.0 - cos_sim)

    def _fine_tune(self, X_probe: np.ndarray, y_probe: np.ndarray,
                   scaler, steps: int = 40):
        self.model.train()
        B, T, F = X_probe.shape
        Xs = scaler.transform(X_probe.reshape(-1, F)).reshape(B, T, F)
        loader = DataLoader(SupplyChainDataset(Xs, y_probe), batch_size=16, shuffle=True)
        step = 0
        while step < steps:
            for Xb, yb in loader:
                if step >= steps:
                    break
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                pred, _ = self.model(Xb)
                loss = self.criterion(pred, yb)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                step += 1

    def correct(self, X_probe: np.ndarray, y_probe: np.ndarray,
                scaler, n_rounds: int = 2) -> dict:
        print("\n  [Drift Correction]")
        results = {"rounds": [], "corrected": False}

        for round_idx in range(n_rounds):
            self.model.eval()
            B, T, F = X_probe.shape
            Xs = scaler.transform(X_probe.reshape(-1, F)).reshape(B, T, F)
            loader = DataLoader(SupplyChainDataset(Xs, np.zeros(B)), batch_size=32)
            embeds = []
            with torch.no_grad():
                for Xb, _ in loader:
                    _, e = self.model(Xb.to(self.device))
                    embeds.append(e.cpu().numpy())
            current_embed = np.mean(np.concatenate(embeds), axis=0)

            drift = self._compute_drift(current_embed)
            print(f"    Round {round_idx+1}: drift = {drift:.6f} "
                  f"(threshold = {self.DRIFT_THRESHOLD})")
            self.drift_history.append(drift)

            round_result = {"round": round_idx + 1, "drift": drift, "action": "none"}

            if drift > self.DRIFT_THRESHOLD:
                print(f"    ⚠ Drift detected! Blending + fine-tuning...")
                blended = (0.65 * self.reference_embedding + 0.35 * current_embed)
                self.reference_embedding = blended / (np.linalg.norm(blended) + 1e-8)
                self._fine_tune(X_probe, y_probe, scaler, steps=40)
                round_result["action"] = "corrected"
                results["corrected"] = True
                print(f"    ✓ Correction applied")
            else:
                print(f"    ✓ Drift within bounds — no correction needed")

            results["rounds"].append(round_result)

        return results
