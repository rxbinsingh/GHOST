"""
GHOST - Bootstrapped Self-Distillation Loop
Novel: model generates its own synthetic disruption scenarios,
       scores them, and retrains on them — no external labels needed.
"""

import numpy as np
import copy
from typing import List

from src.models.lstm_predictor import GHOSTTrainer
from src.models.disruption_gen import GHOSTDisruptionGenerator, DisruptionScenario


class GHOSTBootstrapLoop:
    """
    Self-Distillation Bootstrap Loop.

    Each iteration:
      1. Generate synthetic disruption scenarios using current model
      2. Score scenarios with balanced metric (risk model + heuristic)
      3. Select top-K diverse scenarios
      4. Retrain model on blended (real + synthetic) data
      5. Evaluate improvement — keep new model if better
    """

    def __init__(self, trainer: GHOSTTrainer,
                 generator: GHOSTDisruptionGenerator):
        self.trainer   = trainer
        self.generator = generator
        self.history   = []

    def _blend(self, real_X: np.ndarray, real_y: np.ndarray,
               syn_X: np.ndarray, syn_y: np.ndarray,
               alpha: float = 0.55) -> tuple:
        n_real = int(len(real_X) * alpha)
        n_syn  = min(len(syn_X), max(1, len(real_X) - n_real))
        idx_r  = np.random.choice(len(real_X), n_real,  replace=False)
        idx_s  = np.random.choice(len(syn_X),  n_syn,   replace=False)
        return (np.concatenate([real_X[idx_r], syn_X[idx_s]]),
                np.concatenate([real_y[idx_r], syn_y[idx_s]]))

    def _mse(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        preds, _ = self.trainer.predict(X_val)
        return float(np.mean((preds - y_val) ** 2))

    def run(self, X_real: np.ndarray, y_real: np.ndarray,
            X_val:  np.ndarray, y_val:  np.ndarray,
            n_iterations: int = 3,
            n_scenarios:  int = 16,
            top_k:        int = 8,
            retrain_epochs: int = 20) -> dict:

        print("\n" + "="*55)
        print("  GHOST Bootstrap Self-Distillation Loop")
        print("="*55)

        baseline_mse = self._mse(X_val, y_val)
        best_mse     = baseline_mse
        best_state   = copy.deepcopy(self.trainer.model.state_dict())
        print(f"  Baseline MSE: {baseline_mse:.6f}")

        for iteration in range(n_iterations):
            print(f"\n  Iteration {iteration + 1}/{n_iterations}")

            scenarios: List[DisruptionScenario] = self.generator.generate(
                n_scenarios=n_scenarios,
                risk_model=self.trainer.model,
                top_k=top_k,
            )
            print(f"    Generated {len(scenarios)} diverse scenarios")
            print(f"    Risk scores: {[round(s.risk_score, 3) for s in scenarios]}")

            syn_X, syn_y = self.generator.scenarios_to_training_data(scenarios)
            X_blend, y_blend = self._blend(X_real, y_real, syn_X, syn_y)
            print(f"    Blended dataset: {len(X_blend)} samples")

            split = int(0.85 * len(X_blend))
            self.trainer.train(X_blend[:split], y_blend[:split],
                               X_blend[split:], y_blend[split:],
                               epochs=retrain_epochs)

            new_mse    = self._mse(X_val, y_val)
            improvement = (best_mse - new_mse) / best_mse * 100
            print(f"    MSE: {new_mse:.6f}  (Δ {improvement:+.2f}%)")

            self.history.append({
                "iteration":   iteration + 1,
                "mse":         new_mse,
                "improvement": improvement,
            })

            if new_mse < best_mse:
                best_mse   = new_mse
                best_state = copy.deepcopy(self.trainer.model.state_dict())
                print(f"    ✓ New best model saved")
            else:
                self.trainer.model.load_state_dict(best_state)
                print(f"    ↩ Reverted to previous best (stability mechanism)")

        print(f"\n  Bootstrap complete. Final MSE: {best_mse:.6f}")
        print("="*55)
        return {"baseline_mse": baseline_mse,
                "final_mse":    best_mse,
                "history":      self.history}
