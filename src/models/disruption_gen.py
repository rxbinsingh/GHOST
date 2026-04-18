"""
GHOST - Bootstrapped Synthetic Disruption Generator
Novel: model uses its own learned risk patterns to generate + score
       synthetic disruption scenarios without any external dataset.
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class DisruptionScenario:
    disruption_type: str
    affected_nodes:  List[int]
    severity:        float
    duration:        int
    delay_factor:    float
    cost_impact:     float
    risk_score:      float
    features:        np.ndarray = field(default_factory=lambda: np.array([]))
    text_description:str = ""


DISRUPTION_PARAMS = {
    "port_strike":           {"delay": 0.8, "cost": 1.5, "dur": 7,  "sev": 0.7},
    "raw_material_shortage": {"delay": 0.6, "cost": 2.0, "dur": 14, "sev": 0.8},
    "transportation_delay":  {"delay": 0.7, "cost": 1.3, "dur": 3,  "sev": 0.5},
    "geopolitical_conflict": {"delay": 0.4, "cost": 2.5, "dur": 21, "sev": 0.9},
    "natural_disaster":      {"delay": 0.3, "cost": 3.0, "dur": 10, "sev": 1.0},
    "demand_surge":          {"delay": 0.9, "cost": 1.2, "dur": 5,  "sev": 0.3},
    "supplier_bankruptcy":   {"delay": 0.2, "cost": 4.0, "dur": 30, "sev": 0.95},
    "cyber_attack":          {"delay": 0.5, "cost": 2.2, "dur": 7,  "sev": 0.75},
}

TEXT_MAP = {
    "port_strike":           "workers at the port have gone on strike halting all shipments",
    "raw_material_shortage": "critical raw materials are unavailable due to mining disruptions",
    "transportation_delay":  "severe weather has caused major transportation delays",
    "geopolitical_conflict": "geopolitical tensions have closed key trade routes",
    "natural_disaster":      "earthquake has destroyed key infrastructure nodes",
    "demand_surge":          "unexpected demand spike has overwhelmed supply capacity",
    "supplier_bankruptcy":   "primary supplier has filed for bankruptcy",
    "cyber_attack":          "ransomware attack has disabled logistics management systems",
}


class GHOSTDisruptionGenerator:
    """
    Bootstrapped Synthetic Disruption Generator.

    Pipeline:
      1. Sample disruption type from learned risk distribution
      2. Generate scenario features using model's own risk patterns
      3. Score each scenario with a balanced metric
      4. Return top-K diverse scenarios for retraining
    """

    def __init__(self, n_nodes: int = 20, device: str = "cpu"):
        self.n_nodes = n_nodes
        self.device  = device
        self.generated_scenarios: List[DisruptionScenario] = []

    def _scenario_features(self, dtype: str, affected: List[int],
                            severity: float) -> np.ndarray:
        p = DISRUPTION_PARAMS[dtype]
        noise = np.random.normal(0, 0.05, 8)
        base  = np.array([
            100 * (1 - p["delay"]),
            95  * p["delay"],
            np.random.uniform(20, 100),
            p["dur"],
            50  * p["cost"],
            p["delay"] * 0.9,
            1   - p["delay"],
            p["cost"],
        ])
        return np.clip(base + noise * base, 0, None).astype(np.float32)

    def _score(self, scenario: DisruptionScenario, risk_model=None) -> float:
        p = DISRUPTION_PARAMS[scenario.disruption_type]
        base_score = (scenario.severity * 0.4 +
                      (1 - p["delay"]) * 0.3 +
                      min(p["cost"] / 4.0, 1.0) * 0.3)
        if risk_model is not None:
            try:
                feat = torch.FloatTensor(
                    scenario.features[np.newaxis, np.newaxis, :]
                    .repeat(30, axis=1)).to(self.device)
                with torch.no_grad():
                    pred, _ = risk_model(feat)
                base_score = 0.6 * base_score + 0.4 * pred.item()
            except Exception:
                pass
        return float(np.clip(base_score, 0, 1))

    @staticmethod
    def _is_diverse(candidate: DisruptionScenario,
                    selected: List[DisruptionScenario],
                    threshold: float = 0.85) -> bool:
        if not selected:
            return True
        for s in selected:
            sim = float(np.dot(candidate.features, s.features) /
                        (np.linalg.norm(candidate.features) *
                         np.linalg.norm(s.features) + 1e-8))
            if sim > threshold:
                return False
        return True

    def generate(self, n_scenarios: int = 16,
                 risk_model=None,
                 top_k: int = 8) -> List[DisruptionScenario]:
        candidates = []
        for _ in range(n_scenarios):
            dtype    = np.random.choice(list(DISRUPTION_PARAMS.keys()))
            p        = DISRUPTION_PARAMS[dtype]
            n_aff    = np.random.randint(1, max(2, self.n_nodes // 4))
            affected = np.random.choice(self.n_nodes, n_aff, replace=False).tolist()
            severity = p["sev"] * np.random.uniform(0.8, 1.2)
            feats    = self._scenario_features(dtype, affected, severity)

            sc = DisruptionScenario(
                disruption_type  = dtype,
                affected_nodes   = affected,
                severity         = float(np.clip(severity, 0, 1)),
                duration         = int(p["dur"] * np.random.uniform(0.8, 1.2)),
                delay_factor     = float(1 - p["delay"] * np.random.uniform(0.9, 1.1)),
                cost_impact      = float(p["cost"] * np.random.uniform(0.9, 1.1)),
                risk_score       = 0.0,
                features         = feats,
                text_description = TEXT_MAP[dtype],
            )
            sc.risk_score = self._score(sc, risk_model)
            candidates.append(sc)

        candidates.sort(key=lambda s: s.risk_score, reverse=True)

        selected: List[DisruptionScenario] = []
        for sc in candidates:
            if self._is_diverse(sc, selected):
                selected.append(sc)
            if len(selected) >= top_k:
                break

        if len(selected) < top_k:
            for sc in candidates:
                if sc not in selected:
                    selected.append(sc)
                if len(selected) >= top_k:
                    break

        self.generated_scenarios.extend(selected)
        return selected

    def zero_shot_from_text(self, text: str) -> DisruptionScenario:
        """Zero-shot: map unseen disruption text to a plausible scenario."""
        text_lower = text.lower()
        scores: Dict[str, float] = {}
        for dtype, desc in TEXT_MAP.items():
            words_desc  = set(desc.lower().split())
            words_input = set(text_lower.split())
            overlap     = len(words_desc & words_input)
            scores[dtype] = overlap / (len(words_desc) + 1e-8)

        if max(scores.values()) == 0:
            best_type = np.random.choice(list(DISRUPTION_PARAMS.keys()))
        else:
            best_type = max(scores, key=scores.get)

        p        = DISRUPTION_PARAMS[best_type]
        n_aff    = np.random.randint(1, max(2, self.n_nodes // 4))
        affected = np.random.choice(self.n_nodes, n_aff, replace=False).tolist()
        severity = p["sev"]
        feats    = self._scenario_features(best_type, affected, severity)

        return DisruptionScenario(
            disruption_type  = best_type,
            affected_nodes   = affected,
            severity         = severity,
            duration         = p["dur"],
            delay_factor     = 1 - p["delay"],
            cost_impact      = p["cost"],
            risk_score       = severity,
            features         = feats,
            text_description = text,
        )

    def scenarios_to_training_data(self,
                                   scenarios: List[DisruptionScenario],
                                   seq_len: int = 30) -> tuple:
        X, y = [], []
        for sc in scenarios:
            seq   = np.tile(sc.features, (seq_len, 1))
            noise = np.random.normal(0, 0.02, seq.shape)
            X.append((seq + noise).astype(np.float32))
            y.append(sc.risk_score)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
