"""
GHOST - Zero-Shot Disruption Scenario Generator
Novel: maps unseen disruption text descriptions to plausible supply chain
       impact scenarios without having seen that disruption type in training.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple

from src.models.disruption_gen import (GHOSTDisruptionGenerator,
                                       DisruptionScenario, DISRUPTION_PARAMS, TEXT_MAP)


class GHOSTZeroShotGenerator:
    """
    Zero-Shot Disruption Scenario Generator.

    Given a free-text description of a disruption event, generates a
    plausible supply chain impact scenario by:
      1. Keyword-based semantic matching to known disruption types
      2. Parameter interpolation for unseen disruption types
      3. Graph-aware affected node selection based on disruption semantics
    """

    KEYWORD_MAP = {
        "strike":      "port_strike",
        "port":        "port_strike",
        "dock":        "port_strike",
        "shortage":    "raw_material_shortage",
        "material":    "raw_material_shortage",
        "mining":      "raw_material_shortage",
        "weather":     "transportation_delay",
        "flood":       "transportation_delay",
        "snow":        "transportation_delay",
        "storm":       "transportation_delay",
        "war":         "geopolitical_conflict",
        "sanction":    "geopolitical_conflict",
        "conflict":    "geopolitical_conflict",
        "earthquake":  "natural_disaster",
        "hurricane":   "natural_disaster",
        "disaster":    "natural_disaster",
        "fire":        "natural_disaster",
        "demand":      "demand_surge",
        "surge":       "demand_surge",
        "spike":       "demand_surge",
        "bankrupt":    "supplier_bankruptcy",
        "closure":     "supplier_bankruptcy",
        "cyber":       "cyber_attack",
        "hack":        "cyber_attack",
        "ransomware":  "cyber_attack",
        "malware":     "cyber_attack",
    }

    def __init__(self, G: nx.DiGraph, n_nodes: int = 20):
        self.G       = G
        self.n_nodes = n_nodes
        self.gen     = GHOSTDisruptionGenerator(n_nodes)

    def _match_disruption_type(self, text: str) -> Tuple[str, float]:
        text_lower = text.lower()
        type_scores: Dict[str, float] = {t: 0.0 for t in DISRUPTION_PARAMS}

        for keyword, dtype in self.KEYWORD_MAP.items():
            if keyword in text_lower:
                type_scores[dtype] += 1.0

        for dtype, desc in TEXT_MAP.items():
            words_d = set(desc.lower().split())
            words_i = set(text_lower.split())
            type_scores[dtype] += len(words_d & words_i) * 0.3

        best_type  = max(type_scores, key=type_scores.get)
        confidence = min(type_scores[best_type] / 5.0, 1.0)

        if type_scores[best_type] == 0:
            best_type  = "transportation_delay"
            confidence = 0.1

        return best_type, confidence

    def _select_affected_nodes(self, dtype: str, confidence: float) -> List[int]:
        node_types = nx.get_node_attributes(self.G, "type")
        tier_map = {
            "port_strike":           ["distributor", "retailer"],
            "raw_material_shortage": ["supplier", "manufacturer"],
            "transportation_delay":  ["warehouse", "distributor"],
            "geopolitical_conflict": ["supplier", "manufacturer", "distributor"],
            "natural_disaster":      list(set(node_types.values())),
            "demand_surge":          ["retailer", "distributor"],
            "supplier_bankruptcy":   ["supplier"],
            "cyber_attack":          ["warehouse", "manufacturer"],
        }
        target_tiers   = tier_map.get(dtype, list(set(node_types.values())))
        candidate_nodes = [n for n, t in node_types.items() if t in target_tiers]

        if not candidate_nodes:
            candidate_nodes = list(self.G.nodes())

        n_affected = max(1, int(len(candidate_nodes) * confidence * 0.5))
        n_affected = min(n_affected, len(candidate_nodes))

        criticalities = nx.get_node_attributes(self.G, "criticality")
        if criticalities:
            probs = np.array([criticalities.get(n, 0.5) for n in candidate_nodes])
            probs = probs / probs.sum()
            affected = np.random.choice(candidate_nodes, n_affected,
                                        replace=False, p=probs).tolist()
        else:
            affected = np.random.choice(candidate_nodes, n_affected,
                                        replace=False).tolist()
        return affected

    def generate_from_text(self, text: str) -> DisruptionScenario:
        """Zero-shot: text description → DisruptionScenario."""
        dtype, confidence = self._match_disruption_type(text)
        affected          = self._select_affected_nodes(dtype, confidence)
        p                 = DISRUPTION_PARAMS[dtype]

        severity = p["sev"] * (0.5 + 0.5 * confidence)
        feats    = self.gen._scenario_features(dtype, affected, severity)

        sc = DisruptionScenario(
            disruption_type  = dtype,
            affected_nodes   = affected,
            severity         = float(np.clip(severity, 0, 1)),
            duration         = int(p["dur"] * (0.8 + 0.4 * confidence)),
            delay_factor     = float(1 - p["delay"] * confidence),
            cost_impact      = float(p["cost"] * (0.5 + 0.5 * confidence)),
            risk_score       = float(severity),
            features         = feats,
            text_description = text,
        )

        print(f"  [Zero-Shot] '{text[:60]}'" if len(text) <= 60
              else f"  [Zero-Shot] '{text[:60]}...'")
        print(f"    → Matched: {dtype} (confidence: {confidence:.2f})")
        print(f"    → Severity: {sc.severity:.3f} | Duration: {sc.duration}d | Risk: {sc.risk_score:.3f}")
        return sc

    def batch_generate(self, texts: List[str]) -> List[DisruptionScenario]:
        return [self.generate_from_text(t) for t in texts]
