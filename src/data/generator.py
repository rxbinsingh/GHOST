"""
GHOST - Data Generator: Synthetic Supply Chain Network + Disruption Events
"""

import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

DISRUPTION_TYPES = {
    "port_strike":           {"delay_factor": 0.8, "cost_factor": 1.5, "duration": 7,  "severity": 0.7},
    "raw_material_shortage": {"delay_factor": 0.6, "cost_factor": 2.0, "duration": 14, "severity": 0.8},
    "transportation_delay":  {"delay_factor": 0.7, "cost_factor": 1.3, "duration": 3,  "severity": 0.5},
    "geopolitical_conflict": {"delay_factor": 0.4, "cost_factor": 2.5, "duration": 21, "severity": 0.9},
    "natural_disaster":      {"delay_factor": 0.3, "cost_factor": 3.0, "duration": 10, "severity": 1.0},
    "demand_surge":          {"delay_factor": 0.9, "cost_factor": 1.2, "duration": 5,  "severity": 0.3},
    "supplier_bankruptcy":   {"delay_factor": 0.2, "cost_factor": 4.0, "duration": 30, "severity": 0.95},
    "cyber_attack":          {"delay_factor": 0.5, "cost_factor": 2.2, "duration": 7,  "severity": 0.75},
}

NODE_TYPES = ["supplier", "manufacturer", "warehouse", "distributor", "retailer"]


def generate_supply_chain_graph(n_nodes: int = 20) -> nx.DiGraph:
    G = nx.DiGraph()
    nodes_per_type = n_nodes // len(NODE_TYPES)
    node_id = 0
    for ntype in NODE_TYPES:
        for _ in range(nodes_per_type):
            G.add_node(node_id,
                       type=ntype,
                       capacity=np.random.uniform(100, 1000),
                       reliability=np.random.uniform(0.7, 1.0),
                       inventory=np.random.uniform(50, 500),
                       lead_time=np.random.randint(1, 30),
                       criticality=np.random.uniform(0.3, 1.0))
            node_id += 1

    type_nodes = {t: [n for n, d in G.nodes(data=True) if d["type"] == t]
                  for t in NODE_TYPES}

    for i in range(len(NODE_TYPES) - 1):
        for src in type_nodes[NODE_TYPES[i]]:
            targets = random.sample(type_nodes[NODE_TYPES[i + 1]],
                                    min(np.random.randint(1, 3),
                                        len(type_nodes[NODE_TYPES[i + 1]])))
            for dst in targets:
                G.add_edge(src, dst,
                           weight=np.random.uniform(0.5, 1.0),
                           transport_cost=np.random.uniform(10, 100),
                           transit_time=np.random.randint(1, 14),
                           reliability=np.random.uniform(0.8, 1.0))
    return G


def generate_time_series(n_days: int = 365, n_nodes: int = 20,
                         disruption_prob: float = 0.05) -> pd.DataFrame:
    records = []
    start_date = datetime(2023, 1, 1)
    for day in range(n_days):
        date = start_date + timedelta(days=day)
        for node_id in range(n_nodes):
            disruption, disruption_type, delay, cost_impact, severity = 0, "none", 0.0, 1.0, 0.0
            if np.random.random() < disruption_prob:
                disruption = 1
                disruption_type = random.choice(list(DISRUPTION_TYPES.keys()))
                p = DISRUPTION_TYPES[disruption_type]
                delay       = (1 - p["delay_factor"]) * np.random.uniform(0.8, 1.2)
                cost_impact = p["cost_factor"] * np.random.uniform(0.9, 1.1)
                severity    = p["severity"] * np.random.uniform(0.8, 1.0)
            records.append({
                "date": date, "node_id": node_id,
                "demand":           np.random.normal(100, 20) * (1 + 0.3 * np.sin(2 * np.pi * day / 365)),
                "supply":           np.random.normal(95, 15),
                "inventory_level":  np.random.uniform(20, 200),
                "lead_time":        np.random.randint(1, 30),
                "transport_cost":   np.random.uniform(10, 100) * cost_impact,
                "on_time_delivery": np.random.uniform(0.7, 1.0) * (1 - delay),
                "disruption":       disruption,
                "disruption_type":  disruption_type,
                "delay_factor":     delay,
                "cost_impact":      cost_impact,
                "severity":         severity,
                "risk_score":       disruption * severity,
            })
    return pd.DataFrame(records)


def create_features(df: pd.DataFrame, seq_len: int = 30) -> tuple:
    feature_cols = ["demand", "supply", "inventory_level", "lead_time",
                    "transport_cost", "on_time_delivery", "delay_factor", "cost_impact"]
    X, y, node_ids, dates = [], [], [], []
    for node_id in df["node_id"].unique():
        ndf = df[df["node_id"] == node_id].sort_values("date").reset_index(drop=True)
        feats  = ndf[feature_cols].values
        labels = ndf["risk_score"].values
        for i in range(seq_len, len(feats)):
            X.append(feats[i - seq_len:i])
            y.append(labels[i])
            node_ids.append(node_id)
            dates.append(ndf["date"].iloc[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), node_ids, dates
