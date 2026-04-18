"""
GHOST - Decision Support System
Translates risk scores + impact analysis into actionable recommendations.
"""

import numpy as np
import networkx as nx
from typing import List, Dict

from src.models.disruption_gen import DisruptionScenario


RISK_LEVELS = {
    (0.0, 0.2): ("LOW",      "🟢", "Monitor — no immediate action required"),
    (0.2, 0.4): ("MODERATE", "🟡", "Review inventory buffers and supplier alternatives"),
    (0.4, 0.6): ("HIGH",     "🟠", "Activate contingency plans and diversify suppliers"),
    (0.6, 0.8): ("SEVERE",   "🔴", "Immediate rerouting and emergency procurement"),
    (0.8, 1.0): ("CRITICAL", "🚨", "Full crisis response — activate all backup systems"),
}

MITIGATION_STRATEGIES = {
    "port_strike":           ["Reroute via alternative ports",
                              "Increase air freight allocation",
                              "Pre-position inventory at inland warehouses"],
    "raw_material_shortage": ["Activate secondary suppliers",
                              "Substitute materials where possible",
                              "Increase safety stock levels"],
    "transportation_delay":  ["Switch to alternative transport modes",
                              "Expedite critical shipments",
                              "Notify downstream partners of delays"],
    "geopolitical_conflict": ["Diversify supplier geography",
                              "Stockpile critical components",
                              "Review trade compliance requirements"],
    "natural_disaster":      ["Activate disaster recovery plan",
                              "Reroute all affected node traffic",
                              "Deploy emergency inventory reserves"],
    "demand_surge":          ["Increase production capacity",
                              "Prioritize high-margin orders",
                              "Implement demand rationing"],
    "supplier_bankruptcy":   ["Immediately qualify backup suppliers",
                              "Secure alternative contracts",
                              "Audit all single-source dependencies"],
    "cyber_attack":          ["Isolate affected systems",
                              "Switch to manual backup processes",
                              "Engage cybersecurity incident response"],
}


def get_risk_level(score: float) -> tuple:
    for (lo, hi), info in RISK_LEVELS.items():
        if lo <= score < hi:
            return info
    return ("CRITICAL", "🚨", "Full crisis response")


def identify_critical_nodes(G: nx.DiGraph, impact_scores: np.ndarray) -> List[Dict]:
    centrality = nx.betweenness_centrality(G)
    critical   = []
    for node_id, impact in enumerate(impact_scores):
        if node_id >= G.number_of_nodes():
            break
        node_data  = G.nodes[node_id]
        crit_score = (0.4 * impact +
                      0.4 * centrality.get(node_id, 0) +
                      0.2 * node_data.get("criticality", 0.5))
        critical.append({
            "node_id":    node_id,
            "type":       node_data.get("type", "unknown"),
            "impact":     round(float(impact), 3),
            "centrality": round(centrality.get(node_id, 0), 3),
            "crit_score": round(crit_score, 3),
        })
    return sorted(critical, key=lambda x: x["crit_score"], reverse=True)


def generate_report(scenarios:     List[DisruptionScenario],
                    impact_scores: np.ndarray,
                    G:             nx.DiGraph) -> Dict:
    if len(scenarios) > 0:
        overall_risk = np.mean([s.risk_score for s in scenarios])
    else:
        overall_risk = float(np.mean(impact_scores)) if len(impact_scores) > 0 else 0.0

    level, icon, action = get_risk_level(overall_risk)

    report = {
        "summary": {
            "risk_level":   level,
            "icon":         icon,
            "overall_risk": round(float(overall_risk), 3),
            "action":       action,
            "n_scenarios":  len(scenarios),
            "n_nodes":      G.number_of_nodes(),
        },
        "critical_nodes":  identify_critical_nodes(G, impact_scores)[:5],
        "scenarios":       [],
        "recommendations": [],
        "overall_risk":    round(float(overall_risk), 3),
    }

    seen_types = set()
    for sc in sorted(scenarios, key=lambda s: s.risk_score, reverse=True)[:5]:
        slevel, sicon, _ = get_risk_level(sc.risk_score)
        strategies = MITIGATION_STRATEGIES.get(sc.disruption_type, [])
        report["scenarios"].append({
            "type":           sc.disruption_type,
            "risk_score":     round(sc.risk_score, 3),
            "severity":       round(sc.severity, 3),
            "duration_days":  sc.duration,
            "affected_nodes": sc.affected_nodes,
            "risk_level":     slevel,
            "icon":           sicon,
            "strategies":     strategies,
        })
        if sc.disruption_type not in seen_types:
            report["recommendations"].extend(strategies[:2])
            seen_types.add(sc.disruption_type)

    report["recommendations"] = list(dict.fromkeys(report["recommendations"]))[:8]
    return report


def print_report(report: Dict):
    s = report["summary"]
    print("\n" + "="*60)
    print(f"  GHOST THREAT ASSESSMENT REPORT")
    print("="*60)
    print(f"  {s['icon']} Overall Risk Level : {s['risk_level']}")
    print(f"  Risk Score          : {s['overall_risk']:.3f}")
    print(f"  Action Required     : {s['action']}")
    print(f"  Scenarios Analyzed  : {s['n_scenarios']}")
    print(f"  Network Nodes       : {s['n_nodes']}")

    print("\n  TOP CRITICAL NODES:")
    print(f"  {'Node':>5} {'Type':<15} {'Impact':>8} {'Centrality':>12} {'Score':>8}")
    print("  " + "-"*50)
    for n in report["critical_nodes"]:
        print(f"  {n['node_id']:>5} {n['type']:<15} {n['impact']:>8.3f} "
              f"{n['centrality']:>12.3f} {n['crit_score']:>8.3f}")

    print("\n  TOP DISRUPTION SCENARIOS:")
    for sc in report["scenarios"]:
        print(f"\n  {sc['icon']} {sc['type'].upper().replace('_',' ')}")
        print(f"     Risk: {sc['risk_score']:.3f} | Severity: {sc['severity']:.3f} | Duration: {sc['duration_days']}d")
        for strat in sc["strategies"][:2]:
            print(f"     → {strat}")

    print("\n  RECOMMENDED ACTIONS:")
    for i, rec in enumerate(report["recommendations"], 1):
        print(f"  {i:2d}. {rec}")
    print("="*60)
