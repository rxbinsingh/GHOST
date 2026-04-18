"""
GHOST - GNN Simulator
Novel: Dual-Level Risk Injection
  - Global level : risk embedding broadcast to all nodes (coarse)
  - Local level  : edge-attention gated by per-edge risk features (fine-grained)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np


class GlobalRiskInjection(nn.Module):
    def __init__(self, node_dim, risk_dim):
        super().__init__()
        self.proj  = nn.Sequential(nn.Linear(risk_dim, node_dim),
                                   nn.LayerNorm(node_dim), nn.SiLU())
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))

    def forward(self, h, global_risk):
        return h + self.scale * self.proj(global_risk.unsqueeze(0)) + self.shift


class RiskEdgeAttention(nn.Module):
    def __init__(self, node_dim, edge_dim, heads=4):
        super().__init__()
        self.heads    = heads
        self.head_dim = node_dim // heads
        self.q = nn.Linear(node_dim, node_dim)
        self.k = nn.Linear(node_dim, node_dim)
        self.v = nn.Linear(node_dim, node_dim)
        self.risk_gate = nn.Sequential(nn.Linear(edge_dim, heads), nn.Sigmoid())
        self.out = nn.Linear(node_dim, node_dim)

    def forward(self, h, adj, edge_risk):
        N, D = h.shape
        Q = self.q(h).view(N, self.heads, self.head_dim)
        K = self.k(h).view(N, self.heads, self.head_dim)
        V = self.v(h).view(N, self.heads, self.head_dim)
        scores = torch.einsum("ihd,jhd->ijh", Q, K) / (self.head_dim ** 0.5)
        gate   = self.risk_gate(edge_risk)
        scores = scores * gate
        scores = scores.masked_fill((adj == 0).unsqueeze(-1).expand_as(scores), -1e9)
        attn   = F.softmax(scores, dim=1)
        out    = torch.einsum("ijh,jhd->ihd", attn, V).reshape(N, D)
        return self.out(out)


class GHOSTGNNBlock(nn.Module):
    def __init__(self, node_dim, edge_dim, risk_dim):
        super().__init__()
        self.global_inj = GlobalRiskInjection(node_dim, risk_dim)
        self.edge_attn  = RiskEdgeAttention(node_dim, edge_dim)
        self.norm1 = nn.LayerNorm(node_dim)
        self.norm2 = nn.LayerNorm(node_dim)
        self.ffn   = nn.Sequential(nn.Linear(node_dim, node_dim * 2),
                                   nn.GELU(), nn.Linear(node_dim * 2, node_dim))

    def forward(self, h, adj, edge_risk, global_risk):
        h = self.global_inj(h, global_risk)
        h = self.norm1(h + self.edge_attn(h, adj, edge_risk))
        h = self.norm2(h + self.ffn(h))
        return h


class GHOSTGNNSimulator(nn.Module):
    def __init__(self, node_feat_dim=6, edge_feat_dim=4,
                 risk_dim=16, hidden_dim=64, n_layers=3):
        super().__init__()
        self.node_embed  = nn.Linear(node_feat_dim, hidden_dim)
        self.risk_embed  = nn.Linear(1, risk_dim)
        self.layers      = nn.ModuleList([
            GHOSTGNNBlock(hidden_dim, edge_feat_dim, risk_dim)
            for _ in range(n_layers)])
        self.impact_head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, node_feats, adj, edge_risk, global_risk_scalar):
        h = self.node_embed(node_feats)
        g = self.risk_embed(global_risk_scalar.view(1))
        for layer in self.layers:
            h = layer(h, adj, edge_risk, g)
        return self.impact_head(h).squeeze(-1), h


def graph_to_tensors(G, global_risk=0.5, device="cpu"):
    N = G.number_of_nodes()
    node_feats, adj, edge_risk = [], np.zeros((N, N)), np.zeros((N, N, 4))
    for n in range(N):
        d = G.nodes[n]
        node_feats.append([
            d.get("capacity",    500) / 1000,
            d.get("reliability", 0.9),
            d.get("inventory",   100) / 500,
            d.get("lead_time",   10)  / 30,
            G.in_degree(n)  / N,
            G.out_degree(n) / N,
        ])
    for u, v, d in G.edges(data=True):
        adj[u, v] = 1
        edge_risk[u, v] = [d.get("weight", 0.5),
                           d.get("transport_cost", 50) / 100,
                           d.get("transit_time",   7)  / 14,
                           d.get("reliability",    0.9)]
    return (torch.FloatTensor(node_feats).to(device),
            torch.FloatTensor(adj).to(device),
            torch.FloatTensor(edge_risk).to(device),
            torch.FloatTensor([global_risk]).to(device))
