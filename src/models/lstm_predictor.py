"""
GHOST - Bidirectional LSTM Risk Predictor with Temporal Attention
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os


class SupplyChainDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class GHOSTLSTMPredictor(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.Tanh(), nn.Linear(64, 1))
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 32),
            nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        self.last_embedding = None

    def forward(self, x):
        out, _ = self.lstm(x)
        w = torch.softmax(self.attention(out).squeeze(-1), dim=1)
        ctx = (out * w.unsqueeze(-1)).sum(dim=1)
        self.last_embedding = ctx.detach()
        return self.risk_head(ctx).squeeze(-1), ctx


class GHOSTTrainer:
    def __init__(self, model, lr=1e-3, device="cuda"):
        self.model     = model.to(device)
        self.device    = device
        self.opt       = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.sched     = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=50)
        self.criterion = nn.MSELoss()
        self.scaler    = StandardScaler()
        self.history   = {"train_loss": [], "val_loss": []}

    def _epoch(self, loader, train=True):
        self.model.train(train)
        total = 0.0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for Xb, yb in loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                p, _ = self.model(Xb)
                loss = self.criterion(p, yb)
                if train:
                    self.opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.opt.step()
                total += loss.item()
        return total / len(loader)

    def train(self, X_tr, y_tr, X_va, y_va, epochs=50, batch_size=64):
        B, T, F = X_tr.shape
        Xs = self.scaler.fit_transform(X_tr.reshape(-1, F)).reshape(B, T, F)
        Bv, Tv, _ = X_va.shape
        Xvs = self.scaler.transform(X_va.reshape(-1, F)).reshape(Bv, Tv, F)
        tl = DataLoader(SupplyChainDataset(Xs,  y_tr), batch_size, shuffle=True)
        vl = DataLoader(SupplyChainDataset(Xvs, y_va), batch_size)
        best, best_state = float("inf"), None
        for ep in range(epochs):
            tr = self._epoch(tl, True); va = self._epoch(vl, False)
            self.sched.step()
            self.history["train_loss"].append(tr)
            self.history["val_loss"].append(va)
            if va < best:
                best = va
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            if (ep + 1) % 10 == 0:
                print(f"  Epoch {ep+1:3d}/{epochs} | train={tr:.4f}  val={va:.4f}")
        self.model.load_state_dict(best_state)
        print(f"  Best val loss: {best:.4f}")
        return self.history

    def predict(self, X):
        self.model.eval()
        B, T, F = X.shape
        Xs = self.scaler.transform(X.reshape(-1, F)).reshape(B, T, F)
        loader = DataLoader(SupplyChainDataset(Xs, np.zeros(B)), 64)
        preds, embeds = [], []
        with torch.no_grad():
            for Xb, _ in loader:
                p, e = self.model(Xb.to(self.device))
                preds.append(p.cpu().numpy())
                embeds.append(e.cpu().numpy())
        return np.concatenate(preds), np.concatenate(embeds)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"model": self.model.state_dict(), "scaler": self.scaler}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.scaler = ckpt["scaler"]
