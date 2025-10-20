import torch
import torch.nn as nn
try:
    from libauc.losses import AUCMLoss
except Exception:
    AUCMLoss = None

class NNAUCHead(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.bce = nn.BCEWithLogitsLoss()
        self.aucl = AUCMLoss() if AUCMLoss is not None else None

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def loss(self, logits, y, alpha=0.5):
        y = y.float()
        if self.aucl is None:
            return self.bce(logits, y)
        # комбинируем AUC-лосс и BCE, чтобы не разрушить калибровку полностью
        return alpha*self.bce(logits, y) + (1-alpha)*self.aucl(logits, y)
