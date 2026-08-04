import torch
import torch.nn as nn

FEATURES = ["recency", "frequency", "time_since_last_access", "size"]


class ReuseNet(nn.Module):
    """Predicts P(object is reused before eviction) from recency/frequency/time-since-last-access/size."""

    def __init__(self, n_features: int = len(FEATURES), hidden_sizes=(16, 8)):
        super().__init__()
        h1, h2 = hidden_sizes
        self.n_features = n_features
        self.hidden_sizes = tuple(hidden_sizes)
        self.net = nn.Sequential(
            nn.Linear(n_features, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x):
        return self.net(x)

    def predict_proba(self, x):
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))
