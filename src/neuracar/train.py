from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from .features import extract_features, train_test_split_by_position
from .labels import reuse_labels
from .model import FEATURES, ReuseNet
from .trace import Request, read_trace


def cap_infs(X: np.ndarray) -> np.ndarray:
    """Replaces +inf entries (first-access recency/time_since_last_access) with 2x the column's finite max."""
    X = X.copy()
    for col in range(X.shape[1]):
        finite = X[np.isfinite(X[:, col]), col]
        cap = 2 * finite.max() if finite.size else 1.0
        X[~np.isfinite(X[:, col]), col] = cap
    return X


def standardize(X_train: np.ndarray, X_test: np.ndarray):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma


def build_dataset(trace_path: str, window: float, train_frac: float = 0.7, feature_mask: List[str] = None):
    """Loads a trace, builds standardized (X_train, y_train, X_test, y_test) tensors, split chronologically."""
    requests: List[Request] = list(read_trace(trace_path))
    X = cap_infs(extract_features(requests))
    y = np.array(reuse_labels(requests, window), dtype=np.float64).reshape(-1, 1)

    if feature_mask is not None:
        keep = [i for i, name in enumerate(FEATURES) if name in feature_mask]
        X = X[:, keep]

    train_idx, test_idx = train_test_split_by_position(len(requests), train_frac)
    X_train, X_test, _, _ = standardize(X[train_idx], X[test_idx])
    y_train, y_test = y[train_idx], y[test_idx]

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )


def train_reuse_net(X_train, y_train, X_test, y_test, epochs: int = 200, lr: float = 0.01, seed: int = 42):
    torch.manual_seed(seed)
    model = ReuseNet(n_features=X_train.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "test_loss": []}
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_loss = criterion(model(X_test), y_test).item()
        history["train_loss"].append(loss.item())
        history["test_loss"].append(test_loss)

    return model, history


def evaluate(model: ReuseNet, X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    model.eval()
    proba = model.predict_proba(X)
    pred = (proba >= 0.5).float()
    accuracy = (pred == y).float().mean().item()

    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y.numpy(), proba.numpy())
    except (ImportError, ValueError):
        auc = float("nan")

    return {"accuracy": accuracy, "auc": auc}
