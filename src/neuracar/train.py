from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from .features import extract_features, train_test_split_by_position
from .labels import reuse_labels
from .model import FEATURES, ReuseNet
from .trace import Request, read_trace


def fit_inf_caps(X_train: np.ndarray) -> np.ndarray:
    """Fit first-access replacement values using training data only."""
    caps = np.ones(X_train.shape[1], dtype=np.float64)
    for col in range(X_train.shape[1]):
        finite = X_train[np.isfinite(X_train[:, col]), col]
        caps[col] = 2 * finite.max() if finite.size else 1.0
    return caps


def replace_infs(X: np.ndarray, caps: np.ndarray) -> np.ndarray:
    X = X.copy()
    for col, cap in enumerate(caps):
        X[~np.isfinite(X[:, col]), col] = cap
    return X


def transform_features(X: np.ndarray, feature_transform: str) -> np.ndarray:
    if feature_transform == "identity":
        return X
    if feature_transform == "log1p":
        if np.any(X < 0):
            raise ValueError("log1p features must be non-negative")
        return np.log1p(X)
    raise ValueError(f"unsupported feature transform: {feature_transform!r}")


def standardize(X_train: np.ndarray, X_test: np.ndarray):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma


def build_dataset_with_metadata(
    trace_path: str,
    window: float,
    train_frac: float = 0.7,
    feature_mask: List[str] = None,
    feature_transform: str = "log1p",
):
    """Build a chronological dataset and retain training-set normalization metadata."""
    requests: List[Request] = list(read_trace(trace_path))
    if not requests:
        raise ValueError("trace must contain at least one request")
    if window < 0:
        raise ValueError("window must be non-negative")
    X = extract_features(requests)
    y = np.array(reuse_labels(requests, window), dtype=np.float64).reshape(-1, 1)

    if feature_mask is not None:
        keep = [i for i, name in enumerate(FEATURES) if name in feature_mask]
        X = X[:, keep]

    train_idx, test_idx = train_test_split_by_position(len(requests), train_frac)
    if not len(train_idx) or not len(test_idx):
        raise ValueError("train_frac must leave at least one train and one test request")

    # Keep only examples whose complete future label window is observable
    # inside its own chronological partition.
    train_end_time = requests[train_idx[-1]].time
    test_end_time = requests[test_idx[-1]].time
    train_idx = np.asarray(
        [index for index in train_idx if requests[index].time + window <= train_end_time]
    )
    test_idx = np.asarray(
        [index for index in test_idx if requests[index].time + window <= test_end_time]
    )
    if not len(train_idx) or not len(test_idx):
        raise ValueError("reuse window leaves no fully observed train or test examples")

    caps = fit_inf_caps(X[train_idx])
    X = transform_features(replace_infs(X, caps), feature_transform)
    X_train, X_test, mu, sigma = standardize(X[train_idx], X[test_idx])
    y_train, y_test = y[train_idx], y[test_idx]

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
        mu,
        sigma,
    )


def build_dataset(
    trace_path: str,
    window: float,
    train_frac: float = 0.7,
    feature_mask: List[str] = None,
    feature_transform: str = "log1p",
):
    """Loads standardized train/test tensors, split chronologically."""
    return build_dataset_with_metadata(
        trace_path, window, train_frac, feature_mask, feature_transform
    )[:4]


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
