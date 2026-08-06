"""Checkpoint I/O and normalized inference for the neural eviction expert."""

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .model import FEATURES, ReuseNet


def save_reuse_checkpoint(
    path: str,
    model: ReuseNet,
    mean: Sequence[float],
    scale: Sequence[float],
    feature_names: Sequence[str] = FEATURES,
    feature_transform: str = "identity",
) -> None:
    """Save everything needed to reproduce Person A's standardized inference."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "state_dict": model.state_dict(),
            "n_features": model.n_features,
            "hidden_sizes": list(model.hidden_sizes),
            "feature_names": list(feature_names),
            "feature_transform": feature_transform,
            "mean": np.asarray(mean, dtype=np.float64).tolist(),
            "scale": np.asarray(scale, dtype=np.float64).tolist(),
        },
        destination,
    )


class TorchReusePredictor:
    """Loads a ReuseNet checkpoint and returns P(reuse) for candidate objects."""

    def __init__(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 1:
            raise ValueError("unsupported ReuseNet checkpoint format")

        self.feature_names = tuple(checkpoint["feature_names"])
        if self.feature_names != tuple(FEATURES):
            raise ValueError(
                f"checkpoint features {self.feature_names!r} do not match expected {tuple(FEATURES)!r}"
            )

        self.mean = np.asarray(checkpoint["mean"], dtype=np.float64)
        self.scale = np.asarray(checkpoint["scale"], dtype=np.float64)
        if self.mean.shape != self.scale.shape or self.mean.shape != (len(FEATURES),):
            raise ValueError("checkpoint normalization statistics have the wrong shape")
        if np.any(self.scale <= 0) or not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.scale)):
            raise ValueError("checkpoint normalization statistics must be finite with positive scales")
        self.feature_transform = checkpoint.get("feature_transform", "identity")
        if self.feature_transform not in {"identity", "log1p"}:
            raise ValueError(f"unsupported feature transform: {self.feature_transform!r}")

        self.model = ReuseNet(
            n_features=int(checkpoint["n_features"]),
            hidden_sizes=tuple(checkpoint["hidden_sizes"]),
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def predict_reuse(self, features: Sequence[Sequence[float]]) -> list[float]:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError(f"expected an (n, {len(self.feature_names)}) feature matrix")
        if self.feature_transform == "log1p":
            if np.any(values < 0):
                raise ValueError("log1p features must be non-negative")
            values = np.log1p(values)
        standardized = (values - self.mean) / self.scale
        tensor = torch.tensor(standardized, dtype=torch.float32)
        return self.model.predict_proba(tensor).reshape(-1).tolist()
