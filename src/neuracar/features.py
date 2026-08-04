from typing import Dict, List, Tuple

import numpy as np

from .trace import Request


def extract_features(requests: List[Request]) -> np.ndarray:
    """Streams through requests in order, computing state-at-time-of-request features per object:
    recency (requests since this object was last seen), frequency (access count so far),
    time_since_last_access (time delta since last access), and size.
    First access to an object gets recency = i (distance back to the start of the trace) and
    time_since_last_access = float('inf').
    Returns an (n, 4) array in FEATURES order: [recency, frequency, time_since_last_access, size].
    """
    last_index: Dict[str, int] = {}
    last_time: Dict[str, float] = {}
    freq: Dict[str, int] = {}

    n = len(requests)
    feats = np.zeros((n, 4), dtype=np.float64)
    for i, req in enumerate(requests):
        recency = i - last_index.get(req.obj_id, -1)
        time_since = req.time - last_time.get(req.obj_id, -np.inf)
        frequency = freq.get(req.obj_id, 0)

        feats[i] = [recency, frequency, time_since, req.size]

        last_index[req.obj_id] = i
        last_time[req.obj_id] = req.time
        freq[req.obj_id] = frequency + 1

    return feats


def train_test_split_by_position(n: int, train_frac: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
    """Splits indices [0, n) chronologically: first `train_frac` for training, rest for testing."""
    split = int(round(n * train_frac))
    return np.arange(split), np.arange(split, n)
