"""Tracking-the-best-expert weight updates used by LeCaR and NeuraCaR."""

import math
import random
from collections.abc import Iterable


class MultiplicativeWeights:
    """A normalized distribution over any number of eviction experts.

    A history miss penalizes the expert responsible for the earlier eviction.
    With two experts this is equivalent, after normalization, to LeCaR's update
    that multiplies the other expert's weight by ``exp(learning_rate * reward)``.
    """

    def __init__(self, experts: Iterable[str], learning_rate: float = 0.45):
        names = tuple(experts)
        if not names or len(names) != len(set(names)):
            raise ValueError("experts must be a non-empty sequence of unique names")
        if learning_rate < 0:
            raise ValueError("learning_rate must be non-negative")
        self.experts = names
        self.learning_rate = learning_rate
        self._weights = {name: 1.0 / len(names) for name in names}

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)

    def choose(self, rng: random.Random) -> str:
        draw = rng.random()
        cumulative = 0.0
        for name in self.experts:
            cumulative += self._weights[name]
            if draw <= cumulative:
                return name
        return self.experts[-1]

    def penalize(self, expert: str, reward: float) -> dict[str, float]:
        if expert not in self._weights:
            raise KeyError(f"unknown expert: {expert}")
        if not 0.0 <= reward <= 1.0:
            raise ValueError("reward must lie in [0, 1]")
        self._weights[expert] *= math.exp(-self.learning_rate * reward)
        total = sum(self._weights.values())
        for name in self.experts:
            self._weights[name] /= total
        return self.weights
