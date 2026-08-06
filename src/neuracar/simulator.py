"""Byte-capacity cache simulator for LeCaR and the three-expert NeuraCaR policy."""

from __future__ import annotations

import heapq
import random
from collections import Counter, OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

from experts import MultiplicativeWeights

from .trace import Request

SUPPORTED_EXPERTS = ("lru", "lfu", "nn")


class ReusePredictor(Protocol):
    def predict_reuse(self, features: Sequence[Sequence[float]]) -> list[float]: ...


@dataclass
class CacheEntry:
    size: float
    frequency: int = 1


@dataclass(frozen=True)
class EvictionRecord:
    expert: str
    request_index: int
    size: float


class GhostHistory:
    """Recent evictions, bounded by the same byte budget as the cache by default."""

    def __init__(self, capacity: float):
        if capacity <= 0:
            raise ValueError("history capacity must be positive")
        self.capacity = float(capacity)
        self.size = 0.0
        self._entries: OrderedDict[str, EvictionRecord] = OrderedDict()

    def add(self, obj_id: str, record: EvictionRecord) -> None:
        previous = self._entries.pop(obj_id, None)
        if previous is not None:
            self.size -= previous.size
        if record.size > self.capacity:
            return
        self._entries[obj_id] = record
        self.size += record.size
        while self.size > self.capacity:
            _, oldest = self._entries.popitem(last=False)
            self.size -= oldest.size

    def pop(self, obj_id: str) -> EvictionRecord | None:
        record = self._entries.pop(obj_id, None)
        if record is not None:
            self.size -= record.size
        return record

    def discard(self, obj_id: str) -> None:
        self.pop(obj_id)


@dataclass(frozen=True)
class SimulationResult:
    requests: int
    hits: int
    request_bytes: float
    hit_bytes: float
    evictions: int
    history_misses: int
    selected_experts: dict[str, int]
    final_weights: dict[str, float]
    snapshots: list[dict[str, float]]

    @property
    def hit_ratio(self) -> float:
        return self.hits / self.requests if self.requests else 0.0

    @property
    def byte_hit_ratio(self) -> float:
        return self.hit_bytes / self.request_bytes if self.request_bytes else 0.0


class AdaptiveCache:
    """LeCaR generalized from two eviction experts to an arbitrary expert set."""

    def __init__(
        self,
        capacity: float,
        experts: Sequence[str] = SUPPORTED_EXPERTS,
        predictor: ReusePredictor | None = None,
        learning_rate: float = 0.45,
        history_discount: float = 0.995,
        history_capacity: float | None = None,
        initial_weights: Mapping[str, float] | None = None,
        nn_candidate_count: int | None = None,
        seed: int = 42,
        snapshot_interval: int | None = None,
    ):
        if capacity <= 0:
            raise ValueError("cache capacity must be positive")
        if not 0.0 < history_discount <= 1.0:
            raise ValueError("history_discount must lie in (0, 1]")
        unknown = set(experts) - set(SUPPORTED_EXPERTS)
        if unknown:
            raise ValueError(f"unsupported experts: {sorted(unknown)}")
        if "nn" in experts and predictor is None:
            raise ValueError("the nn expert requires a reuse predictor")
        if snapshot_interval is not None and snapshot_interval <= 0:
            raise ValueError("snapshot_interval must be positive")
        if nn_candidate_count is not None and nn_candidate_count <= 0:
            raise ValueError("nn_candidate_count must be positive")

        self.capacity = float(capacity)
        self.predictor = predictor
        self.history_discount = history_discount
        self.snapshot_interval = snapshot_interval
        self.weights = MultiplicativeWeights(experts, learning_rate, initial_weights)
        self.nn_candidate_count = nn_candidate_count
        ghost_capacity = history_capacity or capacity
        self.histories = {name: GhostHistory(ghost_capacity) for name in self.weights.experts}
        self.rng = random.Random(seed)

        self.cache: dict[str, CacheEntry] = {}
        self.occupied = 0.0
        # LRU victim in O(1): keys stay in access order, oldest first. Kept in exact sync with
        # self.cache (inserted/removed together) so eviction never needs to scan self.cache.
        self._lru_order: OrderedDict[str, None] = OrderedDict()
        # LFU victim in O(log n) amortized via a lazy-deletion min-heap of (frequency, last_index,
        # obj_id). Stale entries (an object's frequency/last_index has since changed, or it's no
        # longer cached) are discarded when popped rather than removed eagerly.
        self._lfu_heap: list[tuple[int, int, str]] = []
        self.last_index: dict[str, int] = {}
        self.last_time: dict[str, float] = {}
        self._trace_time: float | None = None
        self.frequency: Counter[str] = Counter()
        self.requests = 0
        self.hits = 0
        self.request_bytes = 0.0
        self.hit_bytes = 0.0
        self.evictions = 0
        self.history_misses = 0
        self.selected_experts: Counter[str] = Counter()
        self.snapshots: list[dict[str, float]] = []

    def _features(self, obj_id: str, request_index: int, request_time: float) -> list[float]:
        entry = self.cache[obj_id]
        return [
            float(request_index - self.last_index[obj_id]),
            float(self.frequency[obj_id]),
            float(request_time - self.last_time[obj_id]),
            float(entry.size),
        ]

    def _victim(self, expert: str, request_index: int, request_time: float) -> str:
        if expert == "lru":
            return next(iter(self._lru_order))
        if expert == "lfu":
            while True:
                frequency, last_index, obj = self._lfu_heap[0]
                entry = self.cache.get(obj)
                if entry is not None and entry.frequency == frequency and self.last_index.get(obj) == last_index:
                    return obj
                heapq.heappop(self._lfu_heap)

        candidates = self._nn_candidates()
        features = [self._features(obj, request_index, request_time) for obj in candidates]
        probabilities = self.predictor.predict_reuse(features)  # type: ignore[union-attr]
        if len(probabilities) != len(candidates):
            raise ValueError("reuse predictor returned the wrong number of probabilities")
        if any(not 0.0 <= probability <= 1.0 for probability in probabilities):
            raise ValueError("reuse probabilities must lie in [0, 1]")
        # Minimum P(reuse) is maximum P(non-reuse); use LRU as a deterministic tie-breaker.
        return min(
            zip(candidates, probabilities),
            key=lambda pair: (pair[1], self.last_index[pair[0]], pair[0]),
        )[0]

    def _nn_candidates(self) -> list[str]:
        """Guard the NN with a mix of plausible LRU and LFU victims.

        Scoring a bounded set reduces inference cost and prevents an
        out-of-distribution prediction from evicting an obviously hot object.
        ``None`` preserves the original full-cache NN scan.
        """
        count = self.nn_candidate_count
        if count is None or count >= len(self.cache):
            return list(self.cache)

        lru_order = sorted(self.cache, key=lambda obj: (self.last_index[obj], obj))
        lfu_order = sorted(
            self.cache,
            key=lambda obj: (self.cache[obj].frequency, self.last_index[obj], obj),
        )
        selected: dict[str, None] = {}
        lru_quota = (count + 1) // 2
        lfu_quota = count // 2
        for obj_id in lru_order[:lru_quota]:
            selected[obj_id] = None
        for obj_id in lfu_order[:lfu_quota]:
            selected[obj_id] = None
        for obj_id in lru_order + lfu_order:
            if len(selected) >= count:
                break
            selected[obj_id] = None
        return list(selected)

    def _record_snapshot(self) -> None:
        snapshot = {
            "request": float(self.requests),
            "hit_ratio": self.hits / self.requests,
        }
        snapshot.update({f"weight_{name}": value for name, value in self.weights.weights.items()})
        self.snapshots.append(snapshot)

    def access(self, request: Request) -> bool:
        if request.size <= 0:
            raise ValueError(f"object {request.obj_id!r} has non-positive size")
        if self._trace_time is not None and request.time < self._trace_time:
            raise ValueError("trace timestamps must be non-decreasing")

        request_index = self.requests
        self.requests += 1
        self.request_bytes += request.size
        self._trace_time = request.time
        hit = request.obj_id in self.cache

        if hit:
            entry = self.cache[request.obj_id]
            if entry.size != request.size:
                raise ValueError(f"object {request.obj_id!r} changed size within the trace")
            self.hits += 1
            self.hit_bytes += request.size
            entry.frequency += 1
            self._lru_order.move_to_end(request.obj_id)
            heapq.heappush(self._lfu_heap, (entry.frequency, request_index, request.obj_id))
        else:
            regretted = None
            for history in self.histories.values():
                regretted = history.pop(request.obj_id)
                if regretted is not None:
                    break
            if regretted is not None:
                age = request_index - regretted.request_index
                reward = self.history_discount**age
                self.weights.penalize(regretted.expert, reward)
                self.history_misses += 1

            if request.size <= self.capacity:
                while self.occupied + request.size > self.capacity:
                    expert = self.weights.choose(self.rng)
                    victim = self._victim(expert, request_index, request.time)
                    victim_entry = self.cache.pop(victim)
                    del self._lru_order[victim]
                    self.occupied -= victim_entry.size
                    self.histories[expert].add(
                        victim,
                        EvictionRecord(expert, request_index, victim_entry.size),
                    )
                    self.selected_experts[expert] += 1
                    self.evictions += 1
                for history in self.histories.values():
                    history.discard(request.obj_id)
                self.cache[request.obj_id] = CacheEntry(request.size)
                self.occupied += request.size
                self._lru_order[request.obj_id] = None
                heapq.heappush(self._lfu_heap, (1, request_index, request.obj_id))

        self.frequency[request.obj_id] += 1
        self.last_index[request.obj_id] = request_index
        self.last_time[request.obj_id] = request.time
        if self.snapshot_interval and self.requests % self.snapshot_interval == 0:
            self._record_snapshot()
        return hit

    def run(self, requests: Iterable[Request]) -> SimulationResult:
        for request in requests:
            self.access(request)
        if not self.snapshots or self.snapshots[-1]["request"] != float(self.requests):
            self._record_snapshot()
        return self.result()

    def result(self) -> SimulationResult:
        if self.requests and (not self.snapshots or self.snapshots[-1]["request"] != float(self.requests)):
            self._record_snapshot()
        return SimulationResult(
            requests=self.requests,
            hits=self.hits,
            request_bytes=self.request_bytes,
            hit_bytes=self.hit_bytes,
            evictions=self.evictions,
            history_misses=self.history_misses,
            selected_experts=dict(self.selected_experts),
            final_weights=self.weights.weights,
            snapshots=list(self.snapshots),
        )


def simulate_policy(
    requests: Iterable[Request],
    capacity: float,
    policy: str,
    predictor: ReusePredictor | None = None,
    **kwargs,
) -> SimulationResult:
    """Run an internal LRU, LFU, LeCaR, or NeuraCaR baseline."""
    policies = {
        "LRU": ("lru",),
        "LFU": ("lfu",),
        "LeCaR": ("lru", "lfu"),
        "NeuraCaR": ("lru", "lfu", "nn"),
    }
    try:
        experts = policies[policy]
    except KeyError as error:
        raise ValueError(f"unknown policy {policy!r}; choose from {tuple(policies)}") from error
    cache = AdaptiveCache(capacity, experts=experts, predictor=predictor, **kwargs)
    return cache.run(requests)
