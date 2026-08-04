import unittest

from neuracar.simulator import AdaptiveCache, simulate_policy
from neuracar.trace import Request


class RecordingPredictor:
    def __init__(self):
        self.calls = []

    def predict_reuse(self, features):
        self.calls.append([list(row) for row in features])
        # Older objects receive lower P(reuse), so the NN evicts the stalest candidate.
        return [1.0 / row[0] for row in features]


class SimulatorTests(unittest.TestCase):
    def test_lru_known_sequence(self):
        requests = [Request(i, obj, 1) for i, obj in enumerate("ABACBA")]
        result = simulate_policy(requests, capacity=2, policy="LRU")
        self.assertEqual(result.requests, 6)
        self.assertEqual(result.hits, 1)
        self.assertAlmostEqual(result.hit_ratio, 1 / 6)

    def test_byte_capacity_can_require_multiple_evictions(self):
        requests = [Request(0, "a", 2), Request(1, "b", 2), Request(2, "c", 4)]
        result = simulate_policy(requests, capacity=5, policy="LRU")
        self.assertEqual(result.evictions, 2)
        self.assertEqual(result.hits, 0)

    def test_nn_receives_person_a_feature_order_and_evicts_min_reuse(self):
        predictor = RecordingPredictor()
        cache = AdaptiveCache(2, experts=("nn",), predictor=predictor)
        for request in (
            Request(0, "a", 1),
            Request(1, "b", 1),
            Request(2, "a", 1),
            Request(3, "c", 1),
        ):
            cache.access(request)

        self.assertEqual(predictor.calls[-1], [[1.0, 2.0, 1.0, 1.0], [2.0, 1.0, 2.0, 1.0]])
        self.assertIn("a", cache.cache)
        self.assertIn("c", cache.cache)
        self.assertNotIn("b", cache.cache)

    def test_history_miss_penalizes_the_responsible_expert(self):
        predictor = RecordingPredictor()
        cache = AdaptiveCache(
            2,
            predictor=predictor,
            history_discount=1.0,
            learning_rate=1.0,
            seed=3,
        )
        cache.access(Request(0, "a", 1))
        cache.access(Request(1, "b", 1))
        original = set(cache.cache)
        cache.access(Request(2, "c", 1))
        victim = (original - set(cache.cache)).pop()
        responsible = next(iter(cache.selected_experts))
        cache.access(Request(3, victim, 1))

        weights = cache.weights.weights
        self.assertEqual(cache.history_misses, 1)
        self.assertLess(weights[responsible], 1 / 3)
        for expert, weight in weights.items():
            if expert != responsible:
                self.assertGreater(weight, 1 / 3)

    def test_oversized_object_is_not_cached(self):
        cache = AdaptiveCache(2, experts=("lru",))
        self.assertFalse(cache.access(Request(0, "large", 3)))
        self.assertNotIn("large", cache.cache)

    def test_lfu_count_resets_after_reinsertion_but_nn_frequency_is_global(self):
        cache = AdaptiveCache(1, experts=("lfu",))
        cache.access(Request(0, "a", 1))
        cache.access(Request(1, "a", 1))
        self.assertEqual(cache.cache["a"].frequency, 2)
        cache.access(Request(2, "b", 1))
        cache.access(Request(3, "a", 1))
        self.assertEqual(cache.cache["a"].frequency, 1)
        self.assertEqual(cache.frequency["a"], 3)


if __name__ == "__main__":
    unittest.main()
