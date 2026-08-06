import math
import random
import unittest

from experts import MultiplicativeWeights


class MultiplicativeWeightsTests(unittest.TestCase):
    def test_custom_initial_weights_are_normalized(self):
        policy = MultiplicativeWeights(
            ("lru", "lfu", "nn"),
            initial_weights={"lru": 4.5, "lfu": 4.0, "nn": 1.5},
        )
        self.assertEqual(policy.weights, {"lru": 0.45, "lfu": 0.4, "nn": 0.15})

    def test_two_expert_penalty_matches_lecar_other_expert_reward(self):
        policy = MultiplicativeWeights(("lru", "lfu"), learning_rate=0.45)
        weights = policy.penalize("lru", reward=0.8)
        expected_lfu = math.exp(0.45 * 0.8) / (1.0 + math.exp(0.45 * 0.8))
        self.assertAlmostEqual(weights["lfu"], expected_lfu)
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_three_expert_choice_is_reproducible(self):
        first = MultiplicativeWeights(("lru", "lfu", "nn"))
        second = MultiplicativeWeights(("lru", "lfu", "nn"))
        rng_a = random.Random(7)
        rng_b = random.Random(7)
        self.assertEqual([first.choose(rng_a) for _ in range(20)], [second.choose(rng_b) for _ in range(20)])


if __name__ == "__main__":
    unittest.main()
