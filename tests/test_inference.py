import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from neuracar.inference import TorchReusePredictor, save_reuse_checkpoint
from neuracar.model import ReuseNet


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_round_trip_preserves_predictions_and_normalization(self):
        model = ReuseNet()
        mean = np.array([1.0, 2.0, 3.0, 4.0])
        scale = np.array([2.0, 3.0, 4.0, 5.0])
        features = [[5.0, 8.0, 11.0, 14.0], [1.0, 2.0, 3.0, 4.0]]
        standardized = (np.asarray(features) - mean) / scale
        expected = model.predict_proba(torch.tensor(standardized, dtype=torch.float32)).reshape(-1).tolist()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "reuse_net.pt"
            save_reuse_checkpoint(str(checkpoint), model, mean, scale)
            actual = TorchReusePredictor(str(checkpoint)).predict_reuse(features)

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
