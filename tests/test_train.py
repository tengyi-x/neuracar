import tempfile
import unittest
from pathlib import Path

from neuracar.train import build_dataset_with_metadata


class DatasetTests(unittest.TestCase):
    def test_label_windows_are_censored_at_partition_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.csv"
            trace.write_text("".join(f"{index},{index % 3},1\n" for index in range(10)))
            X_train, y_train, X_test, y_test, _, _ = build_dataset_with_metadata(
                str(trace), window=2, train_frac=0.7
            )

        self.assertEqual(X_train.shape[0], 5)
        self.assertEqual(y_train.shape[0], 5)
        self.assertEqual(X_test.shape[0], 1)
        self.assertEqual(y_test.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
