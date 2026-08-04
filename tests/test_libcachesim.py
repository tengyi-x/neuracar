import subprocess
import unittest
from unittest.mock import patch

from neuracar.libcachesim import run_libcachesim_baseline


class LibCacheSimAdapterTests(unittest.TestCase):
    @patch("neuracar.libcachesim.subprocess.run")
    def test_parses_request_and_byte_hit_ratios(self, run):
        run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="LRU miss ratio: 0.25, byte miss ratio: 0.40\n",
            stderr="",
        )
        hit_ratio, byte_hit_ratio, _ = run_libcachesim_baseline(
            "cachesim", "trace.csv", "LRU", 1024
        )
        self.assertAlmostEqual(hit_ratio, 0.75)
        self.assertAlmostEqual(byte_hit_ratio, 0.60)
        run.assert_called_once_with(
            [
                "cachesim",
                "trace.csv",
                "csv",
                "LRU",
                "1024",
                "-t",
                "time-col=1,obj-id-col=2,size-col=3,delimiter=,,has-header=false",
            ],
            check=True,
            text=True,
            capture_output=True,
        )


if __name__ == "__main__":
    unittest.main()
