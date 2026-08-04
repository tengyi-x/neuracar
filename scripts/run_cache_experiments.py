"""Compare NeuraCaR with internal policies and optional libCacheSim baselines across cache sizes."""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from neuracar.inference import TorchReusePredictor
from neuracar.libcachesim import run_libcachesim_baseline
from neuracar.simulator import simulate_policy
from neuracar.trace import read_trace


def parse_size(value: str) -> int:
    suffixes = {"k": 1024, "m": 1024**2, "g": 1024**3}
    normalized = value.strip().lower()
    multiplier = suffixes.get(normalized[-1], 1)
    number = normalized[:-1] if multiplier != 1 else normalized
    size = int(float(number) * multiplier)
    if size <= 0:
        raise argparse.ArgumentTypeError("cache sizes must be positive")
    return size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_path", help="Prepared time,obj_id,size CSV trace")
    parser.add_argument("checkpoint", help="ReuseNet checkpoint produced by train_reuse_net.py")
    parser.add_argument("--cache-sizes", nargs="+", type=parse_size, required=True)
    parser.add_argument("--output", default="results/cache_experiments.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.45)
    parser.add_argument("--history-discount", type=float, default=0.995)
    parser.add_argument("--snapshot-interval", type=int, default=10000)
    parser.add_argument("--libcachesim", help="Path to the optional cachesim executable")
    parser.add_argument(
        "--libcachesim-algorithms",
        nargs="+",
        default=["LRU", "LFU", "ARC", "LeCaR", "LHD", "LRB"],
    )
    parser.add_argument(
        "--trace-params",
        default="time-col=1,obj-id-col=2,size-col=3,delimiter=,,has-header=false",
        help="libCacheSim CSV reader parameters",
    )
    args = parser.parse_args()

    requests = list(read_trace(args.trace_path))
    predictor = TorchReusePredictor(args.checkpoint)
    rows = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for capacity in args.cache_sizes:
        for policy in ("LRU", "LFU", "LeCaR", "NeuraCaR"):
            result = simulate_policy(
                requests,
                capacity,
                policy,
                predictor=predictor if policy == "NeuraCaR" else None,
                seed=args.seed,
                learning_rate=args.learning_rate,
                history_discount=args.history_discount,
                snapshot_interval=args.snapshot_interval,
            )
            rows.append(
                {
                    "source": "python",
                    "policy": policy,
                    "cache_size": capacity,
                    "requests": result.requests,
                    "hit_ratio": result.hit_ratio,
                    "byte_hit_ratio": result.byte_hit_ratio,
                    "final_weights": repr(result.final_weights),
                }
            )
            if policy == "NeuraCaR":
                timeline_path = output.with_name(f"{output.stem}_{capacity}_weights.csv")
                with timeline_path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=result.snapshots[0].keys())
                    writer.writeheader()
                    writer.writerows(result.snapshots)

        if args.libcachesim:
            for algorithm in args.libcachesim_algorithms:
                hit_ratio, byte_hit_ratio, _ = run_libcachesim_baseline(
                    args.libcachesim,
                    args.trace_path,
                    algorithm,
                    capacity,
                    args.trace_params,
                )
                rows.append(
                    {
                        "source": "libCacheSim",
                        "policy": algorithm,
                        "cache_size": capacity,
                        "requests": len(requests),
                        "hit_ratio": hit_ratio,
                        "byte_hit_ratio": byte_hit_ratio if byte_hit_ratio is not None else "",
                        "final_weights": "",
                    }
                )

    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
