"""Measure whether NeuraCaR shifts weight away from an NN after a workload change."""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from neuracar.inference import TorchReusePredictor
from neuracar.simulator import AdaptiveCache
from neuracar.trace import Request, read_trace


def limited_trace(path: str, limit: int | None) -> list[Request]:
    requests = list(read_trace(path))
    return requests[:limit] if limit is not None else requests


def shift_timestamps(
    first: list[Request], second: list[Request], namespace: str | None = "phase_b:"
) -> list[Request]:
    if not first or not second:
        return second
    origin = second[0].time
    start = first[-1].time + 1.0
    return [
        Request(
            start + request.time - origin,
            f"{namespace}{request.obj_id}" if namespace is not None else request.obj_id,
            request.size,
        )
        for request in second
    ]


def fixed_policy_phase_b_hit_ratio(
    phase_a: list[Request],
    phase_b: list[Request],
    capacity: float,
    experts: tuple[str, ...],
    predictor: TorchReusePredictor,
    args: argparse.Namespace,
) -> float:
    cache = AdaptiveCache(
        capacity,
        experts=experts,
        predictor=predictor if "nn" in experts else None,
        learning_rate=args.learning_rate,
        history_discount=args.history_discount,
        seed=args.seed,
    )
    for request in phase_a:
        cache.access(request)
    hits_at_shift = cache.result().hits
    for request in phase_b:
        cache.access(request)
    return (cache.result().hits - hits_at_shift) / len(phase_b)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase_a_trace", help="Workload used to train the NN")
    parser.add_argument("phase_b_trace", help="Different/unfamiliar workload")
    parser.add_argument("checkpoint")
    parser.add_argument("--capacity", type=float, required=True)
    parser.add_argument("--phase-a-limit", type=int)
    parser.add_argument("--phase-b-limit", type=int)
    parser.add_argument("--snapshot-interval", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=0.45)
    parser.add_argument("--history-discount", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shared-object-ids",
        action="store_true",
        help="Treat identical IDs in phase A and phase B as the same objects",
    )
    parser.add_argument("--output-prefix", default="results/workload_shift")
    args = parser.parse_args()

    phase_a = limited_trace(args.phase_a_trace, args.phase_a_limit)
    phase_b = shift_timestamps(
        phase_a,
        limited_trace(args.phase_b_trace, args.phase_b_limit),
        namespace=None if args.shared_object_ids else "phase_b:",
    )
    if not phase_a or not phase_b:
        parser.error("both workload phases must contain at least one request")

    predictor = TorchReusePredictor(args.checkpoint)
    cache = AdaptiveCache(
        args.capacity,
        predictor=predictor,
        learning_rate=args.learning_rate,
        history_discount=args.history_discount,
        seed=args.seed,
        snapshot_interval=args.snapshot_interval,
    )
    for request in phase_a:
        cache.access(request)
    a_result = cache.result()
    weights_at_shift = a_result.final_weights
    for request in phase_b:
        cache.access(request)
    result = cache.result()

    phase_b_hits = result.hits - a_result.hits
    fixed_phase_b = {
        "lru": fixed_policy_phase_b_hit_ratio(phase_a, phase_b, args.capacity, ("lru",), predictor, args),
        "lfu": fixed_policy_phase_b_hit_ratio(phase_a, phase_b, args.capacity, ("lfu",), predictor, args),
        "nn": fixed_policy_phase_b_hit_ratio(phase_a, phase_b, args.capacity, ("nn",), predictor, args),
    }
    nn_wrong = fixed_phase_b["nn"] < max(fixed_phase_b["lru"], fixed_phase_b["lfu"])
    nn_weight_decreased = result.final_weights["nn"] < weights_at_shift["nn"]
    summary = {
        "capacity": args.capacity,
        "phase_a_requests": len(phase_a),
        "phase_a_hit_ratio": a_result.hit_ratio,
        "phase_b_requests": len(phase_b),
        "phase_b_hit_ratio": phase_b_hits / len(phase_b),
        "fixed_expert_phase_b_hit_ratios": fixed_phase_b,
        "nn_wrong_on_phase_b": nn_wrong,
        "weights_at_shift": weights_at_shift,
        "final_weights": result.final_weights,
        "nn_weight_change": result.final_weights["nn"] - weights_at_shift["nn"],
        "nn_weight_decreased": nn_weight_decreased,
        "fallback_observed": nn_wrong and nn_weight_decreased,
    }

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = prefix.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    timeline_path = prefix.with_name(prefix.name + "_weights.csv")
    snapshots = result.snapshots
    with timeline_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=snapshots[0].keys())
        writer.writeheader()
        writer.writerows(snapshots)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path} and {timeline_path}")


if __name__ == "__main__":
    main()
