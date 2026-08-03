"""Generates a small synthetic Zipf-distributed trace for smoke-testing the feature/label/training
pipeline before real traces (from libCacheSim) are available."""

import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_path")
    parser.add_argument("--n-requests", type=int, default=20000)
    parser.add_argument("--n-objects", type=int, default=500)
    parser.add_argument("--zipf-a", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    obj_ids = rng.zipf(args.zipf_a, size=args.n_requests) % args.n_objects
    sizes = rng.integers(1, 4096, size=args.n_objects)

    with open(args.out_path, "w") as f:
        for t, obj_id in enumerate(obj_ids):
            f.write(f"{t},{obj_id},{sizes[obj_id]}\n")

    print(f"Wrote {args.n_requests} requests over {args.n_objects} objects to {args.out_path}")


if __name__ == "__main__":
    main()
