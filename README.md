# NeuraCaR: A Neural Expert for Cache Replacement

CS 486 Project

## Overview

NeuraCaR extends [LeCaR](https://www.usenix.org/conference/hotstorage18/presentation/vietri) by adding a small
feedforward neural network as a third "expert" alongside LRU and LFU in a tracking-the-best-expert weight update.
The network predicts, for each cached object, the probability it will be reused before eviction, using recency,
frequency, time since last access, and size as features. At each eviction the system picks LRU, LFU, or the NN by
weight, observes the hit/miss reward, and updates all three weights — so it can learn patterns the heuristics miss
while still falling back to LRU/LFU on unfamiliar workloads.

## Dependencies

- [libCacheSim](https://github.com/cacheMon/libCacheSim) — cache simulator, baselines (LRU, LFU, ARC, LeCaR, LHD, LRB)
- Python 3.10+, PyTorch, NumPy, pandas, matplotlib

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

All Python commands below should use the virtual environment's interpreter (or an activated `.venv`); no
global packages are required.

## Train and export the NN expert

The simulator needs both the trained network and Person A's training-set normalization statistics. Export them
together as a checkpoint:

```bash
python scripts/train_reuse_net.py data/prepared.csv \
  --window 360 --checkpoint models/reuse_net.pt
```

The checkpoint records the feature order, architecture, weights, mean, and standard deviation. NeuraCaR rejects
incompatible checkpoint feature orders instead of silently making incorrect predictions.

## Three-expert simulator

`AdaptiveCache` shares one byte-capacity cache across LRU, LFU, and NN eviction experts. On each required
eviction, it samples an expert from the current weights. The NN scores every cached candidate and evicts the
object with the lowest predicted probability of reuse (equivalently, highest predicted probability of
non-reuse).

As in LeCaR, evicted objects enter a bounded ghost history. If one is requested while still in history, the
responsible expert is penalized by a discounted multiplicative-weights update and all weights are renormalized.
With two experts, this is algebraically equivalent to LeCaR's update that rewards the other expert; the same
penalty formulation extends unambiguously to three experts.

Run the cache-size experiment:

```bash
python scripts/run_cache_experiments.py data/prepared.csv models/reuse_net.pt \
  --cache-sizes 10m 50m 100m \
  --output results/cache_experiments.csv
```

This always runs the internal LRU, LFU, LeCaR, and NeuraCaR policies. To add the proposal's full set of native
baselines (LRU, LFU, ARC, LeCaR, LHD, and LRB), pass the built executable:

```bash
python scripts/run_cache_experiments.py data/prepared.csv models/reuse_net.pt \
  --cache-sizes 10m 50m 100m \
  --libcachesim third_party/libCacheSim/_build/bin/cachesim
```

The main CSV contains request and byte hit ratios. Each cache size also gets a weight-timeline CSV.

## Workload-shift experiment

Train the checkpoint on phase A, then replay phase A followed by an unfamiliar phase B:

```bash
python scripts/run_workload_shift.py data/phase_a.csv data/phase_b.csv models/reuse_net.pt \
  --capacity 104857600 --snapshot-interval 10000 \
  --output-prefix results/workload_shift
```

The JSON summary records phase-specific hit ratios, counterfactual phase-B hit ratios for fixed LRU/LFU/NN
experts, weights at the boundary, final weights, and whether a fallback was observed. The harness only claims
fallback when the fixed NN is worse than at least one heuristic on phase B *and* its adaptive weight decreases.
The timeline CSV is suitable for plotting the three weights over the workload boundary.

Phase-B object IDs are namespaced by default because IDs from independent traces need not refer to the same
objects (and can have different sizes). Pass `--shared-object-ids` only when the two phases came from one logical
object namespace.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The tests cover the two-expert LeCaR equivalence, three-expert normalization and reproducibility, delayed
history penalties, Person A's online feature order, NN victim choice, byte-capacity eviction, and oversize
requests.
