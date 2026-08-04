# NeuraCaR: A Neural Expert for Cache Replacement

NeuraCaR is a CS 486 team project that extends
[LeCaR](https://www.usenix.org/conference/hotstorage18/presentation/vietri) with a trained neural-network
eviction expert. The system tracks three experts - LRU, LFU, and NN - and adapts their selection weights from
delayed eviction feedback.

The original project description and evaluation commitments are preserved in the
[project proposal](docs/CS486_Project_Proposal.pdf).

## Research question

Can a small neural network learn reuse patterns that LRU and LFU miss while an online
tracking-the-best-expert policy safely shifts weight back to those heuristics when the network encounters an
unfamiliar workload?

NeuraCaR combines two learning settings:

- **Supervised learning:** predict whether a cached object will be reused within a fixed future window.
- **Online learning / RL-style adaptation:** select an eviction expert using multiplicative weights and penalize
  experts whose recent evictions cause avoidable misses.

## Design

The neural network receives four per-object features computed from trace replay:

| Feature | Meaning |
|---|---|
| Recency | Number of requests since the object's previous access |
| Frequency | Number of prior accesses to the object |
| Time since last access | Trace-time difference from the previous access |
| Size | Object size from the trace |

It outputs `P(reuse)`. When the NN expert is selected, NeuraCaR evicts the cached object with the lowest
predicted probability of reuse, equivalently the highest probability of non-reuse.

LRU, LFU, and NN share one byte-capacity cache. Evicted objects are placed in policy-specific ghost histories.
If an object is requested while still in history, the expert responsible for its eviction receives a discounted
multiplicative-weights penalty, and all three expert weights are renormalized.

## Team responsibilities

### Supervised-learning side

- [x] Build libCacheSim and confirm LRU, LFU, ARC, LeCaR, LHD, and LRB baselines.
- [x] Extract recency, frequency, time-since-last-access, and size features.
- [x] Generate fixed-window binary reuse labels.
- [x] Implement the two-hidden-layer feedforward network and chronological training/test split.
- [x] Run feature-ablation experiments.

Recorded NN and ablation results are in [results/nn_reuse_prediction.md](results/nn_reuse_prediction.md), and
libCacheSim build/trace notes are in [NOTES.md](NOTES.md).

### Adaptive-policy side

- [x] Generalize LeCaR's update from two experts to LRU, LFU, and NN.
- [x] Connect normalized NN predictions to eviction decisions.
- [x] Implement cache-size and baseline experiment harnesses.
- [x] Implement workload-shift measurement with fixed-expert counterfactuals and weight timelines.
- [x] Add deterministic unit and synthetic end-to-end tests.
- [ ] Run the final experiments on 2-3 real traces and retain the result artifacts.

## Repository layout

```text
docs/                         Project proposal
results/                      Recorded metrics and experiment outputs
scripts/                      Trace preparation, training, ablation, and simulation CLIs
src/neuracar/                 Features, labels, model, training, inference, and simulator code
src/experts/                  Tracking-the-best-expert policy
tests/                        Policy, simulator, checkpoint, and adapter tests
```

## Environment setup

Use a repository-local virtual environment. Do not install project dependencies into global Python.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

All following commands assume `.venv` is activated. Calling `.venv/bin/python` directly is equivalent.

## Prepare a trace

The Python pipeline expects headerless `time,obj_id,size` CSV rows. Reorder another CSV with:

```bash
python scripts/prepare_trace.py INPUT.csv data/prepared.csv \
  --time-col 0 --obj-id-col 1 --size-col 2 --has-header
```

For a small local smoke test:

```bash
python scripts/gen_synthetic_trace.py data/synthetic.csv \
  --n-requests 20000 --n-objects 500
```

## Train and export the reuse model

The simulator needs both the trained network and the training-set normalization statistics. Export them in one
checkpoint:

```bash
python scripts/train_reuse_net.py data/prepared.csv \
  --window 360 --train-frac 0.7 --epochs 200 \
  --checkpoint models/reuse_net.pt
```

The checkpoint records feature order, architecture, model weights, mean, and standard deviation. NeuraCaR
rejects incompatible feature orders instead of silently producing incorrect predictions.

Run the drop-one-feature ablation:

```bash
python scripts/run_ablation.py data/prepared.csv \
  --window 360 --train-frac 0.7 --epochs 400
```

## Run the adaptive experiments

Compare internal LRU, LFU, LeCaR, and NeuraCaR across cache sizes:

```bash
python scripts/run_cache_experiments.py data/prepared.csv models/reuse_net.pt \
  --cache-sizes 10m 50m 100m \
  --output results/cache_experiments.csv
```

Add the proposal's full native baseline set by providing the built libCacheSim executable:

```bash
python scripts/run_cache_experiments.py data/prepared.csv models/reuse_net.pt \
  --cache-sizes 10m 50m 100m \
  --libcachesim third_party/libCacheSim/_build/bin/cachesim \
  --output results/cache_experiments.csv
```

The main CSV contains request and byte hit ratios. Each cache size also gets a NeuraCaR weight-timeline CSV.

Measure adaptation across a workload boundary:

```bash
python scripts/run_workload_shift.py data/phase_a.csv data/phase_b.csv models/reuse_net.pt \
  --capacity 104857600 --snapshot-interval 10000 \
  --output-prefix results/workload_shift
```

The workload-shift harness only reports fallback when both conditions hold:

1. the fixed NN expert performs worse than fixed LRU or LFU during phase B; and
2. the adaptive NN weight decreases after the workload boundary.

Phase-B object IDs are namespaced by default because independent traces may reuse IDs for unrelated objects.
Pass `--shared-object-ids` only when both phases use the same logical object namespace.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
```

The test suite covers two-expert LeCaR equivalence, three-expert sampling, delayed history penalties, byte
capacity, online feature semantics, NN victim choice, LFU reinsertion, oversize objects, checkpoint round trips,
and libCacheSim output parsing.

## Current status and next milestone

The supervised-learning pipeline, adaptive three-expert implementation, documentation, and synthetic
end-to-end checks are complete. Ten unit tests pass.

The next project milestone is experimental rather than architectural:

1. export trained checkpoints for 2-3 selected real traces;
2. run NeuraCaR at multiple cache sizes;
3. collect native LRU, LFU, ARC, LeCaR, LHD, and LRB results;
4. run the workload-shift experiment and plot the three expert weights;
5. retain CSV/JSON outputs and report both hit ratio and adaptation behavior.

Synthetic results are implementation checks only and should not be presented as final project evidence.
