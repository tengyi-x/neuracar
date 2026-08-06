# Synthetic performance-improvement review

These measurements are controlled implementation evidence, not replacements for the project's real-trace
results. The repository does not currently contain the Twitter or cloudPhysics trace/checkpoint artifacts.

## Setup

- Trace: 20,000 requests, 500 objects, Zipf parameter 1.2, trace seed 486.
- Cache capacities: 64 KiB and 256 KiB.
- Simulator seeds: 1, 7, 42, 86, and 486.
- Training: 120 epochs and a chronological 70/30 split with boundary-censored labels.
- Baseline configuration: identity features, window 100, uniform initial weights, full-cache NN scan.
- Log configuration: `log1p` features, window 100, uniform initial weights, full-cache NN scan.
- Capacity-tuned configuration: `log1p` features, window 500, 32 guarded candidates, and initial weights
  LRU/LFU/NN = 0.45/0.40/0.15.

## Request hit ratio

| Configuration | 64 KiB mean (sd) | 256 KiB mean (sd) |
|---|---:|---:|
| Original | 0.503940 (0.001004) | 0.668100 (0.000511) |
| `log1p` only | 0.516890 (0.000797) | **0.676880 (0.000360)** |
| Capacity-tuned | **0.529180 (0.000725)** | 0.672170 (0.000419) |
| LRU | 0.448450 | 0.659450 |
| LFU | **0.549050** | **0.703800** |
| LeCaR | 0.463000 | 0.662150 |

`log1p` is the only change that improved both capacities: +0.01295 at 64 KiB and +0.00878 at 256 KiB
relative to the original configuration. The longer reuse window and guarded candidate set improved the smaller
cache further, but regressed against `log1p` alone at 256 KiB. For that reason, candidate count, reuse window,
and initial weights remain validation-tuned controls rather than universal defaults.

The stationary Zipf workload strongly favors LFU, and NeuraCaR still trails it by 0.01987 at 64 KiB and
0.02692 at 256 KiB even after choosing the best measured NeuraCaR configuration per capacity. Real-trace
evaluation is required before changing the reported project conclusions.
