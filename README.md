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