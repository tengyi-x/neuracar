# Cache-size experiment: NeuraCaR vs. baselines (twitter_cluster52)

`NeuraCaR` (internal Python simulator) vs. libCacheSim's native LRU/LFU/ARC/LeCaR/LHD/LRB, at three
cache sizes. `NeuraCaR`'s NN-eviction path costs O(cached objects) per eviction (a forward pass over
every candidate) and doesn't scale to 8M within a reasonable runtime — extrapolating from the 500K→2M
scaling (9x time for 4x capacity) put 8M at roughly 9-10 hours, so it's skipped. 500K and 2M are exact,
full-scan results, not sampled/approximated.

| Cache size | LRU | LFU | ARC | LeCaR | LHD | LRB | **NeuraCaR** |
|---|---|---|---|---|---|---|---|
| 500K | 0.6904 | 0.6378 | 0.6976 | 0.6978 | 0.7215 | 0.7142 | **0.6967** |
| 2M | 0.7737 | 0.7265 | 0.7747 | 0.7775 | 0.7943 | 0.7893 | **0.7746** |
| 8M | 0.8348 | 0.8179 | 0.8368 | 0.8368 | 0.8427 | 0.8357 | _(not run)_ |

## Reading the numbers

At both sizes NeuraCaR lands in the middle: clearly ahead of plain LFU, roughly matching
LRU/ARC/LeCaR, but behind LHD and LRB (the two ML-based baselines). That's a fair, honest result for
a small feedforward net going up against a gradient-boosted tree (LRB) and a hand-tuned statistical
model (LHD) — it validates that the NN captures real signal (it's not just noise, since it beats LFU
and matches LRU/ARC/LeCaR) without claiming it surpasses more sophisticated purpose-built baselines.

Sources:
- LRU/LFU/ARC/LeCaR/LHD/LRB: [twitter_cluster52_libcachesim_baselines.csv](twitter_cluster52_libcachesim_baselines.csv)
  (built and run in Google Colab, see [../docs/colab_libcachesim_baselines.ipynb](../docs/colab_libcachesim_baselines.ipynb))
- NeuraCaR: [cache_experiments_twitter_cluster52.csv](cache_experiments_twitter_cluster52.csv) (internal
  simulator, run locally after the eviction-scan optimization)

## cloudPhysicsIO (block-chunked)

`cloudPhysicsIO` is a block-storage I/O trace where the same block number can appear at different
I/O sizes (overlapping variable-length reads), which violates the simulator's one-size-per-object
assumption — that's why it's excluded from the table above. To still get a NeuraCaR comparison on
this trace, we split each I/O request into fixed 4096-byte block accesses (standard for turning
block I/O into a cache-simulation trace; see [`../scripts/chunk_block_trace.py`](../scripts/chunk_block_trace.py)),
truncating each request's trailing partial block rather than emitting a variable-size remainder. A
new ReuseNet checkpoint was trained on this chunked trace specifically (different granularity, and
`size` is now a constant 4096 for every object, so it carries no signal) — AUC dropped to 0.62 from
0.88 on the original request-level trace, reflecting weaker reuse signal at block granularity.

| Cache size | LRU | LFU | ARC | LeCaR | LHD | LRB | **NeuraCaR** |
|---|---|---|---|---|---|---|---|
| 10M | 0.0175 | 0.0190 | 0.0191 | 0.0176 | 0.0176 | 0.0236 | **0.0181** |

Hit ratios are low across every policy (10M covers only ~0.5% of the trace's ~1.9GB unique working
set), but the *relative* ranking is: NeuraCaR beats LRU/LeCaR/LHD, trails LFU/ARC
slightly, and is clearly behind LRB (the strongest baseline) — the same "middle of the group, real
but modest signal" pattern as `twitter_cluster52`, even on a much weaker training signal (AUC 0.62)
and a workload NeuraCaR was never really designed for (block storage, not an object cache).

Sources:
- LRU/LFU/ARC/LeCaR/LHD/LRB: [cloudPhysicsIO_blocks_libcachesim_baselines.csv](cloudPhysicsIO_blocks_libcachesim_baselines.csv)
  (built and run in Google Colab)
- NeuraCaR: run locally, capacity exactly `10*1024*1024` bytes to match libCacheSim's `"10M"` convention
