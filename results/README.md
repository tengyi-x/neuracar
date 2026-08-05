# Results index

## NN reuse prediction (`nn_reuse_prediction.md`)
ReuseNet trained/evaluated on both real traces
(`cloudPhysicsIO`, `twitter_cluster52`), and feature ablation on both.

## Cache-size comparison (`cache_experiments_summary.md`, `cache_experiments_twitter_cluster52.csv`,
`twitter_cluster52_libcachesim_baselines.csv`, `cloudPhysicsIO_libcachesim_baselines.csv`)
NeuraCaR vs. LRU/LFU/ARC/LeCaR/LHD/LRB, only on `twitter_cluster52` — `cloudPhysicsIO` is a
block-storage I/O trace where the same logical block number can appear at different I/O sizes
(overlapping variable-length reads), which violates the Python simulator's "each cached object has
one fixed size" assumption. So `cloudPhysicsIO` only has libCacheSim baseline numbers (no NeuraCaR
column) :(

## Workload shift (`workload_shift_local.json`, `workload_shift_local_weights.csv`)
Tests whether NeuraCaR's expert weights react to an unfamiliar workload: runs `twitter_cluster52`
(phase A) then switches to a synthetically-generated Zipf trace (phase B, a deliberate distribution
shift). Real finding, not a clean "fallback" story: the NN's weight actually *increased* (17.8% ->
54.8%) after the shift, even though the NN was marginally worse than LRU on phase B alone. Worth
reporting as-is rather than as a confirmed fallback mechanism.
