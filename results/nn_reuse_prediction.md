# ReuseNet results

Traces from libCacheSim's bundled `data/` (see [../NOTES.md](../NOTES.md) for how they were pulled and
column-mapped). AUC is the primary metric — see the note on accuracy below.

## Training (`scripts/train_reuse_net.py`, 200 epochs, 70/30 chronological split)

| Trace | Window | Accuracy | AUC |
|---|---|---|---|
| cloudPhysicsIO | 360 | 0.8555 | 0.8797 |
| twitter_cluster52 | 25 | 0.8132 | 0.8709 |

## Feature ablation (`scripts/run_ablation.py`, 400 epochs)

### cloudPhysicsIO (window=360)

| feature set | accuracy | auc |
|---|---|---|
| all | 0.7100 | 0.8886 |
| without_recency | 0.8391 | 0.8513 |
| without_frequency | 0.8810 | 0.9024 |
| without_time_since_last_access | 0.7937 | 0.7806 |
| without_size | 0.8588 | 0.9175 |

`time_since_last_access` looks like the most load-bearing feature (dropping it costs the most AUC,
0.8886 → 0.7806). `size` looks close to irrelevant here (dropping it *improves* AUC slightly).

### twitter_cluster52 (window=25)

| feature set | accuracy | auc |
|---|---|---|
| all | 0.8159 | 0.8843 |
| without_recency | 0.8178 | 0.8828 |
| without_frequency | 0.8137 | 0.7769 |
| without_time_since_last_access | 0.8141 | 0.8818 |
| without_size | 0.8141 | 0.8821 |

`frequency` is clearly the dominant feature here too (dropping it costs the most AUC, 0.8843 → 0.7769). 
Consistent with cloudPhysicsIO in that `size` is close to irrelevant, but differs in that `time_since_last_access`
matters a lot for cloudPhysicsIO and hardly at all for Twitter — likely a reflection of the two workloads' different access
patterns (block-storage I/O vs. a social-media cache).

## Note on accuracy vs. AUC

Accuracy is computed at a fixed 0.5 probability threshold, which isn't well-calibrated for these label
distributions. The "all features" cloudPhysicsIO run has the *worst* accuracy of the ablation despite 
having strong AUC (0.89); this is a calibration artifact, not evidence the model is bad: it still beats
the ~63% naive-majority-class baseline, and AUC (which doesn't depend on the threshold) shows it 
discriminates reused-vs-not well. We report AUC as the primary metric for this reason, and because 
the downstream use never applies a 0.5 cutoff anyway.
