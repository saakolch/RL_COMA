# Final results summary

## Table 1 - Final performance (mean +/- std over 3 seeds)

| Method | Success rate | Deliveries /3 | Episode return | Gini |
|---|---|---|---|---|
| shared | 0.413 +/- 0.038 | 2.15 +/- 0.09 | 4.57 +/- 0.54 | 0.071 +/- 0.019 |
| difference | 0.030 +/- 0.030 | 0.78 +/- 0.14 | -5.58 +/- 0.98 | 0.171 +/- 0.050 |
| coma | 0.103 +/- 0.153 | 1.12 +/- 0.64 | -3.47 +/- 4.51 | 0.242 +/- 0.095 |

## Table 2 - Learning speed (mean +/- std over 3 seeds)

| Method | AULC (success) | Steps to 50% | Steps to 90% of best | Wall-clock per run (s) |
|---|---|---|---|---|
| shared | 0.176 +/- 0.042 | 528526 +/- 13706 | 450945 +/- 50847 | 5664 |
| difference | 0.009 +/- 0.007 | not reached | not reached | 6392 |
| coma | 0.017 +/- 0.024 | not reached | not reached | 5848 |

## Table 3 - Per-seed final metrics

| Mode | Seed | Success | Deliveries | Return | Gini | AULC | Steps to 50% |
|---|---|---|---|---|---|---|---|
| shared | 0 | 0.37 | 2.20 | 4.33 | 0.067 | 0.210 | 518834 |
| shared | 1 | 0.44 | 2.20 | 5.19 | 0.092 | 0.190 | 538217 |
| shared | 2 | 0.43 | 2.05 | 4.19 | 0.053 | 0.129 | not reached |
| difference | 0 | 0.06 | 0.89 | -4.58 | 0.229 | 0.009 | not reached |
| difference | 1 | 0.00 | 0.63 | -6.55 | 0.135 | 0.003 | not reached |
| difference | 2 | 0.03 | 0.83 | -5.60 | 0.149 | 0.016 | not reached |
| coma | 0 | 0.28 | 1.82 | 1.61 | 0.172 | 0.045 | not reached |
| coma | 1 | 0.01 | 0.58 | -7.01 | 0.203 | 0.003 | not reached |
| coma | 2 | 0.02 | 0.95 | -5.00 | 0.350 | 0.003 | not reached |

Notes: shared reaches 50% in 2 of 3 seeds; COMA and difference never reach 50% or 90% of the best final success within 340 iterations (~680k steps).