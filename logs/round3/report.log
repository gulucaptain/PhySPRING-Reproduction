# Round 3 controlled results

Single real cloth subset; 64 object + 4 controller nodes; seeds 0/1/2; 80 updates; 16 substeps; future common-mask Euclidean observation error. Mean ± sample SD. Not the complete paper benchmark.

| Variant | L0 future (mm) | L3 future (mm) |
|---|---:|---:|
| full_global | 26.916 ± 0.242 | 34.239 ± 1.569 |
| full_local | 26.860 ± 0.374 | 27.083 ± 0.792 |
| full_geometry | 27.179 ± 0.027 | 28.214 ± 0.115 |
| full_random | 27.096 ± 0.200 | 27.630 ± 0.203 |
| active_global | 26.488 ± 0.484 | 33.798 ± 4.785 |
| active_local | 27.837 ± 2.642 | 28.249 ± 2.028 |

## Internal-force reduction diagnostics

Frozen ROUND-2 maps, common initial fine-system coefficients, 32 substeps. No parameter fitting. RMS relative to the common fine rollout is distinct from Euclidean error against observations. Oracle RMS uses future data only for hindsight representation analysis.

| Frozen map | Tangent error | Oracle RMS (mm) | Fine-rollout RMS: centroid / bundle / matched (mm) | Observation mean: centroid / bundle / matched (mm) |
|---|---:|---:|---|---|
| full_global | 57.0% | 34.48 | 71.43 / 63.48 / 65.84 | 45.82 / 66.53 / 38.37 |
| full_local | 42.2% | 23.37 | 30.48 / 32.51 / 27.81 | 23.44 / 27.91 / 25.53 |
| full_geometry | 27.5% | 21.07 | 30.23 / 36.85 / 27.52 | 22.12 / 29.10 / 24.77 |
| full_random | 36.9% | 21.67 | 25.00 / 30.32 / 22.79 | 25.44 / 28.67 / 27.64 |
