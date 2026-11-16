# Round 2: controlled real-subset audit

64 object points + 4 controller points, 80 training frames, 35 withheld future frames, 80 epochs, 16 integration substeps. Each variant uses seeds 0/1/2 and the same observations. Reference-loss trials use corrected transition masks from round2_aligned; original round2 reference-loss trials are superseded. Metrics below are common intersection-mask observation metrics, not official sparse annotation metrics. Error bars are sample standard deviations.

| Variant (completed / 3) | L0 future track (mm) | L3 future track (mm) | L3 future symmetric CD (mm) |
|---|---:|---:|---:|
| legacy (3/3) | 20.837 ± 6.726 | 40.276 ± 11.884 | 39.597 ± 7.549 |
| loss_only (3/3) | 20.593 ± 7.756 | 34.010 ± 4.071 | 35.374 ± 2.557 |
| physics_only (2/3) | 40.403 ± 7.485 | 30.762 ± 3.108 | 32.746 ± 2.295 |
| schedule_only (3/3) | 20.789 ± 6.686 | 33.865 ± 5.228 | 34.184 ± 3.699 |
| full_global (3/3) | 27.418 ± 0.573 | 33.413 ± 2.308 | 36.431 ± 3.379 |
| full_local (3/3) | 27.225 ± 0.217 | 30.194 ± 0.868 | 27.932 ± 0.582 |
| full_geometry (3/3) | 27.062 ± 0.110 | 27.969 ± 0.224 | 28.511 ± 0.502 |
| full_random (3/3) | 27.127 ± 0.158 | 27.906 ± 0.923 | 26.913 ± 0.216 |

Failed trials (excluded from aggregates above, explicitly retained):
- physics_only_2: epoch 67, nonfinite loss

Groups with fewer than 3 completions are survivor statistics, not comparable three-seed estimates.
