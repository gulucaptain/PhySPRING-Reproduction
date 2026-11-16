# Publication and reproducibility

This is an exported public repository. The original local research directory remains unchanged. README is the current overview; ITERATIONS.md preserves the detailed development narrative and historical commands. Checkpoint-dependent commands in that narrative require regenerating the excluded binaries.

## Included and excluded

Included: original code/tests, source snapshots, per-run configuration and scalar summaries, training traces, failure records, diagnostic JSON, aggregate CSV, figures, and data/source provenance. Local workspace prefixes in published text were replaced by relative placeholders. `sources/export_transformations.json` records original and exported digests for changed files.

Excluded: virtual environment, caches, all `.pt`/`.npz`/`.pkl` files, raw data, downloaded papers and third-party source files. `sources/manifest.json` describes research copies, including excluded files; `sources/urls.json` locates their origins. `sources/published_artifacts.json` verifies the actual distributed result/log/config files and archived code.

## Three distinct verification levels

1. **Archive verification, no data needed:** `python scripts/verify_archive.py` checks exported hashes, full training traces, final metrics, failure records and the round-3 CSV/aggregate against per-run summaries. It does not rerun simulation or prove equivalence to the paper.
2. **Table/figure reconstruction, no data needed:** `MPLCONFIGDIR=/tmp/physpring-mpl python scripts/report_round3.py` rebuilds tables and figures from the shipped summaries and force-diagnostic JSON. Different plotting versions can change image bytes; `verify_archive.py --skip-hashes` still checks numerical summaries after such regeneration.
3. **Training and checkpoint replay:** prepare data, use fresh `--output-root` locations, then run `python -m physpring.evaluate_v3 <new-run-folder> --substeps 16 32 64 667`. This loads regenerated checkpoints and data. Historical checkpoint-verification scripts target the archived directory layout and cache their prior results; they are not a fresh-clone archive test.

`run_round2.py` and `run_round3.py` support alternate output roots in this publication. Default historical locations are nonempty and will be skipped to protect evidence. The published runner adds output-path selection but does not change training algorithms. Source snapshots retain the actual code used for the archived experiments.

To reproduce checkpoint-dependent audits, regenerate runs into a separate clone with the matching `results/round2_aligned` and `results/round3` layouts (move the shipped scalar directories elsewhere first). Then use `audit_axial_reduction.py`, `audit_projection_gradients.py`, `audit_map_parameter_swap.py`, `audit_multilevel_gradients.py` and `evaluate_sparse_gt_v3.py`. Do not mix newly trained tensors with unrelated archived scalar records.

CPU tests and archive verification run in GitHub Actions. Linux CI does not rerun MPS experiments. The recorded macOS dependency lock is provided for provenance; portable installation uses project dependencies. No paper benchmark or hardware-equivalent result is inferred from passing CI.
