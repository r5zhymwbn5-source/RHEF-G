# RHEF-G

Research artifacts for *Reliability-Weighted Heterogeneous Evidence Fusion for Phage-Host Genus Prediction*, a genus-level prediction method using DNA composition and protein-derived features. The test set was used repeatedly during development. The component-bootstrap interval for the main comparison crosses zero, and the available external cohorts do not establish independent validation or broad superiority. Predictions do not demonstrate infection.

## Contents

- `analysis_replay/`: saved model predictions, per-sample evidence, source snapshots and code for reproducing the principal Q2 comparison.
- `checkpoints/architecture_revision/Q2-canonical30/`: twelve frozen expert checkpoints. Only load PyTorch checkpoint files from trusted sources.
- `features/DatasetG/architecture_revision/Q2-canonical30/`: fitted scaler and token PCA.
- `features/DatasetG/class_info.json`: class index mapping.
- `checkpoints/harp_revision/splits/Q2-canonical30/split.npz`: row-index split; requires the original dataset and row order, which are not included.
- `SHA256SUMS`: hashes of the staged files; see the note on manifest scope below.

## Replay the principal analysis

Use a compatible Python installation with NumPy and scikit-learn. The analysis environment is recorded in `analysis_replay/environment.json`.

```sh
python analysis_replay/replay_primary.py
```

The script reconstructs the fusion and DeepSets baseline from saved predictions and reproduces the paired 2,000-draw bootstrap. This is an analysis replay, not a complete reconstruction from raw genomes.

## Limitations and dependencies

The bundle does not contain the complete source genomes or CDS datasets, full preprocessing inputs, all Q3/external analyses, or a turnkey inference pipeline. ESM-2 embeddings used the independently distributed [facebook/esm2_t33_650M_UR50D](https://huggingface.co/facebook/esm2_t33_650M_UR50D) model. That model and its rights are governed by its own terms and are not included here. Files derived from other public sources require their own license and attribution checks. The source files supplied here are incomplete snapshots; full training cannot be reproduced from these files alone.

## License and release status

The author-owned code and trained RHEF-G expert weights in this repository are made available under the MIT License; see `LICENSE`. Third-party models, software and datasets retain their own licenses and are not relicensed here. Users are responsible for observing those terms and independently verifying research artifacts.

## Integrity

`SHA256SUMS` records every repository file except the manifest itself. `analysis_replay/SHA256SUMS` separately covers the files in that folder as originally staged.
