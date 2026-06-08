# TKNet: TAS/KMA Few-Shot Multimodal Classification

This directory contains the training and evaluation code for the proposed
task-aware sampling (TAS) and key-region-aware multimodal aggregation (KMA)
method. Visualization, plotting, and ablation-only scripts are intentionally
excluded from this public release.

## Repository structure

```text
.
├── configs/
│   └── default.json
├── data/
│   └── README.md
├── src/
│   ├── attention.py
│   ├── config.py
│   ├── data.py
│   ├── datasets.py
│   ├── engine.py
│   ├── losses.py
│   ├── model.py
│   └── utils.py
├── train.py
├── test.py
└── requirements.txt
```

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the CUDA-enabled PyTorch build appropriate for your system if GPU
training is required.

## Private dataset setup

The datasets used in the paper are not released. Users must provide their own
authorized copy. No local or absolute dataset path is stored in the source
code.

Expected directory:

```text
<dataset-root>/
├── ms4.tif       # [H, W, 4]
├── pan.tif       # [4H, 4W]
├── train.npy     # [H, W]
└── test.npy      # [H, W]
```

The split masks must use `0` as background and `1..C` as contiguous class
labels. See [data/README.md](data/README.md) for details.

Provide the path using either:

```bash
export TAS_DATA_DIR=/path/to/private/dataset
```

or the `--data-dir` argument shown below.

## Training

Run from this directory:

```bash
python train.py \
  --data-dir /path/to/private/dataset \
  --config configs/default.json \
  --output-dir outputs
```

The default configuration reproduces the main `5-way 1-shot 10-query` setup.
Each run writes:

```text
outputs/run_01/
├── best_model.pt
├── config.json
├── initial_feature_bank.npz
├── summary.json
└── tensorboard/
```

`best_model.pt` stores the model parameters, experiment configuration, random
seed, and exact reference-set indices. This allows evaluation to reconstruct
the same reference set without hard-coded paths or file names.

To inspect logs:

```bash
tensorboard --logdir outputs
```

## Evaluation

Evaluate one checkpoint:

```bash
python test.py \
  --data-dir /path/to/private/dataset \
  --checkpoints outputs/run_01/best_model.pt \
  --output test_results.json
```

Evaluate multiple repeated runs:

```bash
python test.py \
  --data-dir /path/to/private/dataset \
  --checkpoints \
    outputs/run_01/best_model.pt \
    outputs/run_02/best_model.pt \
    outputs/run_03/best_model.pt
```

The report contains OA, AA, Cohen's kappa, per-class accuracy, and the
confusion matrix for each run, together with mean and standard deviation.

## Configuration

All public experiment options are defined in `configs/default.json`. Important
entries include:

- `train_ratio`: fraction of the official training split used as the reference
  and episodic training pool.
- `difficulty_ratio`: fraction of TAS queries selected as hard queries.
- `support_sampling`: `dissimilar`, `similar`, or `random`.
- `query_sampling`: `custom` for TAS or `random`.
- `topk_ratio`: KMA key-region selection ratio.
- `classification_weight`: weight of classification loss; the contrastive loss
  receives `1 - classification_weight`.
- `num_repeats`: number of independent training runs.

## Notes

- SLIC superpixel generation runs on CPU, as required by scikit-image.
- The initial feature bank is generated once per run and cached locally.
- Data, feature caches, checkpoints, and experiment outputs are ignored by Git.
- This release contains no visualization or ablation plotting code.
