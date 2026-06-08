# Dataset placeholder

The datasets used in the paper are not distributed with this repository.

Prepare a dataset directory outside the repository, or place private files in
this directory locally. The expected layout is:

```text
<dataset-root>/
├── ms4.tif
├── pan.tif
├── train.npy
└── test.npy
```

File requirements:

- `ms4.tif`: multispectral image with shape `[H, W, 4]`.
- `pan.tif`: panchromatic image with shape `[4H, 4W]`.
- `train.npy`: training split mask with shape `[H, W]`.
- `test.npy`: test split mask with shape `[H, W]`.
- Split masks use `0` for background and `1..C` for class labels.
- Class labels must be contiguous.

Pass the directory at runtime:

```bash
python train.py --data-dir /path/to/dataset
```

Alternatively, set the environment variable:

```bash
export TAS_DATA_DIR=/path/to/dataset
```

The `.gitignore` rules prevent private data files in `data/` from being added
to Git.

