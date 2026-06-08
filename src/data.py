from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch


REQUIRED_FILES = ("ms4.tif", "pan.tif", "train.npy", "test.npy")


@dataclass
class DatasetBundle:
    ms: torch.Tensor
    pan: torch.Tensor
    train_xy: np.ndarray
    train_labels: np.ndarray
    test_xy: np.ndarray
    test_labels: np.ndarray
    categories: int


def _validate_data_dir(data_dir):
    data_dir = Path(data_dir).expanduser().resolve()
    missing = [name for name in REQUIRED_FILES if not (data_dir / name).is_file()]
    if missing:
        expected = "\n".join(f"  - {data_dir / name}" for name in REQUIRED_FILES)
        raise FileNotFoundError(
            f"Dataset files are not distributed with this repository.\n"
            f"Missing: {', '.join(missing)}\nExpected layout:\n{expected}"
        )
    return data_dir


def _normalize_image(image):
    image = image.astype(np.float32, copy=False)
    minimum = float(image.min())
    maximum = float(image.max())
    if maximum <= minimum:
        return np.zeros_like(image, dtype=np.float32)
    return (image - minimum) / (maximum - minimum)


def _read_split(mask, background_value=0):
    mask = np.asarray(mask)
    valid = mask != background_value
    xy = np.column_stack(np.where(valid)).astype(np.int64)
    labels = mask[valid].astype(np.int64) - 1
    if labels.size == 0:
        raise ValueError("A split mask contains no labeled samples.")
    if labels.min() < 0:
        raise ValueError("Class labels must be positive integers; 0 is reserved for background.")
    return xy, labels


def load_dataset(data_dir, patch_size=16):
    data_dir = _validate_data_dir(data_dir)
    print(f"Loading private dataset from: {data_dir}")

    ms = tifffile.imread(data_dir / "ms4.tif")
    pan = tifffile.imread(data_dir / "pan.tif")
    train_mask = np.load(data_dir / "train.npy")
    test_mask = np.load(data_dir / "test.npy")

    if ms.ndim != 3 or ms.shape[-1] != 4:
        raise ValueError(f"ms4.tif must have shape [H, W, 4], got {ms.shape}.")
    if pan.ndim == 3 and pan.shape[-1] == 1:
        pan = pan[..., 0]
    if pan.ndim != 2:
        raise ValueError(f"pan.tif must have shape [H, W], got {pan.shape}.")
    if train_mask.shape != ms.shape[:2] or test_mask.shape != ms.shape[:2]:
        raise ValueError("train.npy and test.npy must match the MS spatial dimensions.")

    train_xy, train_labels = _read_split(train_mask)
    test_xy, test_labels = _read_split(test_mask)
    all_labels = np.concatenate([train_labels, test_labels])
    categories = int(all_labels.max()) + 1
    if set(np.unique(all_labels)) != set(range(categories)):
        raise ValueError("Labels must be contiguous and start at 1 in the split masks.")

    ms_pad = (patch_size // 2 - 1, patch_size // 2)
    pan_patch = patch_size * 4
    pan_pad = (pan_patch // 2 - 4, pan_patch // 2)
    border = cv2.BORDER_REFLECT_101

    ms = cv2.copyMakeBorder(ms, ms_pad[0], ms_pad[1], ms_pad[0], ms_pad[1], border)
    pan = cv2.copyMakeBorder(pan, pan_pad[0], pan_pad[1], pan_pad[0], pan_pad[1], border)
    ms = torch.from_numpy(_normalize_image(ms).transpose(2, 0, 1)).float()
    pan = torch.from_numpy(_normalize_image(pan)[None, ...]).float()

    return DatasetBundle(
        ms=ms,
        pan=pan,
        train_xy=train_xy,
        train_labels=train_labels,
        test_xy=test_xy,
        test_labels=test_labels,
        categories=categories,
    )
