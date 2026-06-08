import json
from copy import deepcopy
from pathlib import Path


DEFAULT_CONFIG = {
    "seed": 42,
    "num_repeats": 3,
    "train_ratio": 0.02,
    "patch_size": 16,
    "batch_size": 8,
    "eval_batch_size": 64,
    "num_workers": 4,
    "epochs": 60,
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "num_way": 5,
    "num_shot_support": 1,
    "num_shot_query": 10,
    "difficulty_ratio": 0.3,
    "support_sampling": "dissimilar",
    "query_sampling": "custom",
    "episodes_per_class": 20,
    "recent_window": 50,
    "topk_candidates": 10,
    "fusion_mode": "attention",
    "topk_ratio": 0.6,
    "temperature": 0.1,
    "use_contrastive": True,
    "classification_weight": 0.0,
    "contrastive_margin": 0.3,
    "knn_neighbors": 5,
    "slic_segments": 110,
    "slic_compactness": 10.0,
}


def load_config(path=None):
    config = deepcopy(DEFAULT_CONFIG)
    if path is None:
        return config

    with Path(path).open("r", encoding="utf-8") as handle:
        user_config = json.load(handle)
    unknown = sorted(set(user_config) - set(DEFAULT_CONFIG))
    if unknown:
        raise ValueError(f"Unknown configuration keys: {', '.join(unknown)}")
    config.update(user_config)
    validate_config(config)
    return config


def validate_config(config):
    if not 0 < config["train_ratio"] < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if config["num_way"] < 2:
        raise ValueError("num_way must be at least 2.")
    if config["num_shot_support"] < 1 or config["num_shot_query"] < 1:
        raise ValueError("Support and query sizes must be positive.")
    if not 0 <= config["difficulty_ratio"] <= 1:
        raise ValueError("difficulty_ratio must be between 0 and 1.")
    if not 0 < config["topk_ratio"] <= 1:
        raise ValueError("topk_ratio must be in (0, 1].")
    if not 0 <= config["classification_weight"] <= 1:
        raise ValueError("classification_weight must be between 0 and 1.")


def save_config(config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

