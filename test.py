import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import load_dataset
from src.datasets import PatchDataset
from src.engine import evaluate_knn
from src.model import PrototypeNetwork
from src.utils import resolve_device, save_json, set_seed, worker_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained TAS/KMA checkpoints.")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("TAS_DATA_DIR"),
        help="Private dataset directory. It can also be set with TAS_DATA_DIR.",
    )
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", default="test_results.json")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    return parser.parse_args()


def make_loader(dataset, batch_size, workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_seed,
        persistent_workers=workers > 0,
    )


def run_test(args):
    if not args.data_dir:
        raise SystemExit(
            "A private dataset path is required. Use --data-dir /path/to/data "
            "or set TAS_DATA_DIR."
        )

    device = resolve_device(args.device)
    checkpoints = [Path(path).expanduser().resolve() for path in args.checkpoints]
    missing = [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint files not found: {', '.join(missing)}")

    results = []
    bundle = None
    loaded_patch_size = None
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        set_seed(int(checkpoint["seed"]))
        if bundle is None:
            bundle = load_dataset(args.data_dir, patch_size=config["patch_size"])
            loaded_patch_size = config["patch_size"]
        elif config["patch_size"] != loaded_patch_size:
            raise ValueError("All checkpoints must use the same patch size.")

        if checkpoint["categories"] != bundle.categories:
            raise ValueError(
                f"Checkpoint expects {checkpoint['categories']} classes, "
                f"but the dataset contains {bundle.categories}."
            )
        reference_indices = np.asarray(checkpoint["reference_indices"], dtype=np.int64)
        reference = PatchDataset(
            bundle.ms,
            bundle.pan,
            bundle.train_xy[reference_indices],
            bundle.train_labels[reference_indices],
            patch_size=config["patch_size"],
        )
        test_set = PatchDataset(
            bundle.ms,
            bundle.pan,
            bundle.test_xy,
            bundle.test_labels,
            patch_size=config["patch_size"],
        )
        reference_loader = make_loader(
            reference, config["eval_batch_size"], config["num_workers"]
        )
        test_loader = make_loader(
            test_set, config["eval_batch_size"], config["num_workers"]
        )

        model = PrototypeNetwork(
            bundle.categories,
            slic_segments=config["slic_segments"],
            slic_compactness=config["slic_compactness"],
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        metrics = evaluate_knn(
            model,
            reference_loader,
            test_loader,
            device,
            bundle.categories,
            neighbors=config["knn_neighbors"],
            fusion_mode=config["fusion_mode"],
            topk_ratio=config["topk_ratio"],
        )
        results.append(
            {
                "checkpoint": str(checkpoint_path),
                "run_id": checkpoint["run_id"],
                "seed": checkpoint["seed"],
                "metrics": metrics,
            }
        )
        print(
            f"{checkpoint_path.name}: OA={metrics['oa'] * 100:.2f}%, "
            f"AA={metrics['aa'] * 100:.2f}%, Kappa={metrics['kappa']:.4f}"
        )

    oa = [item["metrics"]["oa"] for item in results]
    aa = [item["metrics"]["aa"] for item in results]
    kappa = [item["metrics"]["kappa"] for item in results]
    summary = {
        "runs": results,
        "mean_oa": float(np.mean(oa)),
        "std_oa": float(np.std(oa)),
        "mean_aa": float(np.mean(aa)),
        "std_aa": float(np.std(aa)),
        "mean_kappa": float(np.mean(kappa)),
        "std_kappa": float(np.std(kappa)),
    }
    save_json(summary, args.output)
    print(f"Test results saved to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    run_test(parse_args())
