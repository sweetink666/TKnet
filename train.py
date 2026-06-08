import argparse
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.config import load_config, save_config
from src.data import load_dataset
from src.datasets import EpisodicDataset, PatchDataset
from src.engine import build_feature_bank, evaluate_knn, train_one_epoch
from src.model import PrototypeNetwork
from src.utils import (
    checkpoint_payload,
    resolve_device,
    save_json,
    set_seed,
    worker_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the TAS/KMA few-shot model.")
    default_config = Path(__file__).resolve().parent / "configs" / "default.json"
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("TAS_DATA_DIR"),
        help="Private dataset directory. It can also be set with TAS_DATA_DIR.",
    )
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--refresh-feature-cache",
        action="store_true",
        help="Recompute the initial feature bank if a cache already exists.",
    )
    return parser.parse_args()


def make_loader(dataset, batch_size, workers, shuffle=False, generator=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_seed,
        generator=generator,
        persistent_workers=workers > 0,
    )


def run_training(args):
    if not args.data_dir:
        raise SystemExit(
            "A private dataset path is required. Use --data-dir /path/to/data "
            "or set TAS_DATA_DIR."
        )

    config = load_config(args.config)
    device = resolve_device(args.device)
    bundle = load_dataset(args.data_dir, patch_size=config["patch_size"])
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    all_results = []

    for run_index in range(config["num_repeats"]):
        run_id = run_index + 1
        seed = config["seed"] + run_index
        set_seed(seed)
        run_dir = output_root / f"run_{run_id:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, run_dir / "config.json")

        all_indices = np.arange(len(bundle.train_labels))
        reference_indices, validation_indices = train_test_split(
            all_indices,
            train_size=config["train_ratio"],
            stratify=bundle.train_labels,
            random_state=seed,
        )
        reference = PatchDataset(
            bundle.ms,
            bundle.pan,
            bundle.train_xy[reference_indices],
            bundle.train_labels[reference_indices],
            patch_size=config["patch_size"],
        )
        validation = PatchDataset(
            bundle.ms,
            bundle.pan,
            bundle.train_xy[validation_indices],
            bundle.train_labels[validation_indices],
            patch_size=config["patch_size"],
        )
        reference_loader = make_loader(
            reference, config["eval_batch_size"], config["num_workers"]
        )
        validation_loader = make_loader(
            validation, config["eval_batch_size"], config["num_workers"]
        )

        model = PrototypeNetwork(
            bundle.categories,
            slic_segments=config["slic_segments"],
            slic_compactness=config["slic_compactness"],
        ).to(device)
        feature_cache = run_dir / "initial_feature_bank.npz"
        if args.refresh_feature_cache and feature_cache.exists():
            feature_cache.unlink()
        feature_bank = build_feature_bank(
            model,
            reference_loader,
            device,
            config["fusion_mode"],
            config["topk_ratio"],
            feature_cache,
        )

        episodic_base = PatchDataset(
            bundle.ms,
            bundle.pan,
            bundle.train_xy[reference_indices],
            bundle.train_labels[reference_indices],
            patch_size=config["patch_size"],
            augment=True,
        )
        episodic = EpisodicDataset(
            episodic_base,
            feature_bank,
            num_way=config["num_way"],
            num_support=config["num_shot_support"],
            num_query=config["num_shot_query"],
            difficulty_ratio=config["difficulty_ratio"],
            support_sampling=config["support_sampling"],
            query_sampling=config["query_sampling"],
            episodes_per_class=config["episodes_per_class"],
            topk_candidates=config["topk_candidates"],
            recent_window=config["recent_window"],
        )
        generator = torch.Generator().manual_seed(seed)
        train_loader = make_loader(
            episodic,
            config["batch_size"],
            config["num_workers"],
            shuffle=True,
            generator=generator,
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["epochs"], eta_min=1e-6
        )
        writer = SummaryWriter(run_dir / "tensorboard")
        best_aa = -1.0
        best_metrics = None

        for epoch in range(config["epochs"]):
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, device, config, epoch
            )
            validation_metrics = evaluate_knn(
                model,
                reference_loader,
                validation_loader,
                device,
                bundle.categories,
                neighbors=config["knn_neighbors"],
                fusion_mode=config["fusion_mode"],
                topk_ratio=config["topk_ratio"],
            )
            scheduler.step()

            print(
                f"Run {run_id} epoch {epoch + 1}: "
                f"loss={train_metrics['loss']:.4f}, "
                f"OA={validation_metrics['oa'] * 100:.2f}%, "
                f"AA={validation_metrics['aa'] * 100:.2f}%"
            )
            for name, value in train_metrics.items():
                writer.add_scalar(f"train/{name}", value, epoch)
            writer.add_scalar("validation/OA", validation_metrics["oa"], epoch)
            writer.add_scalar("validation/AA", validation_metrics["aa"], epoch)
            writer.add_scalar("validation/kappa", validation_metrics["kappa"], epoch)

            if validation_metrics["aa"] > best_aa:
                best_aa = validation_metrics["aa"]
                best_metrics = validation_metrics
                payload = checkpoint_payload(
                    model,
                    config,
                    run_id,
                    seed,
                    bundle.categories,
                    reference_indices,
                    epoch + 1,
                    validation_metrics,
                )
                torch.save(payload, run_dir / "best_model.pt")

        writer.close()
        result = {
            "run_id": run_id,
            "seed": seed,
            "checkpoint": str(run_dir / "best_model.pt"),
            "best_validation": best_metrics,
        }
        save_json(result, run_dir / "summary.json")
        all_results.append(result)

    oa = [item["best_validation"]["oa"] for item in all_results]
    aa = [item["best_validation"]["aa"] for item in all_results]
    summary = {
        "runs": all_results,
        "mean_validation_oa": float(np.mean(oa)),
        "std_validation_oa": float(np.std(oa)),
        "mean_validation_aa": float(np.mean(aa)),
        "std_validation_aa": float(np.std(aa)),
    }
    save_json(summary, output_root / "training_summary.json")
    print(f"Training complete. Results: {output_root / 'training_summary.json'}")


if __name__ == "__main__":
    run_training(parse_args())
