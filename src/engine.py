from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from .losses import dual_modal_contrastive_loss


def extract_features(model, loader, device, fusion_mode="attention", topk_ratio=0.25):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for ms, pan, _, batch_labels in tqdm(loader, desc="Extracting features", leave=False):
            vectors = model.encode(
                ms.to(device),
                pan.to(device),
                fusion_mode=fusion_mode,
                topk_ratio=topk_ratio,
            )
            features.append(vectors.cpu().numpy())
            labels.append(np.asarray(batch_labels))
    return np.concatenate(features), np.concatenate(labels)


def build_feature_bank(model, loader, device, fusion_mode, topk_ratio, cache_path=None):
    if cache_path is not None and Path(cache_path).is_file():
        cached = np.load(cache_path)
        return cached["features"]

    features, _ = extract_features(model, loader, device, fusion_mode, topk_ratio)
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, features=features)
    return features


def _local_labels(labels, episode_classes):
    remapped = torch.empty_like(labels.reshape(-1))
    for local_id, class_id in enumerate(episode_classes):
        remapped[labels.reshape(-1) == class_id] = local_id
    return remapped


def _episode_loss(model, episode, device, config):
    support_ms, support_pan, query_ms, query_pan, support_labels, query_labels = episode
    support_ms = support_ms.unsqueeze(0).to(device)
    support_pan = support_pan.unsqueeze(0).to(device)
    query_ms = query_ms.unsqueeze(0).to(device)
    query_pan = query_pan.unsqueeze(0).to(device)
    support_labels = support_labels.to(device)
    query_labels = query_labels.to(device)

    (
        support_features_ms,
        support_features_pan,
        query_features_ms,
        query_features_pan,
        prototypes_ms,
        prototypes_pan,
    ) = model(support_ms, support_pan, query_ms, query_pan, support_labels)

    query_fused, query_mask_ms, query_mask_pan = model.fuse(
        query_features_ms,
        query_features_pan,
        mode=config["fusion_mode"],
        topk_ratio=config["topk_ratio"],
    )
    support_fused, _, _ = model.fuse(
        support_features_ms,
        support_features_pan,
        mode=config["fusion_mode"],
        topk_ratio=config["topk_ratio"],
    )
    query_vectors = F.normalize(query_fused.mean(dim=-1), p=2, dim=1)
    support_vectors = F.normalize(support_fused.mean(dim=-1), p=2, dim=1)

    episode_classes = torch.unique(support_labels.reshape(-1), sorted=True)
    local_support_labels = _local_labels(support_labels, episode_classes)
    local_query_labels = _local_labels(query_labels, episode_classes)
    vector_prototypes = torch.stack([
        support_vectors[local_support_labels == local_id].mean(dim=0)
        for local_id in range(len(episode_classes))
    ])
    logits = F.cosine_similarity(
        query_vectors[:, None, :], vector_prototypes[None, :, :], dim=2
    ) / config["temperature"]
    classification_loss = F.cross_entropy(logits, local_query_labels)

    contrastive_loss = query_vectors.sum() * 0
    if config["use_contrastive"]:
        if query_mask_ms is None:
            regions = query_features_ms.shape[-1]
            query_mask_ms = torch.ones(len(query_features_ms), regions, device=device)
            query_mask_pan = torch.ones_like(query_mask_ms)
        contrastive_loss = dual_modal_contrastive_loss(
            query_features_ms,
            support_features_ms,
            query_features_pan,
            support_features_pan,
            prototypes_ms[episode_classes],
            prototypes_pan[episode_classes],
            local_query_labels,
            local_support_labels,
            query_mask_ms,
            query_mask_pan,
            margin=config["contrastive_margin"],
        )
        weight = config["classification_weight"]
        loss = weight * classification_loss + (1 - weight) * contrastive_loss
    else:
        loss = classification_loss

    predictions = logits.argmax(dim=1)
    correct = int((predictions == local_query_labels).sum().item())
    return loss, classification_loss, contrastive_loss, correct, len(local_query_labels)


def train_one_epoch(model, loader, optimizer, device, config, epoch):
    model.train()
    totals = {"loss": 0.0, "classification": 0.0, "contrastive": 0.0}
    correct = 0
    count = 0

    progress = tqdm(loader, desc=f"Epoch {epoch + 1}", unit="batch")
    for batch in progress:
        optimizer.zero_grad(set_to_none=True)
        batch_losses = []
        for episode_index in range(batch[0].shape[0]):
            episode = tuple(item[episode_index] for item in batch)
            loss, cls_loss, ctr_loss, episode_correct, episode_count = _episode_loss(
                model, episode, device, config
            )
            batch_losses.append(loss)
            totals["classification"] += float(cls_loss.detach())
            totals["contrastive"] += float(ctr_loss.detach())
            correct += episode_correct
            count += episode_count

        loss = torch.stack(batch_losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        totals["loss"] += float(loss.detach()) * len(batch_losses)
        progress.set_postfix(loss=f"{float(loss.detach()):.4f}", acc=f"{100 * correct / count:.2f}%")

    episodes = len(loader.dataset)
    return {
        "loss": totals["loss"] / episodes,
        "classification_loss": totals["classification"] / episodes,
        "contrastive_loss": totals["contrastive"] / episodes,
        "accuracy": correct / count,
    }


def evaluate_knn(
    model,
    reference_loader,
    query_loader,
    device,
    categories,
    neighbors=5,
    fusion_mode="attention",
    topk_ratio=0.25,
):
    reference_features, reference_labels = extract_features(
        model, reference_loader, device, fusion_mode, topk_ratio
    )
    classifier = KNeighborsClassifier(
        n_neighbors=min(neighbors, len(reference_labels)),
        metric="cosine",
    )
    classifier.fit(reference_features, reference_labels)

    predictions, targets = [], []
    model.eval()
    with torch.no_grad():
        for ms, pan, _, labels in tqdm(query_loader, desc="Evaluating", leave=False):
            vectors = model.encode(
                ms.to(device), pan.to(device), fusion_mode, topk_ratio
            ).cpu().numpy()
            predictions.append(classifier.predict(vectors))
            targets.append(np.asarray(labels))

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    class_ids = np.arange(categories)
    matrix = confusion_matrix(targets, predictions, labels=class_ids)
    totals = matrix.sum(axis=1)
    per_class = np.divide(
        matrix.diagonal(),
        totals,
        out=np.zeros(categories, dtype=np.float64),
        where=totals > 0,
    )
    return {
        "oa": float(accuracy_score(targets, predictions)),
        "aa": float(per_class[totals > 0].mean()),
        "kappa": float(cohen_kappa_score(targets, predictions)),
        "per_class_accuracy": per_class.tolist(),
        "confusion_matrix": matrix.tolist(),
    }
