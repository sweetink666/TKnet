import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    def __init__(self, ms, pan, xy, labels, patch_size=16, augment=False):
        self.ms = ms
        self.pan = pan
        self.xy = np.asarray(xy, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.patch_size = patch_size
        self.pan_patch_size = patch_size * 4
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def _safe_start(coord, length, size):
        return min(max(0, int(coord)), length - size)

    def get_pair(self, index):
        row, col = self.xy[index]
        row_ms = self._safe_start(row, self.ms.shape[1], self.patch_size)
        col_ms = self._safe_start(col, self.ms.shape[2], self.patch_size)
        row_pan = self._safe_start(row * 4, self.pan.shape[1], self.pan_patch_size)
        col_pan = self._safe_start(col * 4, self.pan.shape[2], self.pan_patch_size)

        ms = self.ms[:, row_ms:row_ms + self.patch_size,
                     col_ms:col_ms + self.patch_size].clone()
        pan = self.pan[:, row_pan:row_pan + self.pan_patch_size,
                      col_pan:col_pan + self.pan_patch_size].clone()
        if self.augment:
            ms, pan = self._augment_pair(ms, pan)
        return ms, pan

    @staticmethod
    def _augment_pair(ms, pan):
        if random.random() < 0.5:
            ms = torch.flip(ms, dims=(2,))
            pan = torch.flip(pan, dims=(2,))
        if random.random() < 0.5:
            ms = torch.flip(ms, dims=(1,))
            pan = torch.flip(pan, dims=(1,))
        rotations = random.randrange(4)
        if rotations:
            ms = torch.rot90(ms, rotations, dims=(1, 2))
            pan = torch.rot90(pan, rotations, dims=(1, 2))
        return ms, pan

    def __getitem__(self, index):
        ms, pan = self.get_pair(index)
        return ms, pan, index, int(self.labels[index])


class EpisodicDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        features,
        num_way,
        num_support,
        num_query,
        difficulty_ratio=0.3,
        support_sampling="dissimilar",
        query_sampling="custom",
        episodes_per_class=20,
        topk_candidates=10,
        recent_window=50,
    ):
        self.base = base_dataset
        self.features = torch.as_tensor(features, dtype=torch.float32)
        if len(self.features) != len(self.base):
            raise ValueError("The feature bank must align with the base dataset.")

        self.num_way = num_way
        self.num_support = num_support
        self.num_query = num_query
        self.difficulty_ratio = difficulty_ratio
        self.support_sampling = support_sampling
        self.query_sampling = query_sampling
        self.topk_candidates = topk_candidates
        self.recent_window = recent_window
        self.recent = OrderedDict()

        self.labels = self.base.labels
        self.classes = np.unique(self.labels)
        self.class_indices = {
            int(label): np.where(self.labels == label)[0].tolist()
            for label in self.classes
        }
        required = num_support + num_query
        self.valid_classes = [
            label for label in self.classes
            if len(self.class_indices[int(label)]) >= required
        ]
        if len(self.valid_classes) < num_way:
            raise ValueError(
                f"Only {len(self.valid_classes)} classes have at least {required} samples; "
                f"{num_way} are required."
            )
        self.total_episodes = len(self.valid_classes) * episodes_per_class

    def __len__(self):
        return self.total_episodes

    def _select_support(self, local_features):
        count = len(local_features)
        if self.support_sampling == "random":
            return random.sample(range(count), self.num_support)

        center = local_features.mean(dim=0, keepdim=True)
        similarity = F.cosine_similarity(local_features, center.expand_as(local_features), dim=1)
        largest = self.support_sampling == "similar"
        if self.support_sampling not in {"similar", "dissimilar"}:
            raise ValueError(f"Unknown support sampling strategy: {self.support_sampling}")
        return torch.topk(similarity, self.num_support, largest=largest).indices.tolist()

    def _select_queries(self, local_features, dataset_indices, support_local):
        remaining = [i for i in range(len(dataset_indices)) if i not in support_local]
        if self.query_sampling == "random":
            return random.sample(remaining, self.num_query)
        if self.query_sampling != "custom":
            raise ValueError(f"Unknown query sampling strategy: {self.query_sampling}")

        available = [i for i in remaining if dataset_indices[i] not in self.recent]
        if len(available) < self.num_query:
            available = remaining

        hard_count = int(self.num_query * self.difficulty_ratio)
        random_count = self.num_query - hard_count
        selected = []
        if hard_count:
            prototype = local_features[support_local].mean(dim=0, keepdim=True)
            distances = 1 - F.cosine_similarity(
                local_features[available], prototype.expand(len(available), -1), dim=1
            )
            candidate_count = min(self.topk_candidates, len(available))
            candidate_positions = torch.topk(distances, candidate_count).indices.tolist()
            candidates = [available[position] for position in candidate_positions]
            selected = random.sample(candidates, min(hard_count, len(candidates)))
            for local_index in selected:
                key = dataset_indices[local_index]
                self.recent[key] = None
                self.recent.move_to_end(key, last=False)
                if len(self.recent) > self.recent_window:
                    self.recent.popitem(last=True)

        remaining_after_hard = [i for i in available if i not in selected]
        fill_count = self.num_query - len(selected)
        if len(remaining_after_hard) < fill_count:
            remaining_after_hard = [i for i in remaining if i not in selected]
        selected.extend(random.sample(remaining_after_hard, fill_count))
        return selected

    def __getitem__(self, _):
        selected_classes = np.random.choice(self.valid_classes, self.num_way, replace=False)
        support_ms, support_pan, support_labels = [], [], []
        query_ms, query_pan, query_labels = [], [], []

        for label in selected_classes:
            dataset_indices = self.class_indices[int(label)].copy()
            random.shuffle(dataset_indices)
            local_features = self.features[dataset_indices]
            support_local = self._select_support(local_features)
            query_local = self._select_queries(local_features, dataset_indices, support_local)

            for local_index in support_local:
                ms, pan = self.base.get_pair(dataset_indices[local_index])
                support_ms.append(ms)
                support_pan.append(pan)
                support_labels.append(int(label))
            for local_index in query_local:
                ms, pan = self.base.get_pair(dataset_indices[local_index])
                query_ms.append(ms)
                query_pan.append(pan)
                query_labels.append(int(label))

        return (
            torch.stack(support_ms),
            torch.stack(support_pan),
            torch.stack(query_ms),
            torch.stack(query_pan),
            torch.tensor(support_labels, dtype=torch.long),
            torch.tensor(query_labels, dtype=torch.long),
        )
