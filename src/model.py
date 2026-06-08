import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.segmentation import slic

from .attention import cross_modal_attention


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, inputs):
        output = F.relu(self.bn1(self.conv1(inputs)), inplace=True)
        output = self.bn2(self.conv2(output))
        return self.skip(inputs) + output


class DualEncoder(nn.Module):
    def __init__(self, output_dim=512, slic_segments=110, slic_compactness=10.0):
        super().__init__()
        self.slic_segments = slic_segments
        self.slic_compactness = slic_compactness
        self.ms_encoder = self._make_encoder(4, output_dim)
        self.pan_encoder = self._make_encoder(1, output_dim)

    @staticmethod
    def _make_encoder(channels, output_dim):
        return nn.Sequential(
            nn.Conv2d(channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128),
            ResidualBlock(128, 256),
            ResidualBlock(256, output_dim),
        )

    def forward(self, ms, pan):
        pan = F.interpolate(pan, size=ms.shape[-2:], mode="bilinear", align_corners=False)
        ms_features = self.ms_encoder(ms)
        pan_features = self.pan_encoder(pan)
        masks = self._slic_masks(torch.cat([pan, ms], dim=1))
        return (
            superpixel_pool(ms_features, masks),
            superpixel_pool(pan_features, masks),
        )

    def _slic_masks(self, joint):
        masks = []
        for sample in joint.detach().cpu():
            pan = sample[0].numpy()
            ms_mean = sample[1:5].mean(dim=0).numpy()
            grayscale = 0.7 * pan + 0.3 * ms_mean
            segments = slic(
                grayscale,
                n_segments=self.slic_segments,
                compactness=self.slic_compactness,
                channel_axis=None,
                start_label=0,
            )
            masks.append(torch.from_numpy(np.asarray(segments, dtype=np.int64)))
        return torch.stack(masks)


def superpixel_pool(features, masks):
    batch, channels, height, width = features.shape
    masks = masks.to(features.device).reshape(batch, -1)
    flattened = features.reshape(batch, channels, height * width)
    regions = int(masks.max().item()) + 1

    pooled = torch.zeros(batch, channels, regions, device=features.device)
    counts = torch.zeros(batch, regions, device=features.device)
    pooled.scatter_add_(2, masks[:, None, :].expand(-1, channels, -1), flattened)
    counts.scatter_add_(1, masks, torch.ones_like(masks, dtype=features.dtype))
    return pooled / counts.clamp_min(1)[:, None, :]


class PrototypeNetwork(nn.Module):
    def __init__(self, categories, slic_segments=110, slic_compactness=10.0):
        super().__init__()
        self.categories = categories
        self.encoder = DualEncoder(
            slic_segments=slic_segments,
            slic_compactness=slic_compactness,
        )

    def extract_regions(self, ms, pan):
        if ms.ndim == 3:
            ms = ms.unsqueeze(0)
            pan = pan.unsqueeze(0)
        return self.encoder(ms, pan)

    def forward(self, support_ms, support_pan, query_ms, query_pan, support_labels):
        batch, samples, channels, height, width = support_ms.shape
        support_ms = support_ms.reshape(batch * samples, channels, height, width)
        support_pan = support_pan.reshape(
            batch * samples, support_pan.shape[2], support_pan.shape[3], support_pan.shape[4]
        )
        query_ms = query_ms.reshape(-1, query_ms.shape[2], query_ms.shape[3], query_ms.shape[4])
        query_pan = query_pan.reshape(-1, query_pan.shape[2], query_pan.shape[3], query_pan.shape[4])

        support_features_ms, support_features_pan = self.extract_regions(support_ms, support_pan)
        query_features_ms, query_features_pan = self.extract_regions(query_ms, query_pan)
        prototypes_ms = self.compute_prototypes(support_features_ms, support_labels)
        prototypes_pan = self.compute_prototypes(support_features_pan, support_labels)
        return (
            support_features_ms,
            support_features_pan,
            query_features_ms,
            query_features_pan,
            prototypes_ms,
            prototypes_pan,
        )

    def compute_prototypes(self, features, labels):
        labels = labels.reshape(-1)
        prototypes = []
        for class_id in range(self.categories):
            class_features = features[labels == class_id]
            if len(class_features):
                prototypes.append(class_features.mean(dim=0))
            else:
                prototypes.append(torch.zeros_like(features[0]))
        return torch.stack(prototypes)

    @staticmethod
    def fuse(features_ms, features_pan, mode="attention", topk_ratio=0.25):
        if mode == "sum":
            return features_ms + features_pan, None, None
        if mode != "attention":
            raise ValueError(f"Unsupported fusion mode: {mode}")

        similarity, mask_ms, mask_pan = cross_modal_attention(
            features_ms, features_pan, topk_ratio
        )
        scale = 10.0
        attention_ms = F.softmax(similarity * scale, dim=-1)
        attention_pan = F.softmax(similarity.transpose(1, 2) * scale, dim=-1)
        enhanced_ms = torch.bmm(
            attention_ms, features_pan.transpose(1, 2)
        ).transpose(1, 2)
        enhanced_pan = torch.bmm(
            attention_pan, features_ms.transpose(1, 2)
        ).transpose(1, 2)
        fused = (features_ms + enhanced_ms) + (features_pan + enhanced_pan)
        return fused, mask_ms, mask_pan

    def encode(self, ms, pan, fusion_mode="attention", topk_ratio=0.25):
        features_ms, features_pan = self.extract_regions(ms, pan)
        fused, _, _ = self.fuse(features_ms, features_pan, fusion_mode, topk_ratio)
        return F.normalize(fused.mean(dim=-1), p=2, dim=1)

