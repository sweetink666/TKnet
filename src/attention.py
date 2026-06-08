import torch
import torch.nn.functional as F


def cross_modal_attention(features_ms, features_pan, topk_ratio=0.25):
    """Return cross-modal similarity and per-modality top-k importance masks."""
    batch_size, _, regions = features_ms.shape
    ms = F.normalize(features_ms, p=2, dim=1, eps=1e-6).transpose(1, 2)
    pan = F.normalize(features_pan, p=2, dim=1, eps=1e-6).transpose(1, 2)
    similarity = torch.bmm(ms, pan.transpose(1, 2))

    score_ms = similarity.mean(dim=2)
    score_pan = similarity.mean(dim=1)
    k = min(regions, max(1, int(regions * topk_ratio)))

    threshold_ms = torch.topk(score_ms, k, dim=1).values[:, -1].view(batch_size, 1)
    threshold_pan = torch.topk(score_pan, k, dim=1).values[:, -1].view(batch_size, 1)
    mask_ms = (score_ms >= threshold_ms).float()
    mask_pan = (score_pan >= threshold_pan).float()
    return similarity, mask_ms, mask_pan

