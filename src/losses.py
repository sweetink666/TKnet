import torch
import torch.nn.functional as F


def dual_modal_contrastive_loss(
    query_ms,
    support_ms,
    query_pan,
    support_pan,
    prototypes_ms,
    prototypes_pan,
    query_labels,
    support_labels,
    importance_ms,
    importance_pan,
    margin=0.3,
    eps=1e-8,
):
    query_labels = query_labels.reshape(-1)
    support_labels = support_labels.reshape(-1)
    regions = query_ms.shape[-1]
    classes = prototypes_ms.shape[0]
    support_count = support_ms.shape[0]

    proto_sim_ms = F.cosine_similarity(query_ms[:, None], prototypes_ms[None], dim=2)
    proto_sim_pan = F.cosine_similarity(query_pan[:, None], prototypes_pan[None], dim=2)
    support_sim_ms = F.cosine_similarity(query_ms[:, None], support_ms[None], dim=2)
    support_sim_pan = F.cosine_similarity(query_pan[:, None], support_pan[None], dim=2)

    gather_labels = query_labels[:, None, None].expand(-1, 1, regions)
    positive_proto_ms = proto_sim_ms.gather(1, gather_labels).squeeze(1)
    positive_proto_pan = proto_sim_pan.gather(1, gather_labels).squeeze(1)

    same_class = (
        query_labels[:, None, None].expand(-1, support_count, regions)
        == support_labels[None, :, None].expand(-1, -1, regions)
    )
    positive_support_ms = support_sim_ms.masked_fill(~same_class, -torch.inf).max(dim=1).values
    positive_support_pan = support_sim_pan.masked_fill(~same_class, -torch.inf).max(dim=1).values
    positive_ms = 0.5 * (positive_proto_ms + positive_support_ms)
    positive_pan = 0.5 * (positive_proto_pan + positive_support_pan)

    class_ids = torch.arange(classes, device=query_labels.device)
    negative_mask = (query_labels[:, None] != class_ids[None, :])[:, :, None]
    negative_mask = negative_mask.expand(-1, -1, regions)
    negative_ms = proto_sim_ms.masked_fill(~negative_mask, -torch.inf).max(dim=1).values
    negative_pan = proto_sim_pan.masked_fill(~negative_mask, -torch.inf).max(dim=1).values

    weights = (importance_ms * importance_pan).clamp(0, 1)
    weight_sum = weights.sum()
    if weight_sum < 1:
        return query_ms.sum() * 0

    loss_ms = F.relu(margin + negative_ms - positive_ms)
    loss_pan = F.relu(margin + negative_pan - positive_pan)
    return ((loss_ms * weights).sum() + (loss_pan * weights).sum()) / (
        2 * weight_sum + eps
    )
