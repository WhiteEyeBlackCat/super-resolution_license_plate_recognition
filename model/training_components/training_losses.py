import torch
import torch.nn.functional as F


def multi_view_consistency_loss(view_features: torch.Tensor) -> torch.Tensor:
    """
    view_features: [B, V, D]

    Lmvc = sum_i ||z_i - z_bar||_1
    We optimize the mean L1 distance across batch, view, and feature dims.
    """
    view_center = view_features.mean(dim=1, keepdim=True)
    return F.l1_loss(view_features, view_center.expand_as(view_features))
