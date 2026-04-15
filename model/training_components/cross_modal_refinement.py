import torch
import torch.nn as nn


class CMRM(nn.Module):
    """
    Cross-Modal Refinement Module with Gated Residual.

    現在主要提供跨幀融合後的 plate-level visual features：
      - cache["feat"] 會被上層拿去做 multi-view consistency loss
      - reg_loss 則由語言模型直接做 LR 文字重建

    初始化保證：
      alpha=0 → gate=tanh(0)=0 → g=0 → x_refined=x32（LM 完全不受影響）
    """

    def __init__(self, num_slots=7, dim=4096, num_heads=8):
        super().__init__()
        self.slots = nn.Parameter(1e-4 * torch.randn(num_slots, dim))
        self.pre_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )
        # gate：初始 0 → CMRM 初始對 LM 完全無影響
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, return_slots=False):
        """
        x: (B, N, dim)  — original visual tokens
        return: (B, N, dim) — gated refinement
        """
        orig_dtype = x.dtype
        B = x.size(0)

        x32 = x.float()
        x_normed = self.pre_norm(x32)

        slots = self.slots.unsqueeze(0).expand(B, -1, -1).contiguous().float()
        slots = torch.clamp(slots, -10.0, 10.0)

        slots_out, attn_weights = self.cross_attn(
            query=slots,
            key=x_normed,
            value=x_normed,
        )

        slots = slots + slots_out                          # (B, K, dim)

        gate = torch.tanh(self.alpha)
        g = (gate * slots).mean(dim=1, keepdim=True)      # gate 只在 residual

        x_refined = x32 + g                               # gate=0 → x_refined=x32
        x_refined = x_refined.to(orig_dtype)

        if return_slots:
            return x_refined, attn_weights, slots
        return x_refined, attn_weights
