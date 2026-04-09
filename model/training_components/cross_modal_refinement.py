import torch
import torch.nn as nn


class CMRM(nn.Module):
    """
    Cross-Modal Refinement Module with Gated Residual.

    梯度路徑設計（解耦）：
      - slot_loss  ← cache["slots"] = ungated slots
                     訓練 cross-attention 學出好的 slot 表示
                     (LR slot ↔ HR slot 的 L1，與 gate 無關，避免 loss=0 死鎖)
      - feat_loss  ← cache["feat"] = x_refined.mean()
                     = (x32 + gate * slots.mean()).mean()
                     訓練 gate (alpha)「要不要打開」
      - reg_loss   ← 只訓練 LoRA

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

        # ungated slots：cross-attention 的真實輸出
        # slot_loss 用這個訓練 cross-attn，不依賴 gate 所以不會消失
        slots = slots + slots_out                          # (B, K, dim)

        gate = torch.tanh(self.alpha)
        g = (gate * slots).mean(dim=1, keepdim=True)      # gate 只在 residual

        x_refined = x32 + g                               # gate=0 → x_refined=x32
        x_refined = x_refined.to(orig_dtype)

        if return_slots:
            # 回傳 ungated slots → slot_loss 的梯度訓練 cross-attn
            # feat_loss 透過 x_refined → gate 路徑訓練 alpha
            return x_refined, attn_weights, slots
        return x_refined, attn_weights
