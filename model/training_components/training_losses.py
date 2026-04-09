import torch
import torch.nn.functional as F

def cmrm_losses(lr_slots, lr_feat, hr_slots, hr_feat):
        """
        lr_slots: [B, V, K, D]
        lr_feat : [B, V, D]
        hr_slots: [B, V, K, D]
        hr_feat : [B, V, D]
        """
        with torch.no_grad():
            hr_slot_teacher = hr_slots.mean(dim=1, keepdim=True)  # [B,1,K,D]
            hr_feat_teacher = hr_feat.mean(dim=1, keepdim=True)   # [B,1,D]

        slot_loss = F.l1_loss(lr_slots, hr_slot_teacher.expand_as(lr_slots))
        feat_loss = F.l1_loss(lr_feat, hr_feat_teacher.expand_as(lr_feat))
        return slot_loss, feat_loss