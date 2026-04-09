import copy
import torch


def new_visual_forward(self, *args, **kwargs):
    # 1) 先拿原始 visual output 物件
    original_output = self.original_forward(*args, **kwargs)

    # 2) Qwen3-VL 後面會用到 pooler_output
    vision_tokens = original_output.pooler_output

    if not torch.is_tensor(vision_tokens):
        raise RuntimeError("original_output.pooler_output is not a tensor")

    # 3) 取得 grid 資訊
    grid_thw = kwargs.get("image_grid_thw", kwargs.get("grid_thw", None))
    if grid_thw is None:
        raise RuntimeError("Cannot find image_grid_thw/grid_thw in visual.forward kwargs")

    if not torch.is_tensor(grid_thw):
        grid_thw = torch.tensor(grid_thw, device=vision_tokens.device)

    # -------------------------------------------------
    # Case A: 2D pooler_output  [sum(N_i_merged), D]
    # -------------------------------------------------
    if vision_tokens.dim() == 2:
        merge_size = getattr(self, "spatial_merge_size", 2)

        lengths = [
            int(
                grid_thw[i, 0]
                * (grid_thw[i, 1] // merge_size)
                * (grid_thw[i, 2] // merge_size)
            )
            for i in range(grid_thw.shape[0])
        ]

        if sum(lengths) != vision_tokens.shape[0]:
            raise RuntimeError(
                f"merged lengths sum {sum(lengths)} != pooler_output.shape[0] {vision_tokens.shape[0]}"
            )

        chunks = torch.split(vision_tokens, lengths, dim=0)   # list of [Ni, D]

        # ---------------------------------------------------------
        # 跨幀 CMRM：把同一張車牌的所有幀 token 合在一起過 CMRM
        #
        # 動機：HR 與 LR 並非 pair，單幀的 slot 只能看到單幀的資訊，
        # 無法在 LR-HR 不對齊的情況下找到跨幀互補特徵。
        # 把同一車牌全部幀的 token 一起餵進 CMRM，slots 可以
        # 同時 attend 到所有幀，學到「這張車牌整體長什麼樣」的表示。
        #
        # 做法：
        #   1. 依 _num_views_per_plate 將 chunks 分組（每組 = 一張車牌的所有幀）
        #   2. 每組 concat → [1, sum_V_N, D] → CMRM → shared plate slots
        #   3. 把 refined tokens 切回各幀，slots 對同車牌所有幀共用
        #
        # _num_views_per_plate 由 model.py 的 train_step / evaluate_plate
        # 在每次 forward 前設定，預設 1（退化為 per-image）。
        # ---------------------------------------------------------
        num_views = getattr(self, "_num_views_per_plate", 1)
        num_images = len(lengths)
        # 若無法整除（例如最後一個 batch 較小），fallback to num_images
        if num_images % num_views != 0:
            num_views = 1
        num_plates = num_images // num_views

        refined_chunks = []
        feat_list = []
        slots_for_cache = []   # 每張圖對應一個 [1, K, D]，最後 cat 成 [B*V, K, D]

        for plate_idx in range(num_plates):
            # 取出這張車牌的所有幀 chunks
            plate_view_chunks = chunks[plate_idx * num_views : (plate_idx + 1) * num_views]

            # 合併成一個大序列：[1, sum_V_N, D]
            plate_tokens_cat = torch.cat(plate_view_chunks, dim=0).unsqueeze(0)

            # CMRM：slots 跨幀 attend，得到整車牌的表示
            x_refined_plate, _, plate_slots = self.cmrm(plate_tokens_cat, return_slots=True)
            # plate_slots: [1, K, D]  — 這張車牌所有幀的共同表示
            # x_refined_plate: [1, sum_V_N, D]

            # 切回各幀
            view_lengths = [c.shape[0] for c in plate_view_chunks]
            refined_views = torch.split(x_refined_plate.squeeze(0), view_lengths, dim=0)

            for frame_refined in refined_views:
                refined_chunks.append(frame_refined)               # [Ni, D]
                feat_list.append(frame_refined.mean(dim=0))        # [D]
                slots_for_cache.append(plate_slots)                # [1, K, D]（同車牌共用）

        feat = torch.stack(feat_list, dim=0)              # [B*V, D]
        x_refined_2d = torch.cat(refined_chunks, dim=0)  # [sum(Ni), D]
        slots_refined = torch.cat(slots_for_cache, dim=0) # [B*V, K, D]

        self._cmrm_cache = {
            "vision_tokens": vision_tokens,
            "slots": slots_refined,   # [B*V, K, D]  (同車牌各幀共用同一 slots)
            "feat": feat,             # [B*V, D]
            "lengths": lengths,
        }

        output = copy.copy(original_output)
        output.pooler_output = x_refined_2d
        return output

    # -------------------------------------------------
    # Case B: 3D pooler_output  [B, N, D]
    # -------------------------------------------------
    elif vision_tokens.dim() == 3:
        x_refined, attn_weights, slots_refined = self.cmrm(
            vision_tokens,
            return_slots=True
        )
        feat = x_refined.mean(dim=1)

        self._cmrm_cache = {
            "vision_tokens": vision_tokens,
            "slots": slots_refined,
            "feat": feat,
            "x_refined": x_refined,
            "attn_weights": attn_weights,
        }

        output = copy.copy(original_output)
        output.pooler_output = x_refined
        return output

    else:
        raise RuntimeError(f"Unexpected pooler_output dim: {vision_tokens.dim()}")
