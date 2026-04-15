import json
import types
from pathlib import Path
from collections import defaultdict

import torch
from unsloth import FastVisionModel
from transformers import get_linear_schedule_with_warmup
from transformers import AutoProcessor

from peft import PeftModel

from model.training_components.cross_modal_refinement import CMRM
from model.training_components.visual_forward_patch import new_visual_forward
from model.training_components.plate_metrics import extract_plate, plate_score, vote_plate
from model.training_components.training_losses import multi_view_consistency_loss
from model.training_components.plate_track_dataset import build_track_dataset, split_track_dataset, build_dataloader

class LPLLM:
    def __init__(self, model_path, num_slots=7, cmrm_dim=1152, num_heads=8, use_cmrm=True, lora_path=None, train_LoRA=False, train_cmrm=False):
        self.path = str(model_path)
        self.processor = AutoProcessor.from_pretrained(self.path)

        model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name=self.path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )
        if train_LoRA:
            model = FastVisionModel.get_peft_model(
                model,
                finetune_vision_layers=True,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=16,
                lora_alpha=16,
                lora_dropout=0.0,
                bias="none",
                random_state=3407,
                use_rslora=False,
            )


        self.model = model
        self.device = next(self.model.parameters()).device
        self.train_LoRA = train_LoRA
        self.train_cmrm = train_cmrm

        if lora_path is not None:
            self.load_lora_adapter(lora_path)
        self.use_cmrm = use_cmrm
        if self.use_cmrm:
            self.attach_cmrm(num_slots=num_slots, dims=cmrm_dim, num_heads=num_heads)
        else:
            self.cmrm = None


    def _get_visual_module(self):
        obj = self.model
        for _ in range(6):
            if obj is None:
                break
            if hasattr(obj, "visual"):
                return obj.visual
            if hasattr(obj, "model"):
                obj = obj.model
            elif hasattr(obj, "base_model"):
                obj = obj.base_model
            else:
                obj = None
        raise AttributeError("Could not locate visual module.")

    # -----------------------------------------------------
    # model / patching
    # -----------------------------------------------------
    def attach_cmrm(self, num_slots=7, dims=1152, num_heads=8):
        if self.use_cmrm:
            self.cmrm = CMRM(num_slots=num_slots, dim=dims, num_heads=num_heads).to(
                device=self.device, dtype=torch.float32
            )

            visual = self._get_visual_module()
            visual.cmrm = self.cmrm

            if not hasattr(visual, "original_forward"):
                visual.original_forward = visual.forward

            visual.forward = types.MethodType(new_visual_forward, visual)

            print("CMRM attached:", hasattr(visual, "cmrm"))
            print("Patched forward:", visual.forward.__func__.__name__)



    def build_track_dataset(self, root_dir, num_frames=5, categories=("brazilian", "mercosur")):
        return build_track_dataset(
            root_dir=root_dir,
            num_frames=num_frames,
            categories=categories,
        )

    def split_track_dataset(self, dataset, train_ratio=0.8, seed=42):
        return split_track_dataset(
            dataset,
            train_ratio=train_ratio,
            seed=seed,
        )

    def build_dataloader(self, dataset, batch_size=2, shuffle=True):
        return build_dataloader(
            dataset,
            processor=self.processor,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    # -----------------------------------------------------
    # LoRA
    # -----------------------------------------------------
    def load_lora_adapter(self, lora_path):
        self.model = PeftModel.from_pretrained(self.model, str(lora_path))
        self.device = next(self.model.parameters()).device
        print(f"LoRA loaded from: {lora_path}")


    def freeze_params(self):
        for p in self.model.parameters():
            p.requires_grad = False

        if self.cmrm is not None:
            for p in self.cmrm.parameters():
                p.requires_grad = False

        # case 1: full model baseline
        if (not self.train_LoRA) and (not self.train_cmrm):
            for p in self.model.parameters():
                p.requires_grad = True
            return

        # case 2: LoRA
        if self.train_LoRA:
            for name, p in self.model.named_parameters():
                if "lora_" in name:
                    p.requires_grad = True

        # case 3: CMRM
        if self.train_cmrm and self.cmrm is not None:
            for p in self.cmrm.parameters():
                p.requires_grad = True




    def _get_visual_cache(self):
        visual = self._get_visual_module()
        if not hasattr(visual, "_cmrm_cache"):
            raise RuntimeError("visual._cmrm_cache not found. Check monkey patch.")
        return visual._cmrm_cache

    @staticmethod
    def _reshape_track(x, batch_size, num_views):
        return x.view(batch_size, num_views, *x.shape[1:])

    def _move_to_device(self, inputs):
        return {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in inputs.items()
        }

    def _set_num_views(self, num_views: int):
        """在每次 forward 前告訴 patches.py 每張車牌有幾幀，讓 CMRM 跨幀 attend。"""
        if self.use_cmrm:
            self._get_visual_module()._num_views_per_plate = num_views


    def train_step(self, batch, lambda_reg=1.0, lambda_mvc=0.1):
        batch_size = len(batch["plate_ids"])
        num_views = batch["num_views"][0]
        if any(v != num_views for v in batch["num_views"]):
            raise ValueError("Current implementation expects equal num_views within a batch.")

        self._set_num_views(num_views)   # 讓 CMRM 知道要跨幾幀做 attend
        reg_inputs = self._move_to_device(batch["reg_inputs"])
        outputs = self.model(**reg_inputs)
        reg_loss = outputs.loss

        # ===== baseline branch =====
        if not self.use_cmrm:
            total_loss = reg_loss
            loss_dict = {
                "reg": float(reg_loss.detach().item()),
                "mvc": 0.0,
                "total": float(total_loss.detach().item()),
            }
            return total_loss, loss_dict

        # ===== CMRM + multi-view consistency branch =====
        lr_cache = self._get_visual_cache()
        lr_feat = self._reshape_track(lr_cache["feat"], batch_size, num_views)
        mvc_loss = multi_view_consistency_loss(lr_feat)

        if torch.isnan(reg_loss) or torch.isinf(reg_loss):
            raise RuntimeError(f"reg_loss invalid: {reg_loss}")

        if torch.isnan(mvc_loss) or torch.isinf(mvc_loss):
            raise RuntimeError(f"mvc_loss invalid: {mvc_loss}")

        if torch.isnan(lr_feat).any() or torch.isinf(lr_feat).any():
            raise RuntimeError("lr_feat contains NaN/Inf")

        total_loss = lambda_reg * reg_loss + lambda_mvc * mvc_loss

        loss_dict = {
            "reg": float(reg_loss.detach().item()),
            "mvc": float(mvc_loss.detach().item()),
            "total": float(total_loss.detach().item()),
        }
        return total_loss, loss_dict

    def evaluate_plate(self, val_loader, max_new_tokens=8, debug_seq=5, max_batches=None):
        self.model.eval()
        preds_by_plate = defaultdict(list)
        gt_by_plate = {}

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                lr_images_flat = batch["lr_images_flat"]
                labels = batch["text_labels"]
                plate_ids = batch["plate_ids"]
                num_views = batch["num_views"]

                flat_plate_ids = []
                flat_gt = []
                for plate_id, gt, v in zip(plate_ids, labels, num_views):
                    flat_plate_ids.extend([plate_id] * v)
                    flat_gt.extend([gt] * v)

                messages = [
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {
                                    "type": "text",
                                    "text": "只輸出車牌，禁止其他文字。如'ABC1234'，但'您的答案是ABC1234'則是非法回答。車牌會是7個字的英文字母及數字排列組合",
                                },
                            ],
                        }
                    ]
                    for _ in lr_images_flat
                ]

                prompt_texts = [
                    self.processor.apply_chat_template(
                        m,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for m in messages
                ]

                model_inputs = self.processor(
                    text=prompt_texts,
                    images=lr_images_flat,
                    padding=True,
                    return_tensors="pt",
                )
                model_inputs = {
                    k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in model_inputs.items()
                }

                # 告訴 CMRM 每張車牌有幾幀（假設同一 batch 內幀數相同）
                self._set_num_views(num_views[0] if num_views else 1)

                outputs = self.model.generate(
                    **model_inputs,
                    max_new_tokens=8,
                    do_sample=False,
                    num_beams=1,
                )
                input_len = model_inputs["input_ids"].shape[1]
                gen_tokens = outputs[:, input_len:]
                pred_texts = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

                for pred_text, gt, plate_id in zip(pred_texts, flat_gt, flat_plate_ids):
                    pred_plate = extract_plate(pred_text, target_len=len(gt))
                    preds_by_plate[plate_id].append(pred_plate)
                    if plate_id not in gt_by_plate:
                        gt_by_plate[plate_id] = gt

        self.model.train()

        total_score = 0
        exact_match = 0
        char_correct = 0
        char_total = 0
        valid_count = 0
        shown = 0

        for plate_id, preds in preds_by_plate.items():
            gt = gt_by_plate[plate_id]
            voted_pred = vote_plate(preds, target_len=len(gt), ignore_chars={"?"})

            if shown < debug_seq:
                print("=" * 60)
                print("PLATE ID  :", plate_id)
                print("GT        :", gt)
                print("PREDS     :", preds)
                print("VOTED     :", voted_pred)
                shown += 1

            if voted_pred is None:
                continue

            if voted_pred == gt:
                exact_match += 1

            for p, g in zip(voted_pred, gt):
                if p == g:
                    char_correct += 1
                char_total += 1

            s = plate_score(voted_pred, gt)
            if s is not None:
                total_score += s
                valid_count += 1

        n_plate = len(preds_by_plate)
        return {
            "avg_score": total_score / valid_count if valid_count > 0 else None,
            "exact_match": exact_match / n_plate if n_plate > 0 else None,
            "char_acc": char_correct / char_total if char_total > 0 else None,
            "valid_count": valid_count,
            "num_plate": n_plate,
        }
    
    # -----------------------------------------------------
    # Val
    # -----------------------------------------------------
    
    @torch.no_grad()
    def val_step(self, batch, lambda_reg=1.0, lambda_mvc=0.1):
        self.model.eval()

        total_loss, loss_dict = self.train_step(
            batch,
            lambda_reg=lambda_reg,
            lambda_mvc=lambda_mvc,
        )

        return total_loss.detach().item(), loss_dict
    
    @torch.no_grad()
    def validate(self, val_loader, lambda_reg=1.0, lambda_mvc=0.1, max_batches=None):
        was_training = self.model.training
        self.model.eval()

        total_meter = 0.0
        reg_meter = 0.0
        mvc_meter = 0.0
        count = 0

        for batch_idx, batch in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            total_loss, loss_dict = self.val_step(
                batch,
                lambda_reg=lambda_reg,
                lambda_mvc=lambda_mvc,
            )

            total_meter += total_loss
            reg_meter += loss_dict["reg"]
            mvc_meter += loss_dict["mvc"]
            count += 1

        if was_training:
            self.model.train()

        if count == 0:
            return {
                "total": float("nan"),
                "reg": float("nan"),
                "mvc": float("nan"),
            }

        return {
            "total": total_meter / count,
            "reg": reg_meter / count,
            "mvc": mvc_meter / count,
        }

    # -----------------------------------------------------
    # save
    # -----------------------------------------------------
    def save_checkpoint(self, save_dir, step, epoch, optimizer=None, best_val=None):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        ckpt = {
            "step": step,
            "epoch": epoch,
            "best_val": best_val,
        }

        if self.use_cmrm:
            ckpt["cmrm_state_dict"] = self.cmrm.state_dict()
        else:
            ckpt["model_state_dict"] = self.model.state_dict()

        if optimizer is not None:
            ckpt["optimizer_state_dict"] = optimizer.state_dict()
        
        torch.save(ckpt, save_dir / "latest.pt")

        if self.train_LoRA:
            self.model.save_pretrained(save_dir / "lora_adapter")
            self.processor.save_pretrained(save_dir / "lora_adapter")

        

    def save_best(self, save_dir, step, epoch, val_loss, optimizer=None):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        ckpt = {
            "step": step,
            "epoch": epoch,
            "val_loss": val_loss,
        }

        if self.use_cmrm:
            ckpt["cmrm_state_dict"] = self.cmrm.state_dict()
        else:
            ckpt["model_state_dict"] = self.model.state_dict()

        if optimizer is not None:
            ckpt["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(ckpt, save_dir / "best.pt")

        if self.train_LoRA:
            self.model.save_pretrained(save_dir / "lora_best_adapter")
            self.processor.save_pretrained(save_dir / "lora_best_adapter")


    def save_lora(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.processor.save_pretrained(save_dir)


    def train(
        self,
        train_loader,
        val_loader,
        save_dir,
        epochs=3,
        lr=1e-5,
        weight_decay=1e-4,
        lambda_reg=1.0,
        lambda_mvc=0.1,
        grad_clip=1.0,
        grad_accum_steps=1,         # 累積幾個 mini-batch 再 optimizer.step()
        log_every=10,
        val_every=100,
        save_every=200,
        max_val_batches=20,
        eval_plate_every=None,
        max_plate_batches=20,
        plate_max_new_tokens=8,
        plate_debug_seq=2,
        early_stopping_patience=3,
        early_stopping_min_delta=0.0,
    ):
        self.freeze_params()
        self.model.train()

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if self.cmrm is not None:
            trainable_params.extend([p for p in self.cmrm.parameters() if p.requires_grad])

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
        )

        # scheduler 以 optimizer step 數為基準（非 mini-batch 數）
        steps_per_epoch = max(1, len(train_loader) // grad_accum_steps)
        num_training_steps = steps_per_epoch * epochs
        warmup_steps = int(0.05 * num_training_steps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

        if eval_plate_every is None:
            eval_plate_every = val_every

        global_step = 0
        best_val = float("inf")
        no_improve_count = 0
        early_stop = False
        history = []
        history_path = Path(save_dir) / "history.json"
        n_batches = len(train_loader)

        for epoch in range(1, epochs + 1):
            print(f"\n===== Epoch {epoch}/{epochs} =====")
            self.model.train()

            running_total = 0.0
            running_reg = 0.0
            running_mvc = 0.0
            running_count = 0

            for batch_idx, batch in enumerate(train_loader, start=1):
                total_loss, loss_dict = self.train_step(
                    batch,
                    lambda_reg=lambda_reg,
                    lambda_mvc=lambda_mvc,
                )

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    raise RuntimeError(f"Invalid loss at step {global_step}: {total_loss}")

                # ---- 梯度累積：loss 除以累積步數後才 backward ----
                (total_loss / grad_accum_steps).backward()

                cmrm_sq = 0.0
                if self.cmrm is not None:
                    for name, p in self.cmrm.named_parameters():
                        if p.grad is not None:
                            g = p.grad.detach()
                            cmrm_sq += g.norm(2).item() ** 2
                cmrm_grad_norm = cmrm_sq ** 0.5
                if self.cmrm is not None:
                    print(f"[grad] step={global_step} cmrm_grad_norm={cmrm_grad_norm:.6f}")

                bad_grads = []
                for name, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                            bad_grads.append(name)

                if self.cmrm is not None:
                    for name, p in self.cmrm.named_parameters():
                        if p.requires_grad and p.grad is not None:
                            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                                bad_grads.append(f"cmrm.{name}")

                if bad_grads:
                    # NaN/Inf grad：清掉本窗口所有累積梯度，重新開始
                    print(
                        f"[warn] step={global_step} NaN/Inf in {len(bad_grads)} grad(s): "
                        f"{bad_grads[:3]} — skipping accum window"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue

                # 累積 mini-batch 的 loss（用原始值，未除以 accum steps）
                running_total += loss_dict["total"]
                running_reg += loss_dict["reg"]
                running_mvc += loss_dict["mvc"]
                running_count += 1

                # ---- 判斷是否到達 optimizer.step() 時機 ----
                is_last_batch = (batch_idx == n_batches)
                is_update_step = (batch_idx % grad_accum_steps == 0) or is_last_batch

                if not is_update_step:
                    continue  # 繼續累積，不做 optimizer step

                # ---- optimizer step ----
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)

                optimizer.step()

                bad_params = []
                for name, p in self.model.named_parameters():
                    if p.requires_grad and p.data is not None:
                        if torch.isnan(p.data).any() or torch.isinf(p.data).any():
                            bad_params.append(name)

                if self.cmrm is not None:
                    for name, p in self.cmrm.named_parameters():
                        if p.requires_grad and p.data is not None:
                            if torch.isnan(p.data).any() or torch.isinf(p.data).any():
                                bad_params.append(f"cmrm.{name}")

                if bad_params:
                    raise RuntimeError(f"NaN/Inf found in parameters after optimizer.step(): {bad_params[:10]}")

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                # -------------------------------
                # train log（以 optimizer step 為單位）
                # -------------------------------
                if global_step % log_every == 0:
                    avg_total = running_total / running_count
                    avg_reg = running_reg / running_count
                    avg_mvc = running_mvc / running_count

                    print(
                        f"[train] step={global_step} "
                        f"total={avg_total:.4f} "
                        f"reg={lambda_reg * avg_reg:.4f} "
                        f"mvc={lambda_mvc * avg_mvc:.4f}"
                    )

                    history.append({
                        "step": global_step,
                        "epoch": epoch,
                        "split": "train",
                        "total": avg_total,
                        "reg": avg_reg,
                        "mvc": avg_mvc,
                    })
                    with open(history_path, "w", encoding="utf-8") as _hf:
                        json.dump(history, _hf, indent=2, ensure_ascii=False)

                    running_total = 0.0
                    running_reg = 0.0
                    running_mvc = 0.0
                    running_count = 0

                # -------------------------------
                # validation loss
                # -------------------------------
                if global_step % val_every == 0:
                    val_metrics = self.validate(
                        val_loader,
                        lambda_reg=lambda_reg,
                        lambda_mvc=lambda_mvc,
                        max_batches=max_val_batches,
                    )

                    print(
                        f"[val]   step={global_step} "
                        f"total={val_metrics['total']:.4f} "
                        f"reg={lambda_reg * val_metrics['reg']:.4f} "
                        f"mvc={lambda_mvc * val_metrics['mvc']:.4f}"
                    )

                    history.append({
                        "step": global_step,
                        "epoch": epoch,
                        "split": "val",
                        **val_metrics,
                    })
                    with open(history_path, "w", encoding="utf-8") as _hf:
                        json.dump(history, _hf, indent=2, ensure_ascii=False)

                    self.save_checkpoint(
                        save_dir=save_dir,
                        step=global_step,
                        epoch=epoch,
                        optimizer=optimizer,
                        best_val=best_val,
                    )

                    if val_metrics["total"] < (best_val - early_stopping_min_delta):
                        best_val = val_metrics["total"]
                        print(f"[best]  new best val total = {best_val:.4f}")
                        self.save_best(
                            save_dir=save_dir,
                            step=global_step,
                            epoch=epoch,
                            val_loss=best_val,
                            optimizer=optimizer,
                        )
                    else:
                        no_improve_count += 1
                        print(f"no improvement count = {no_improve_count}/{early_stopping_patience}")
                        if no_improve_count >= early_stopping_patience:
                            print(f"[early-stopping] stop at step={global_step}, best_val={best_val:.4f}")
                            early_stop = True
                            break

                # -------------------------------
                # plate evaluation
                # -------------------------------
                if global_step % eval_plate_every == 0:
                    torch.cuda.empty_cache()
                    seq_metrics = self.evaluate_plate(
                        val_loader,
                        max_new_tokens=plate_max_new_tokens,
                        debug_seq=plate_debug_seq,
                        max_batches=max_plate_batches,
                    )

                    avg_score = seq_metrics["avg_score"]
                    exact_match = seq_metrics["exact_match"]
                    char_acc = seq_metrics["char_acc"]

                    msg = f"[plate] step={global_step} "
                    msg += f"avg_score={avg_score:.4f} " if avg_score is not None else "avg_score=None "
                    msg += f"exact_match={exact_match:.4f} " if exact_match is not None else "exact_match=None "
                    msg += f"char_acc={char_acc:.4f}" if char_acc is not None else "char_acc=None"
                    print(msg)

                    history.append({
                        "step": global_step,
                        "epoch": epoch,
                        "split": "plate_eval",
                        **seq_metrics,
                    })
                    with open(history_path, "w", encoding="utf-8") as _hf:
                        json.dump(history, _hf, indent=2, ensure_ascii=False)

                # -------------------------------
                # periodic save
                # -------------------------------
                if global_step % save_every == 0:
                    self.save_checkpoint(
                        save_dir=save_dir,
                        step=global_step,
                        epoch=epoch,
                        optimizer=optimizer,
                        best_val=best_val,
                    )
            if early_stop:
                break


            # -------------------------------
            # epoch-end validation loss
            # -------------------------------
            val_metrics = self.validate(
                val_loader,
                lambda_reg=lambda_reg,
                lambda_mvc=lambda_mvc,
                max_batches=max_val_batches,
            )

            print(
                f"[epoch-end val] epoch={epoch} "
                f"total={val_metrics['total']:.4f} "
                f"reg={val_metrics['reg']:.4f} "
                f"mvc={val_metrics['mvc']:.4f}"
            )

            history.append({
                "step": global_step,
                "epoch": epoch,
                "split": "val_epoch_end",
                **val_metrics,
            })
            with open(history_path, "w", encoding="utf-8") as _hf:
                json.dump(history, _hf, indent=2, ensure_ascii=False)

            # -------------------------------
            # epoch-end plate evaluation
            # -------------------------------
            seq_metrics = self.evaluate_plate(
                val_loader,
                max_new_tokens=plate_max_new_tokens,
                debug_seq=plate_debug_seq,
                max_batches=max_plate_batches,
            )

            avg_score = seq_metrics["avg_score"]
            exact_match = seq_metrics["exact_match"]
            char_acc = seq_metrics["char_acc"]

            msg = f"[epoch-end plate] epoch={epoch} "
            msg += f"avg_score={avg_score:.4f} " if avg_score is not None else "avg_score=None "
            msg += f"exact_match={exact_match:.4f} " if exact_match is not None else "exact_match=None "
            msg += f"char_acc={char_acc:.4f} " if char_acc is not None else "char_acc=None "
            msg += f"valid_count={seq_metrics['valid_count']} num_plate={seq_metrics['num_plate']}"
            print(msg)

            history.append({
                "step": global_step,
                "epoch": epoch,
                "split": "plate_eval_epoch_end",
                **seq_metrics,
            })
            with open(history_path, "w", encoding="utf-8") as _hf:
                json.dump(history, _hf, indent=2, ensure_ascii=False)

            self.save_checkpoint(
                save_dir=save_dir,
                step=global_step,
                epoch=epoch,
                optimizer=optimizer,
                best_val=best_val,
            )

            if val_metrics["total"] < best_val:
                best_val = val_metrics["total"]
                print(f"[best]  new best val total = {best_val:.4f}")
                self.save_best(
                    save_dir=save_dir,
                    step=global_step,
                    epoch=epoch,
                    val_loss=best_val,
                    optimizer=optimizer,
                )

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        print("\nTraining finished.")
        print(f"Best val total: {best_val:.4f}")

    @torch.no_grad()
    def eval_only(self, val_loader, max_plate_batches=5):
        self.model.eval()

        metrics = self.evaluate_plate(
            val_loader,
            max_batches=max_plate_batches,
        )

        print(
            f"[eval-only] avg_score={metrics['avg_score']:.4f} "
            f"exact_match={metrics['exact_match']:.4f} "
            f"char_acc={metrics['char_acc']:.4f} "
            f"valid_count={metrics['valid_count']} "
            f"num_plate={metrics['num_plate']}"
        )

        return metrics
