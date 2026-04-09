from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
from torch.utils.data import Subset

from model.plate_recognition_trainer import LPLLM
from model.training_components.plate_track_dataset import build_track_dataset, split_track_dataset

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "Qwen3-VL" / "Qwen3-VL-8B-Instruct"
UFPR_ROOT_DIR = PROJECT_ROOT / "UFPR-SR-plates" / "1920_1080"
OUTPUT_ROOT = PROJECT_ROOT / "model" / "outputs" / "cmrm_ablation"

NUM_FRAMES = 5          # 恢復 5 幀；gradient accumulation 讓每次只跑 1 幀的 forward
TRAIN_RATIO = 0.8
SPLIT_SEED = 42
SUBSET_SEED = 3407
TRAIN_TRACKS_PER_CATEGORY = 400
VAL_TRACKS_PER_CATEGORY = 100
TRAIN_BATCH_SIZE = 1    # 每次 forward = 1 plate × 1 frame = 1 張圖（最省記憶體）
VAL_BATCH_SIZE = 1

COMMON_TRAIN_KWARGS = {
    "epochs": 3,
    "lr": 1e-5,
    "weight_decay": 1e-4,
    "grad_clip": 0.1,
    "grad_accum_steps": 5,  # 累積 5 個 mini-batch 再 optimizer.step()
                             # 等效 batch = 5 plates × 5 frames，與原始設定相近
    "log_every": 10,
    "val_every": 50,
    "save_every": 100,
    "max_val_batches": 25,
    "eval_plate_every": 100,
    "max_plate_batches": 10,
    "plate_max_new_tokens": 8,
    "plate_debug_seq": 2,
    "early_stopping_patience": 10,
    "early_stopping_min_delta": 0.02,
}

EXPERIMENTS = [
    {
        "name": "01_lora_only",
        "title": "LoRA",
        "use_cmrm": False,
        "train_lora": True,
        "train_cmrm": False,
        "lambda_reg": 1.0,
        "lambda_slot": 0.0,
        "lambda_feat": 0.0,
    },
    {
        "name": "02_lora_cmrm_reg",
        "title": "LoRA + CMRM (loss_reg)",
        "use_cmrm": True,
        "train_lora": True,
        "train_cmrm": True,
        "lambda_reg": 1.0,
        "lambda_slot": 0.0,
        "lambda_feat": 0.0,
    },
    {
        "name": "03_lora_cmrm_reg_slot",
        "title": "LoRA + CMRM (loss_reg + loss_slot)",
        "use_cmrm": True,
        "train_lora": True,
        "train_cmrm": True,
        "lambda_reg": 1.0,
        "lambda_slot": 0.1,
        "lambda_feat": 0.0,
    },
    {
        "name": "04_lora_cmrm_reg_feats",
        "title": "LoRA + CMRM (loss_reg + loss_feats)",
        "use_cmrm": True,
        "train_lora": True,
        "train_cmrm": True,
        "lambda_reg": 1.0,
        "lambda_slot": 0.0,
        "lambda_feat": 0.1,   # 原本 1.0 太強，直接把 LR feature 拉向 HR，干擾 LM
    },
    {
        "name": "05_lora_cmrm_reg_slot_feats",
        "title": "LoRA + CMRM (loss_reg + loss_slot + loss_feats)",
        "use_cmrm": True,
        "train_lora": True,
        "train_cmrm": True,
        "lambda_reg": 1.0,
        "lambda_slot": 0.1,
        "lambda_feat": 0.1,   # 同上
    },
]


def _select_balanced_subset(dataset, per_category: int, seed: int) -> Subset:
    if not isinstance(dataset, Subset):
        raise TypeError("Balanced subset selection expects torch.utils.data.Subset input.")

    rng = random.Random(seed)
    base_dataset = dataset.dataset
    grouped: Dict[str, List[int]] = {}

    for idx in dataset.indices:
        plate_dir = base_dataset.samples[idx]
        category = plate_dir.parent.name
        grouped.setdefault(category, []).append(idx)

    chosen: List[int] = []
    for category, indices in sorted(grouped.items()):
        indices = list(indices)
        rng.shuffle(indices)
        take_n = min(per_category, len(indices))
        chosen.extend(indices[:take_n])
        print(f"[subset] category={category} requested={per_category} selected={take_n}")

    chosen.sort()
    return Subset(base_dataset, chosen)


def _subset_overview(dataset, split_name: str) -> Dict[str, object]:
    if not isinstance(dataset, Subset):
        raise TypeError("Overview helper expects torch.utils.data.Subset input.")

    base_dataset = dataset.dataset
    category_counts: Dict[str, int] = {}
    labels = []

    for idx in dataset.indices:
        sample = base_dataset[idx]
        category = sample["plate_id"].split("/", 1)[0]
        category_counts[category] = category_counts.get(category, 0) + 1
        labels.append(sample["text_label"])

    unique_labels = len(set(labels))
    return {
        "split": split_name,
        "num_tracks": len(dataset),
        "category_counts": category_counts,
        "unique_labels": unique_labels,
    }


def _safe_value(value: Optional[float]) -> float:
    return float("nan") if value is None else float(value)


def _best_record(history: List[dict], split_name: str, key: str, mode: str):
    candidates = [r for r in history if r.get("split") == split_name and r.get(key) is not None]
    if not candidates:
        return None
    if mode == "min":
        return min(candidates, key=lambda r: r[key])
    if mode == "max":
        return max(candidates, key=lambda r: r[key])
    raise ValueError(f"Unsupported mode: {mode}")


def summarize_history(history: List[dict], config: dict, dataset_summary: dict) -> dict:
    train_records = [r for r in history if r.get("split") == "train"]
    val_records = [r for r in history if r.get("split") in {"val", "val_epoch_end"}]
    plate_records = [r for r in history if r.get("split") in {"plate_eval", "plate_eval_epoch_end"}]

    summary = {
        "experiment": config["name"],
        "title": config["title"],
        "config": config,
        "dataset": dataset_summary,
        "num_train_logs": len(train_records),
        "num_val_logs": len(val_records),
        "num_plate_logs": len(plate_records),
    }

    best_val = _best_record(history, "val", "total", "min") or _best_record(history, "val_epoch_end", "total", "min")
    best_plate = _best_record(history, "plate_eval", "avg_score", "max") or _best_record(history, "plate_eval_epoch_end", "avg_score", "max")
    last_train = train_records[-1] if train_records else None
    last_val = val_records[-1] if val_records else None
    last_plate = plate_records[-1] if plate_records else None

    if best_val is not None:
        summary["best_val"] = {
            "step": int(best_val["step"]),
            "epoch": int(best_val["epoch"]),
            "total": float(best_val["total"]),
            "reg": float(best_val["reg"]),
            "slot": float(best_val["slot"]),
            "feat": float(best_val["feat"]),
        }
    if best_plate is not None:
        summary["best_plate_eval"] = {
            "step": int(best_plate["step"]),
            "epoch": int(best_plate["epoch"]),
            "avg_score": _safe_value(best_plate.get("avg_score")),
            "exact_match": _safe_value(best_plate.get("exact_match")),
            "char_acc": _safe_value(best_plate.get("char_acc")),
            "valid_count": int(best_plate.get("valid_count", 0)),
            "num_plate": int(best_plate.get("num_plate", 0)),
        }
    if last_train is not None:
        summary["last_train"] = {
            key: float(last_train[key]) if key in {"total", "reg", "slot", "feat"} else int(last_train[key])
            for key in ["step", "epoch", "total", "reg", "slot", "feat"]
        }
    if last_val is not None:
        summary["last_val"] = {
            key: float(last_val[key]) if key in {"total", "reg", "slot", "feat"} else int(last_val[key])
            for key in ["step", "epoch", "total", "reg", "slot", "feat"]
        }
    if last_plate is not None:
        summary["last_plate_eval"] = {
            "step": int(last_plate["step"]),
            "epoch": int(last_plate["epoch"]),
            "avg_score": _safe_value(last_plate.get("avg_score")),
            "exact_match": _safe_value(last_plate.get("exact_match")),
            "char_acc": _safe_value(last_plate.get("char_acc")),
            "valid_count": int(last_plate.get("valid_count", 0)),
            "num_plate": int(last_plate.get("num_plate", 0)),
        }

    return summary


def plot_history(history, save_dir, name, reg_weight=1.0, lambda_slot=1.0, lambda_feat=1.0):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_records = [x for x in history if x["split"] == "train"]
    val_records = [x for x in history if x["split"] == "val"]
    val_epoch_records = [x for x in history if x["split"] == "val_epoch_end"]
    plate_records = [r for r in history if r.get("split") in {"plate_eval", "plate_eval_epoch_end"}]

    if train_records:
        x = [r["step"] for r in train_records]
        train_reg_w = [reg_weight * r["reg"] for r in train_records]
        train_slot_w = [lambda_slot * r["slot"] for r in train_records]
        train_feat_w = [lambda_feat * r["feat"] for r in train_records]

        plt.figure(figsize=(9, 5))
        plt.plot(x, [r["total"] for r in train_records], marker="o", label="train_total")
        plt.plot(x, train_reg_w, marker="o", label=f"train_reg*x{reg_weight}")
        plt.plot(x, train_slot_w, marker="o", label=f"train_slot*x{lambda_slot}")
        plt.plot(x, train_feat_w, marker="o", label=f"train_feat*x{lambda_feat}")
        plt.title(f"{name} Training Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / f"{name}_train_loss.png", dpi=200)
        plt.close()

    if val_records:
        x = [r["step"] for r in val_records]
        val_reg_w = [reg_weight * r["reg"] for r in val_records]
        val_slot_w = [lambda_slot * r["slot"] for r in val_records]
        val_feat_w = [lambda_feat * r["feat"] for r in val_records]

        plt.figure(figsize=(9, 5))
        plt.plot(x, [r["total"] for r in val_records], marker="o", label="val_total")
        plt.plot(x, val_reg_w, marker="o", label=f"val_reg*x{reg_weight}")
        plt.plot(x, val_slot_w, marker="o", label=f"val_slot*x{lambda_slot}")
        plt.plot(x, val_feat_w, marker="o", label=f"val_feat*x{lambda_feat}")
        plt.title(f"{name} Validation Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / f"{name}_val_loss.png", dpi=200)
        plt.close()

    plt.figure(figsize=(9, 5))
    drew = False

    if train_records:
        plt.plot([r["step"] for r in train_records], [r["total"] for r in train_records], marker="o", label="train_total")
        drew = True
    if val_records:
        plt.plot([r["step"] for r in val_records], [r["total"] for r in val_records], marker="o", label="val_total")
        drew = True
    if val_epoch_records:
        plt.plot([r["step"] for r in val_epoch_records], [r["total"] for r in val_epoch_records], marker="o", label="val_epoch_end_total")
        drew = True

    if drew:
        plt.title(f"{name} Train / Val Total Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / f"{name}_train_val_total_loss.png", dpi=200)
    plt.close()

    if plate_records:
        xs = [r["step"] for r in plate_records]
        avg_scores = [_safe_value(r.get("avg_score")) for r in plate_records]
        exact_matches = [_safe_value(r.get("exact_match")) for r in plate_records]
        char_accs = [_safe_value(r.get("char_acc")) for r in plate_records]

        plt.figure(figsize=(9, 5))
        plt.plot(xs, avg_scores, marker="o", label="avg_score")
        plt.plot(xs, exact_matches, marker="o", label="exact_match")
        plt.plot(xs, char_accs, marker="o", label="char_acc")
        plt.title(f"{name} Plate Metrics")
        plt.xlabel("Step")
        plt.ylabel("Metric")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / f"{name}_plate_metrics.png", dpi=200)
        plt.close()


def _extract_series(history: List[dict], split_names: Iterable[str], key: str):
    xs, ys = [], []
    for record in history:
        if record.get("split") in split_names and key in record and record[key] is not None:
            xs.append(record["step"])
            ys.append(record[key])
    return xs, ys


def plot_combined_comparison(results: List[dict], save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 2, figsize=(16, 10))

    for result in results:
        label = result["title"]
        history = result["history"]

        xs, ys = _extract_series(history, {"val", "val_epoch_end"}, "total")
        if xs:
            axes[0, 0].plot(xs, ys, marker="o", label=label)

        xs, ys = _extract_series(history, {"val", "val_epoch_end"}, "reg")
        if xs:
            axes[0, 1].plot(xs, ys, marker="o", label=label)

        xs, ys = _extract_series(history, {"plate_eval", "plate_eval_epoch_end"}, "avg_score")
        if xs:
            axes[1, 0].plot(xs, ys, marker="o", label=label)

        xs, ys = _extract_series(history, {"plate_eval", "plate_eval_epoch_end"}, "char_acc")
        if xs:
            axes[1, 1].plot(xs, ys, marker="o", label=label)

    titles = [
        "Validation Total Loss",
        "Validation Reg Loss",
        "Plate Avg Score",
        "Plate Char Accuracy",
    ]
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[0, 0].set_ylabel("Loss")
    axes[0, 1].set_ylabel("Loss")
    axes[1, 0].set_ylabel("Score")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].set_ylim(0, 1.05)

    figure.tight_layout()
    figure.savefig(save_dir / "all_experiments_comparison.png", dpi=220)
    plt.close(figure)

    labels = [result["title"] for result in results]
    best_val = [result["summary"].get("best_val", {}).get("total", math.nan) for result in results]
    best_avg_score = [result["summary"].get("best_plate_eval", {}).get("avg_score", math.nan) for result in results]
    best_exact_match = [result["summary"].get("best_plate_eval", {}).get("exact_match", math.nan) for result in results]
    best_char_acc = [result["summary"].get("best_plate_eval", {}).get("char_acc", math.nan) for result in results]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    metrics = [
        (best_val, "Best Validation Total", "Loss"),
        (best_avg_score, "Best Plate Avg Score", "Score"),
        (best_exact_match, "Best Plate Exact Match", "Rate"),
        (best_char_acc, "Best Plate Char Accuracy", "Rate"),
    ]

    for ax, (values, title, ylabel) in zip(axes.flat, metrics):
        ax.bar(range(len(labels)), values)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_dir / "all_experiments_best_metrics.png", dpi=220)
    plt.close(fig)


def write_summary_csv(results: List[dict], save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "experiment_summary.csv"
    fieldnames = [
        "experiment",
        "title",
        "lambda_reg",
        "lambda_slot",
        "lambda_feat",
        "best_val_total",
        "best_val_reg",
        "best_val_slot",
        "best_val_feat",
        "best_plate_avg_score",
        "best_plate_exact_match",
        "best_plate_char_acc",
        "train_tracks",
        "val_tracks",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            summary = result["summary"]
            config = summary["config"]
            best_val = summary.get("best_val", {})
            best_plate = summary.get("best_plate_eval", {})
            dataset = summary.get("dataset", {})
            writer.writerow(
                {
                    "experiment": summary["experiment"],
                    "title": summary["title"],
                    "lambda_reg": config["lambda_reg"],
                    "lambda_slot": config["lambda_slot"],
                    "lambda_feat": config["lambda_feat"],
                    "best_val_total": best_val.get("total", math.nan),
                    "best_val_reg": best_val.get("reg", math.nan),
                    "best_val_slot": best_val.get("slot", math.nan),
                    "best_val_feat": best_val.get("feat", math.nan),
                    "best_plate_avg_score": best_plate.get("avg_score", math.nan),
                    "best_plate_exact_match": best_plate.get("exact_match", math.nan),
                    "best_plate_char_acc": best_plate.get("char_acc", math.nan),
                    "train_tracks": dataset.get("train", {}).get("num_tracks", 0),
                    "val_tracks": dataset.get("val", {}).get("num_tracks", 0),
                }
            )


def run_experiment(config: dict, train_dataset, val_dataset, dataset_summary: dict, output_root: Path) -> dict:
    exp_dir = output_root / config["name"]
    exp_dir.mkdir(parents=True, exist_ok=True)

    history_path = exp_dir / "history.json"
    if history_path.exists():
        print("\n" + "=" * 80)
        print(f"[skip] {config['title']} — history.json already exists, loading.")
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        plot_history(
            history=history,
            save_dir=exp_dir,
            name=config["name"],
            reg_weight=config["lambda_reg"],
            lambda_slot=config["lambda_slot"],
            lambda_feat=config["lambda_feat"],
        )
        summary = summarize_history(history, config=config, dataset_summary=dataset_summary)
        with open(exp_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return {
            "name": config["name"],
            "title": config["title"],
            "history": history,
            "summary": summary,
            "save_dir": str(exp_dir),
        }

    print("\n" + "=" * 80)
    print(f"[experiment] {config['title']}")
    print(json.dumps(config, indent=2, ensure_ascii=False))

    trainer = LPLLM(
        model_path=MODEL_PATH,
        num_slots=7,
        cmrm_dim=4096,
        num_heads=8,
        use_cmrm=config["use_cmrm"],
        train_LoRA=config["train_lora"],
        train_cmrm=config["train_cmrm"],
    )

    train_loader = trainer.build_dataloader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = trainer.build_dataloader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=exp_dir,
        lambda_reg=config["lambda_reg"],
        lambda_slot=config["lambda_slot"],
        lambda_feat=config["lambda_feat"],
        **COMMON_TRAIN_KWARGS,
    )

    history_path = exp_dir / "history.json"
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    plot_history(
        history=history,
        save_dir=exp_dir,
        name=config["name"],
        reg_weight=config["lambda_reg"],
        lambda_slot=config["lambda_slot"],
        lambda_feat=config["lambda_feat"],
    )

    summary = summarize_history(history, config=config, dataset_summary=dataset_summary)
    with open(exp_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return {
        "name": config["name"],
        "title": config["title"],
        "history": history,
        "summary": summary,
        "save_dir": str(exp_dir),
    }


def main():
    output_root = Path(OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    full_dataset = build_track_dataset(
        root_dir=UFPR_ROOT_DIR,
        num_frames=NUM_FRAMES,
        categories=("brazilian", "mercosur"),
    )
    train_dataset, val_dataset = split_track_dataset(
        full_dataset,
        train_ratio=TRAIN_RATIO,
        seed=SPLIT_SEED,
    )

    balanced_train = _select_balanced_subset(train_dataset, TRAIN_TRACKS_PER_CATEGORY, SUBSET_SEED)
    balanced_val = _select_balanced_subset(val_dataset, VAL_TRACKS_PER_CATEGORY, SUBSET_SEED + 1)

    dataset_summary = {
        "train": _subset_overview(balanced_train, "train"),
        "val": _subset_overview(balanced_val, "val"),
    }
    with open(output_root / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, indent=2, ensure_ascii=False)

    print("[dataset-summary]")
    print(json.dumps(dataset_summary, indent=2, ensure_ascii=False))

    results = []
    for config in EXPERIMENTS:
        results.append(
            run_experiment(
                config=config,
                train_dataset=balanced_train,
                val_dataset=balanced_val,
                dataset_summary=dataset_summary,
                output_root=output_root,
            )
        )

    with open(output_root / "all_experiments_summary.json", "w", encoding="utf-8") as f:
        json.dump([result["summary"] for result in results], f, indent=2, ensure_ascii=False)

    write_summary_csv(results, output_root)
    plot_combined_comparison(results, output_root)
    print(f"[done] outputs saved to: {output_root}")


if __name__ == "__main__":
    main()
