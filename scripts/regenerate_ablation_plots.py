"""
Regenerate all plots from existing history.json files without re-training.
Run from the plate_recognition directory:
    python scripts/regenerate_ablation_plots.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_cmrm_ablation import (
    EXPERIMENTS,
    OUTPUT_ROOT,
    plot_history,
    plot_combined_comparison,
    summarize_history,
    write_summary_csv,
)

OUTPUT_ROOT_PATH = Path(OUTPUT_ROOT)

# Fake dataset_summary in case we don't have it (used only for summary metadata)
_ds_summary_path = OUTPUT_ROOT_PATH / "dataset_summary.json"
if _ds_summary_path.exists():
    with open(_ds_summary_path, "r", encoding="utf-8") as f:
        dataset_summary = json.load(f)
else:
    dataset_summary = {"train": {}, "val": {}}

results = []
for config in EXPERIMENTS:
    exp_dir = OUTPUT_ROOT_PATH / config["name"]
    history_path = exp_dir / "history.json"

    if not history_path.exists():
        print(f"[skip] {config['name']} — no history.json found")
        continue

    print(f"[plot] {config['name']}")
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    plot_history(
        history=history,
        save_dir=exp_dir,
        name=config["name"],
        reg_weight=config["lambda_reg"],
        lambda_mvc=config["lambda_mvc"],
    )

    summary = summarize_history(history, config=config, dataset_summary=dataset_summary)
    with open(exp_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    results.append({
        "name": config["name"],
        "title": config["title"],
        "history": history,
        "summary": summary,
        "save_dir": str(exp_dir),
    })

if results:
    write_summary_csv(results, OUTPUT_ROOT_PATH)
    plot_combined_comparison(results, OUTPUT_ROOT_PATH)
    print(f"\n[done] Combined plots saved to: {OUTPUT_ROOT_PATH}")
    print(f"       Experiments plotted: {[r['name'] for r in results]}")
else:
    print("[warn] No history.json files found — nothing to plot.")
