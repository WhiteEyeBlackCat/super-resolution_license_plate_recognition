# Plate Recognition Project

## Project layout

- `train_cmrm_ablation.py`: training and ablation entrypoint.
- `model/`: core model code, training components, checkpoints, and training outputs.
- `scripts/`: utility scripts such as ablation runner and plot regeneration.
- `notebooks/`: exploratory notebooks for preprocessing, model inspection, and checkpoint inspection.
- `UFPR-SR-plates/`: UFPR super-resolution plate dataset.
- `challenge_development_set_final/`: challenge development data.
- `Qwen3-VL/`: local model weights and upstream Qwen3-VL source snapshot.
- `archives/`: downloaded zip/html artifacts kept only for reference.

## Common commands

```bash
python -m train_cmrm_ablation
```

```bash
bash scripts/run_cmrm_ablation.sh
```

```bash
python scripts/regenerate_ablation_plots.py
```

## Notes

- Code paths now use project-relative locations instead of hard-coded absolute paths.
- Historical logs may still mention the old `new/` directory; those are preserved as old run records.
