# DeSP Core

This repository contains the core implementation of DeSP for LLM-based recommendation experiments.

The release is intentionally code-focused. It does not include raw datasets, model checkpoints, generated results, paper assets, or comparison-method implementations.

## Contents

```text
.
├── main.py                  # DeSP training, inference, and evaluation entry point
├── compute_desp_weights.py  # Builds DeSP semantic debiasing weights
├── new_trainer.py           # Custom Trainer with weighted SFT loss
├── data_collator.py         # Local collator for labels and DeSP weights
├── requirements.txt
└── README.md
```

## Scope

- `main.py` contains the main DeSP fine-tuning, inference, and evaluation pipeline.
- `compute_desp_weights.py` contains the semantic debiasing weight construction logic.
- `new_trainer.py` implements the weighted SFT training objective.
- `data_collator.py` provides the local batch collation used by the custom trainer.

## External Assets

The following assets are expected to be prepared separately:

- recommendation datasets
- item id/name mappings
- item embedding tensors
- base LLM checkpoints
- experiment-specific output directories

These files are not distributed in this repository.

## Usage

The implementation is organized around two stages:

1. Construct DeSP semantic debiasing weights.
2. Run weighted supervised fine-tuning and evaluation.

Both stages are parameterized through command-line arguments in the corresponding Python entry points. Paths and hyperparameters should be adapted to the local experiment setup.

## Dependencies

The main Python dependencies are listed in `requirements.txt`. CUDA, PyTorch, and distributed training settings should be configured according to the target machine.

## Exclusions

- Raw data and processed data are excluded.
- Trained LoRA adapters and base model checkpoints are excluded.
- Generated figures, logs, and metric files are excluded.
- Baseline and comparison-method scripts are excluded.
