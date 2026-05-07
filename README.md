# VLM Caching, Training, and Evaluation

This repository provides a workflow for building cached CIFAR-100 conversation datasets, training LVM and SigLIP2 models, and evaluating the resulting runs with Monte-Carlo dropout.

## Installation

Create a virtual environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
uv pip install -e .
```

## Workflow

### 1. Generate cached CIFAR-100 conversations

Use the conversation generation script to build cached train and test datasets for CIFAR-100. The output is stored in a cache directory such as `cache_data/`, with separate train and test cache roots when needed.

```bash
python generate_conversations.py
```

Repeat the generation step for both splits so you have cached train and test datasets ready before training and evaluation.

### 2. Start an LVM training run

Launch the LVM training runner with a JSON config file. The default config used in this workspace is `configs/cifar100_base.json`.

```bash
python train_runner.py --config configs/cifar100_base.json
```

This writes each experiment to `runs/<experiment_name>/` with:

- `config.json`
- `metrics.csv`
- `checkpoint-*/`

### 3. Start a SigLIP2 training run

Use the SigLIP2 runner for the SigLIP2-based model family.

```bash
python siglip2_runner.py
```

As with the LVM runner, each run writes its configuration, metrics, and checkpoints into `runs/<experiment_name>/`.

### 4. Evaluate with Monte-Carlo dropout

After training, run the evaluation script to produce the final classifier metrics from the saved checkpoints.

```bash
python evaluate_mc_dropout.py \
  --run_dir runs/<experiment_name> \
  --model_type lvm \
  --train_cache_dir cached_cifar100 \
  --test_cache_dir cached_cifar100_test
```

Use `--model_type siglip` for SigLIP2 runs.

## Repository Layout

- `generate_conversations.py`: Builds cached CIFAR-100 conversations.
- `train_runner.py`: Runs LVM training from a JSON config.
- `siglip2_runner.py`: Runs SigLIP2 training.
- `evaluate_mc_dropout.py`: Evaluates saved runs using Monte-Carlo dropout.
- `lvm_utils/`: Shared dataset, model, cache, and utility code.
- `configs/`: Training configuration files.

## Outputs

Typical run artifacts include:

- `metrics.csv` for loss and accuracy tracking over time
- TensorBoard logs
- model checkpoints saved at the configured interval
- `mc_head.pth` and related checkpoint state for resuming or evaluation

## Notes

- The cache directories must exist before training or evaluation.
- The runners are designed to resume from checkpoints when needed.
- For exact command-line options, inspect the script help or the config file used for a given run.
