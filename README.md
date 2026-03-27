# VLM Caching & Contrastive Fine-Tuning

A robust, VRAM-efficient pipeline for interacting with Large Vision Language Models (specifically `LiquidAI/LFM2-VL-450M`), caching multi-modal generative responses, and extracting dense conversational embeddings for contrastive classification (SetFit style).

## Features

* **Resumable Multi-Modal Cache (`lvm_utils.cache_store`)**: A thread-safe, Parquet-backed index that tracks deterministic SHA-256 hashes of input images. It seamlessly caches conversation states (JSON) and tightly compresses images (WebP) to prevent costly repetitive inference. 
* **Extreme VRAM Optimizations (`lvm_utils.model_helpers`)**: 
  * Safely implements 4-bit `nf4` BitsAndBytes quantization on causal VLMs (preserving `bfloat16` inputs to the Vision Tower patch embeddings to allow PEFT gradients).
  * Automatically applies gradient checkpointing for extensive sequence context.
  * Overrides the VLM `lm_head` to mathematically skip logits calculation during embedding extraction, cutting batch-training VRAM footprints significantly.
* **Streamlined Packaging**: Easily extendable pipeline logic bundled as the local `lvm_utils` package.

## Installation

This project utilizes `uv` for fast dependency resolution. All utilities are packaged as an editable python module.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the pipeline package and dependencies
uv pip install -e .
```

## Project Structure

* **`lvm_utils/`**: Core python package.
  * `model_helpers.py`: Model loading, 4-bit/PEFT configurations, and rapid batch-embedding extractions.
  * `cache_store.py` / `cache_schema.py`: Parquet-dataframe driven data persistence.
  * `utils.py`: Image formatting, deterministic SHA256 hashing, and system helpers.
* **`generate_conversations.py`**: Stage 1 script. Pushes an image dataset through the VLM to generate detailed descriptions, natively saving the output locally to `cache_data/`. Safe to stop and resume at any time.
* **`LVM_train.py`**: Stage 2 script. Loads the verified cache, pulls out batched representations efficiently without regenerating text, and orchestrates the PEFT/LoRA contrastive setup.

## Pipeline Workflow

### 1. Data Generation (Caching)
Run the first-stage generation to build a local dataset of descriptions out of your initial image set (e.g. CIFAR).
```bash
python generate_conversations.py
```
This builds an index in `cache_data/index.parquet` and deduplicates identical images.

### 2. SetFit / Contrastive Finetuning
Engage the training loop. This script hooks into the generated cache and utilizes the VRAM-efficient batch encoder to train continuous embeddings over the prompt contexts.

```bash
python LVM_train.py
```

## Memory Requirements
Optimized targeting consumer GPUs (e.g., 12GB+ VRAM). The combination of 4-bit base weights, PEFT adapters, gradient checkpointing, and dynamic `lm_head` bypassing permits handling long vision-language multi-turn prompts within constrained memory limits.
