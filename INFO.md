## What's implemented

### BPE Tokenizer (`src/tokenizer/`)
- `bpe.py` — byte-pair encoding training algorithm
- `tokenizer.py` — encode / decode with learned vocab and merges
- `tkz_main.py` — CLI entry point for training a tokenizer, tokenizing data, and creating a .bin file

### Transformer model (`hw1.py`)
| Class / function | Description |
|---|---|
| `Linear` | Weight matrix with truncated-normal init (Xavier std) |
| `Embedding` | Learnable token embedding table |
| `RMSNorm` | Root-mean-square layer normalization |
| `SwiGLU` | SwiGLU feed-forward block (w1, w2, w3) |
| `FFNSiLU` | Two-layer SiLU feed-forward block |
| `RotaryPositionalEmbedding` | RoPE with precomputed sin/cos buffers |
| `MultiHeadSelfAttention` | Multi-head attention with causal mask and RoPE |
| `TransformerBlock` | Pre-norm residual block (RMSNorm → Attn → RMSNorm → FFN) |
| `TransformerLM` | Full decoder-only language model |
| `AdamW` | AdamW optimizer with bias-corrected moment estimates |
| `get_cosine_lr` | Linear warmup → cosine decay → constant min LR |
| `clip_gradient` | Global gradient norm clipping |
| `cross_entropy` | Numerically stable cross-entropy loss |
| `data_loader` | Random-batch sampler over a memory-mapped token array |
| `save_checkpoint` / `load_checkpoint` | Model + optimizer serialization |

### Training (`hw1_main.py`)
- OmegaConf-based config with CLI dotlist overrides (`key=value`)
- Weights & Biases logging (train loss, val loss, LR, wallclock time)
- Automatic checkpoint resume (carries wallclock offset across runs)
- W&B sweep support for hyperparameter search

### Inference (`hw1_inference.py`)
- Top-p (nucleus) sampling with temperature scaling
- Accepts an ad-hoc `--prompt` or samples continuations from a validation file

### Utilities (`utils.py`)
- `WallclockTimer` — cumulative elapsed-time tracker, resume-aware
- `upload_bin_to_hf` / `download_bin_from_hf` — push/pull tokenized `.bin` files to/from HuggingFace Hub
- `convert_to_hf` — convert custom vocab/merges JSON to HuggingFace BPE format


## Setup

### Download data

```sh
mkdir -p data/tinystories

# TinyStories
wget -P data/tinystories https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget -P data/tinystories https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# OpenWebText sample
wget -P data/openwebtext https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
wget -P data/openwebtext https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip data/openwebtext/owt_train.txt.gz data/openwebtext/owt_valid.txt.gz
```

### Experiments
- `wandb sweep configs/sweep_lr_grid.yaml`
- SiLU: commit 1243503 ([link](https://github.com/atrakriv/cs336-assignment1/commit/1243503e1d59730f122f24f6868ebd693c305b10))
- Nope: commit 3535781 ([link](https://github.com/atrakriv/cs336-assignment1/commit/3535781d7c28e6d3249a9975f7c1ca1793d5307e))



## Training

Run with defaults: `uv run hw1_main.py`


Override individual config values via dotlist: `uv run hw1_main.py training.max_lr=1e-3 training.num_iters=10000`


Load a YAML config file and optionally add dotlist overrides: `uv run hw1_main.py --config configs/sweep_lr.yaml training.batch_size=128`


Checkpoints are written to `outputs/` (or `outputs/<wandb_run_id>/` during a sweep).

### Key hyperparameters (defaults)

| Parameter | Default | Description |
|---|---|---|
| `d_model` | 512 | Model dimension |
| `num_heads` | 16 | Attention heads |
| `d_ff` | 1344 | FFN hidden dim (~8/3 × d_model) |
| `num_layers` | 4 | Transformer blocks |
| `context_len` | 256 | Sequence length |
| `batch_size` | 64 | Batch size |
| `max_lr` | 3e-4 | Peak learning rate |
| `min_lr_ratio` | 0.1 | `min_lr = max_lr × min_lr_ratio` |
| `warmup_iters` | 250 | Linear warmup steps |
| `cosine_cycle_iters` | 5000 | Cosine decay steps |
| `weight_decay` | 0.1 | AdamW weight decay |


## Inference

Custom prompt: `uv run hw1_inference.py --prompt "Once upon a time" --max_new_tokens 200`

Sample continuations from the validation set: `uv run hw1_inference.py --num_samples 5 --prompt_tokens 20`

Load a specific checkpoint: `uv run hw1_inference.py --checkpoint outputs/checkpoint_latest.pt --prompt "In a kingdom far away"`