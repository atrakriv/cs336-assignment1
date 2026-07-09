from dataclasses import dataclass, field
from typing import List


@dataclass
class PathConfig:
    train_text_file: str = "data/tinystories/TinyStoriesV2-GPT4-train.txt"
    val_text_file: str = "data/tinystories/TinyStoriesV2-GPT4-valid.txt"
    train_bin_file: str = "data/tinystories/TinyStoriesV2-GPT4-train.bin"
    val_bin_file: str = "data/tinystories/TinyStoriesV2-GPT4-valid.bin"
    dataset_dtype: str = "uint16"
    output_dir: str = "outputs"

@dataclass
class TrainingConfig:
    # common
    seed: int = 1337
    device: str = "cuda:0"
    # dtype: str = "bfloat16"

    # model architecture
    vocab_size: int = 10000
    context_len: int = 256
    d_model: int = 512
    num_heads: int = 16
    d_ff: int = 1344                                    # ~ d_model * 8/3 & d_ff % 64 == 0
    num_layers: int = 4
    theta: float = 10_000.0
    # use_compile: bool = False                         # whether to use torch.compile for the model
    # attention_type: str = "flash_attention_2"

    # training loop
    batch_size: int = 64
    num_iters: int = 5000
    log_interval: int = 10

    # AdamW optimizer
    max_lr: float = 3e-4
    min_lr_ratio: float = 0.1  # min_lr = max_lr * min_lr_ratio
    warmup_iters: int = 250
    cosine_cycle_iters: int = 5000
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    # evaluation
    eval_interval: int = 100                            # evaluate every N grpo steps (-1 = disabled)
    max_val_examples: int = 1024                        # max number of val examples per eval
    
    # sampling parameters
    temperature: float = 1.0
    top_p: float = 1.0
    min_tokens: int = 4
    max_tokens: int = 1024

    # wandb
    wandb_project: str = "cs336-hw1"                    # wandb project name (empty = disabled)
    wandb_run_name: str = "celestial-universe"          # wandb run name
    # wandb_tags: List[str] = field(default_factory=lambda: ["prt"])
    
    
@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)