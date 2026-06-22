from dataclasses import dataclass, field
from typing import List


@dataclass
class PathConfig:
    input_text_file: str = "data/tiny-stories/TinyStoriesV2-GPT4-valid.txt"
    train_data_file: str = "data/tiny-stories/TinyStoriesV2-GPT4-valid.bin"
    val_data_file: str = "data/tiny-stories/TinyStoriesV2-GPT4-valid.bin"
    output_dir: str = "outputs/prt"

@dataclass
class TrainingConfig:
    # common
    seed: int = 1337
    device: str = "cuda:0"
    # dtype: str = "bfloat16"

    # model architecture
    context_len: int = 256
    d_model: int = 512
    num_heads: int = 16
    d_ff: int = 1344               # ~ d_model * 8/3 & d_ff % 64 == 0
    num_layers: int = 4
    theta: float = 10_000.0
    # attention_type: str = "flash_attention_2"
    # use_compile: bool = False    # whether to use torch.compile for the model

    # training loop
    batch_size: int = 64
    num_iters: int = 10     # 5000
    log_interval: int = 1   # 100
    warmup_iters: int = 1
    cosine_cycle_iters: int = 10

    # AdamW optimizer
    learning_rate: float = 1e-5
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    # sampling parameters
    temperature: float = 1.0
    top_p: float = 1.0
    min_tokens: int = 4
    max_tokens: int = 1024
    
    # evaluation
    eval_interval: int = 4                              # evaluate every N grpo steps (-1 = disabled)
    max_val_examples: int = 1024                        # max number of val examples per eval

    # checkpointing
    checkpoint_interval: int = 4                        # save a checkpoint every N grpo steps (-1 = disabled)

    # logging
    n_rollouts_to_log: int = 16                         # number of random rollouts to save to disk per grpo step (0 = disabled)

    # wandb
    wandb_project: str = "cs336-hw1"                    # wandb project name (empty = disabled)
    wandb_run_name: str = "celestial-universe"          # wandb run name
    # wandb_tags: List[str] = field(default_factory=lambda: ["prt"])
    
    # memory optimization flags
    # track_peak_memory: bool = True                      # log peak VRAM at key points in the training loop
    # use_gradient_checkpointing: bool = True             # trade compute for activation memory
    # use_bnb_adamw8bit: bool = True                      # use bitsandbytes AdamW8bit optimizer
    
    
@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)