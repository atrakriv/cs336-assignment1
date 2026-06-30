import os
import json
import argparse
import random

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf

from configs.defaults import Config
# from src.tokenizer import Tokenizer
from hw1 import *
from util import WallclockTimer

if __name__ == "__main__":    

    # ── Load the config ──────────────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to a yaml config file (overrides defaults)")
    args, overrides = parser.parse_known_args()  # extra args as dotlist overrides, e.g. training.max_lr=3e-4

    default_config = OmegaConf.structured(Config)

    if args.config is not None:
        cfg = OmegaConf.merge(default_config, OmegaConf.load(args.config))
    else:
        cfg = default_config

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    # ── Reproducibility ──────────────────────────────────────────────────────
    seed = cfg.training.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ── Hyperparameters ──────────────────────────────────────────────────────
    device  = cfg.training.device
    context_len  = cfg.training.context_len
    batch_size = cfg.training.batch_size
    num_iters = cfg.training.num_iters
    log_interval = cfg.training.log_interval
    d_model = cfg.training.d_model
    num_heads = cfg.training.num_heads
    d_ff = cfg.training.d_ff
    num_layers = cfg.training.num_layers
    theta = cfg.training.theta
    max_lr = cfg.training.max_lr
    min_lr = cfg.training.min_lr
    betas = (cfg.training.adam_beta1, cfg.training.adam_beta2)
    warmup_iters = cfg.training.warmup_iters
    cosine_cycle_iters = cfg.training.cosine_cycle_iters
    eval_interval = cfg.training.eval_interval
    max_val_examples = cfg.training.max_val_examples

    # ── Data ──────────────────────────────────────────────────────────────────
    # def tokenize_to_bin(input_path, dataset_bin):
    #     special_tokens = ["<|endoftext|>"]
    #     tokenizer = Tokenizer.from_files(
    #         vocab_filepath="vocab.json",
    #         merges_filepath="merges.txt",
    #         special_tokens=special_tokens,
    #     )
    #     if not os.path.exists(dataset_bin):
    #         arr = tokenizer.encode_file_parallel(input_path)
    #         arr.tofile(dataset_bin)
    #         n_tokens = len(arr)
    #         del arr
    #     else:
    #         n_tokens = os.path.getsize(dataset_bin) // np.dtype(cfg.paths.dataset_dtype).itemsize
    #     return np.memmap(dataset_bin, dtype=cfg.paths.dataset_dtype, mode="r", shape=(n_tokens,))

    def load_bin(dataset_bin):
        n_tokens = os.path.getsize(dataset_bin) // np.dtype(cfg.paths.dataset_dtype).itemsize
        return np.memmap(dataset_bin, dtype=cfg.paths.dataset_dtype, mode="r", shape=(n_tokens,))

    train_dataset = load_bin(cfg.paths.train_bin_file)
    val_dataset = load_bin(cfg.paths.val_bin_file)

    # ── Logging ──────────────────────────────────────────────────────────────
    in_sweep = os.environ.get("WANDB_SWEEP_ID") is not None
    wandb.init(
        project=cfg.training.wandb_project,
        name=None if in_sweep else cfg.training.wandb_run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # ── Model + optimizer ────────────────────────────────────────────────────
    # actual_vocab_size = max(tokenizer.vocab.keys()) + 1
    model = TransformerLM(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        theta=theta,
        context_length=context_len,
        vocab_size=cfg.training.vocab_size,
        num_layers=num_layers,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=max_lr,
        betas=betas,
        weight_decay=cfg.training.weight_decay,
        eps=cfg.training.adam_eps,
    )

    # ── Checkpointing ────────────────────────────────────────────────────────
    checkpoint_dir = cfg.paths.output_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
    best_val_loss_path = os.path.join(checkpoint_dir, "best_val_loss.json")

    start_step = 1
    best_val_loss = float("inf")
    wallclock_offset = 0.0  # seconds accumulated across prior (resumed) runs
    if os.path.exists(latest_checkpoint_path):
        last_step = load_checkpoint(latest_checkpoint_path, model, optimizer)
        start_step = last_step + 1
        if os.path.exists(best_val_loss_path):
            with open(best_val_loss_path) as f:
                state = json.load(f)
                best_val_loss = state["best_val_loss"]
                wallclock_offset = state.get("cumulative_wallclock_sec", 0.0)
        print(f"Resumed from checkpoint at step {last_step} (best_val_loss={best_val_loss:.4f}, "
              f"wallclock_so_far={wallclock_offset:.1f}s)")

    timer = WallclockTimer(wallclock_offset)

    # ── Training loop ────────────────────────────────────────────────────────
    model.train()
    for step in range(start_step, num_iters + 1):
        lr = get_cosine_lr(step, max_lr, min_lr, warmup_iters, cosine_cycle_iters)
        for pg in optimizer.param_groups: pg['lr'] = lr
        x, y = data_loader(train_dataset, batch_size, context_len, device)
        logits = model(x)  # (B, T, V)
        loss = cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        clip_gradient(model.parameters(), max_norm = 1.0)
        optimizer.step()
        if step % log_interval == 0:
            elapsed = timer.elapsed()
            print(f"step {step:>6} | loss {loss.item():.4f} | lr {lr:.2e} | time {elapsed:.1f}s")
            wandb.log({"train/loss": loss.item(), "train/lr": lr, "train/wallclock_sec": elapsed}, step=step)
        if eval_interval > 0 and step % eval_interval == 0:
            model.eval()
            num_val_batches = max(1, max_val_examples // batch_size)
            with torch.no_grad():
                val_losses = []
                for _ in range(num_val_batches):
                    x_val, y_val = data_loader(val_dataset, batch_size, context_len, device)
                    val_logits = model(x_val)
                    val_losses.append(cross_entropy(val_logits, y_val).item())
            val_loss = sum(val_losses) / len(val_losses)
            elapsed = timer.elapsed()
            print(f"step {step:>6} | val_loss {val_loss:.4f} | time {elapsed:.1f}s")
            wandb.log({"val/loss": val_loss, "val/wallclock_sec": elapsed}, step=step)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, step, latest_checkpoint_path)
                with open(best_val_loss_path, "w") as f:
                    json.dump({"best_val_loss": best_val_loss, "cumulative_wallclock_sec": elapsed}, f)
            model.train()

    wandb.finish()
