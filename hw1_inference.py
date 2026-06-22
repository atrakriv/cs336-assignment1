import argparse
import random

import numpy as np
import torch
from omegaconf import OmegaConf

from configs.defaults import Config
from hw1 import *


def top_p_sample(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    probs = softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumprobs = torch.cumsum(sorted_probs, dim=-1)
    mask = (cumprobs - sorted_probs) > top_p  # always keep the top token
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    choice = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_idx.gather(-1, choice).squeeze(-1)


@torch.no_grad()
def generate(model, prompt_ids, context_len, max_new_tokens, temperature, top_p, min_tokens, eos_id, device):
    model.eval()
    ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        context = ids[-context_len:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)[:, -1, :] / max(temperature, 1e-6)
        next_id = top_p_sample(logits, top_p).item()
        ids.append(next_id)
        if eos_id is not None and next_id == eos_id and len(ids) - len(prompt_ids) >= min_tokens:
            break
    return ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to a yaml config file (overrides defaults)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (defaults to <output_dir>/checkpoint_latest.pt)")
    parser.add_argument("--prompt", type=str, default=None, help="Ad hoc prompt; if omitted, samples prompts from the validation file")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of validation excerpts to test when --prompt is not given")
    parser.add_argument("--prompt_tokens", type=int, default=10, help="Number of tokens taken from each validation excerpt as the seed prompt")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    default_config = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(default_config, OmegaConf.load(args.config)) if args.config else default_config

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = cfg.training.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    temperature = args.temperature if args.temperature is not None else cfg.training.temperature
    top_p = args.top_p if args.top_p is not None else cfg.training.top_p
    min_tokens = cfg.training.min_tokens

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(
        vocab_filepath="vocab.json",
        merges_filepath="merges.txt",
        special_tokens=special_tokens,
    )
    eos_id = tokenizer.inverse_vocab[special_tokens[0].encode("utf-8")]
    actual_vocab_size = max(tokenizer.vocab.keys()) + 1

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TransformerLM(
        d_model=cfg.training.d_model,
        num_heads=cfg.training.num_heads,
        d_ff=cfg.training.d_ff,
        theta=cfg.training.theta,
        context_length=cfg.training.context_len,
        vocab_size=actual_vocab_size,
        num_layers=cfg.training.num_layers,
    ).to(device)

    checkpoint_path = args.checkpoint or os.path.join(cfg.paths.output_dir, "checkpoint_latest.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded checkpoint '{checkpoint_path}' (iteration {checkpoint['iteration']})")

    # ── Decode loop ───────────────────────────────────────────────────────────
    if args.prompt is not None:
        prompt_ids = tokenizer.encode(args.prompt)
        out_ids = generate(
            model, prompt_ids, cfg.training.context_len, args.max_new_tokens,
            temperature, top_p, min_tokens, eos_id, device,
        )
        print("--- prompt ---")
        print(args.prompt)
        print("--- generated ---")
        print(tokenizer.decode(out_ids[len(prompt_ids):]))
    else:
        with open(cfg.paths.input_text_file) as f:
            text = f.read()
        stories = [s.strip() for s in text.split("<|endoftext|>") if s.strip()]
        samples = random.sample(stories, min(args.num_samples, len(stories)))

        for i, story in enumerate(samples):
            story_ids = tokenizer.encode(story)
            prompt_ids = story_ids[:args.prompt_tokens]
            out_ids = generate(
                model, prompt_ids, cfg.training.context_len, args.max_new_tokens,
                temperature, top_p, min_tokens, eos_id, device,
            )
            print(f"\n===== sample {i + 1} =====")
            print(f"--- prompt ---\n{tokenizer.decode(prompt_ids)}")
            print(f"--- generated continuation ---\n{tokenizer.decode(out_ids[len(prompt_ids):])}")
            print(f"--- ground truth continuation ---\n{tokenizer.decode(story_ids[args.prompt_tokens:args.prompt_tokens + args.max_new_tokens])}")
