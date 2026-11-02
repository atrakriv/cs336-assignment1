"""
KV caching from scratch (PyTorch).

A tiny GPT-style decoder (pre-norm, learned positions, multi-head causal
attention, GELU MLP) with two generation loops:

  generate_naive   - re-runs the whole sequence through the model every step.
  generate_cached  - prefill the prompt once, then feed ONE token per step and
                     reuse the keys/values stored in a KVCache.

Both produce identical outputs. Per-step cost drops from
O(t * d^2 + t^2 * d) to O(d^2 + t * d).
"""
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False   # keep cached/uncached numerics tight
torch.backends.cudnn.allow_tf32 = False


@dataclass
class Config:
    vocab: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    max_len: int = 512

    @property
    def d_head(self):
        return self.d_model // self.n_heads


# --------------------------------------------------------------------------- #
# KV cache
# --------------------------------------------------------------------------- #
class KVCache:
    """Preallocated per-layer key/value buffers.

    Layout: (n_layers, batch, n_heads, max_len, d_head). `length` is the number
    of positions already committed. During one forward pass every layer writes
    its new K/V at [length : length+T]; `advance(T)` is called once at the end.
    """

    def __init__(self, cfg: Config, batch=1, device="cpu", dtype=torch.float32):
        shape = (cfg.n_layers, batch, cfg.n_heads, cfg.max_len, cfg.d_head)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.length = 0
        self.max_len = cfg.max_len

    def update(self, layer, k_new, v_new):
        """k_new, v_new: (B, H, T, Dh). Returns K, V covering all positions up
        to and including the new ones (views, no copy)."""
        T = k_new.shape[2]
        s, e = self.length, self.length + T
        if e > self.max_len:
            raise ValueError(f"KV cache overflow: {e} > {self.max_len}")
        self.k[layer, :, :, s:e] = k_new
        self.v[layer, :, :, s:e] = v_new
        return self.k[layer, :, :, :e], self.v[layer, :, :, :e]

    def advance(self, T):
        self.length += T

    def reset(self):
        self.length = 0

    @property
    def nbytes(self):
        return self.k.numel() * self.k.element_size() * 2


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.h, self.dh = cfg.n_heads, cfg.d_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, layer_idx, cache, start):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        # (B, T, D) -> (B, H, T, Dh)
        q, k, v = (t.view(B, T, self.h, self.dh).transpose(1, 2) for t in (q, k, v))

        if cache is not None:
            k, v = cache.update(layer_idx, k, v)        # (B, H, start+T, Dh)

        S = k.shape[2]
        # Query i sits at absolute position start+i and may attend to keys
        # j <= start+i. With T == 1 this masks nothing.
        qpos = start + torch.arange(T, device=x.device)[:, None]
        kpos = torch.arange(S, device=x.device)[None, :]
        mask = kpos <= qpos                              # (T, S) bool, True = attend

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x, layer_idx, cache, start):
        x = x + self.attn(self.ln1(x), layer_idx, cache, start)
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, cache=None):
        """idx: (B, T) NEW token ids. Returns logits (B, T, vocab).
        Without a cache, `idx` must be the full sequence."""
        B, T = idx.shape
        start = cache.length if cache is not None else 0
        if start + T > self.cfg.max_len:
            raise ValueError("sequence exceeds max_len")

        pos = torch.arange(start, start + T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        for i, blk in enumerate(self.blocks):
            x = blk(x, i, cache, start)
        if cache is not None:
            cache.advance(T)
        return self.ln_f(x) @ self.tok.weight.T          # weight tying


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def pick(logits, temperature, gen):
    """logits: (B, vocab) -> (B, 1) next-token ids."""
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1, generator=gen)


@torch.no_grad()
def generate_naive(model, prompt, n_new, temperature=0.0, seed=0):
    """Reprocesses the entire sequence at every step."""
    gen = torch.Generator(device=prompt.device).manual_seed(seed)
    seq = prompt
    for _ in range(n_new):
        logits = model(seq)[:, -1]
        seq = torch.cat([seq, pick(logits, temperature, gen)], dim=1)
    return seq


@torch.no_grad()
def generate_cached(model, prompt, n_new, temperature=0.0, seed=0, cache=None):
    """Prefill once, then decode one token at a time using the KV cache."""
    gen = torch.Generator(device=prompt.device).manual_seed(seed)
    cache = cache or KVCache(model.cfg, prompt.shape[0], prompt.device)
    cache.reset()
    seq = prompt
    logits = model(seq, cache)[:, -1]                    # prefill
    for _ in range(n_new):
        nxt = pick(logits, temperature, gen)
        seq = torch.cat([seq, nxt], dim=1)
        logits = model(nxt, cache)[:, -1]                # decode: 1 token
    return seq


# --------------------------------------------------------------------------- #
# Verification + benchmark
# --------------------------------------------------------------------------- #
def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, device):
    sync(device)
    t0 = time.perf_counter()
    out = fn()
    sync(device)
    return out, time.perf_counter() - t0


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    model = TinyGPT(cfg).to(device).eval()
    B, n_new = 1, 200
    prompt = torch.randint(0, cfg.vocab, (B, 16), device=device)
    print(f"device: {device}")

    with torch.no_grad():
        # 1) Logit equivalence: full pass vs chunked prefill + token-by-token.
        seq = torch.randint(0, cfg.vocab, (B, 40), device=device)
        full = model(seq)
        cache = KVCache(cfg, B, device)
        parts = [model(seq[:, :25], cache)]              # prefill chunk
        for t in range(25, 40):
            parts.append(model(seq[:, t:t + 1], cache))  # one at a time
        inc = torch.cat(parts, dim=1)
        diff = (full - inc).abs().max().item()
        print(f"max |logit diff| full vs cached: {diff:.2e}")
        assert torch.allclose(full, inc, atol=1e-5), "cached logits diverge"

    # 2) Identical generations (greedy and sampled with the same seed).
    for temp in (0.0, 0.8):
        a = generate_naive(model, prompt, n_new, temp, seed=3)
        b = generate_cached(model, prompt, n_new, temp, seed=3)
        assert torch.equal(a, b), f"mismatch at temperature={temp}"
        print(f"temperature={temp}: naive == cached  ({a.shape[1]} tokens)")

    # 3) Timing (one warm-up each so allocator/kernels don't skew results).
    generate_naive(model, prompt, 5)
    generate_cached(model, prompt, 5)
    _, t_naive = timed(lambda: generate_naive(model, prompt, n_new), device)
    _, t_cache = timed(lambda: generate_cached(model, prompt, n_new), device)
    print(f"\nnaive : {t_naive:6.2f}s  ({n_new / t_naive:7.1f} tok/s)")
    print(f"cached: {t_cache:6.2f}s  ({n_new / t_cache:7.1f} tok/s)")
    print(f"speedup: {t_naive / t_cache:.1f}x")

    # 4) Work done: token-positions pushed through the network.
    L = prompt.shape[1]
    naive_work = sum(L + i for i in range(n_new))
    cached_work = L + n_new
    print(f"\ntoken-positions processed: naive={naive_work}, cached={cached_work} "
          f"({naive_work / cached_work:.0f}x fewer)")
    mb = KVCache(cfg, B, "cpu").nbytes / 1e6
    print(f"cache memory: {mb:.1f} MB (2 * layers * batch * heads * max_len * d_head * 4B)")


if __name__ == "__main__":
    main()