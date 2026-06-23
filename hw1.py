from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import os
import re
import regex
import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from multiprocessing import Pool

import numpy as np

import math
import torch
import torch.nn as nn


class BPETrainer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self, vocab_size: int, special_tokens: list[str], num_processes: int = os.cpu_count() or 1):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.num_processes = num_processes

    @staticmethod
    def _find_chunk_boundaries(file, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
        """Chunk the file into parts aligned to special token boundaries."""
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096
        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)
            while True:
                mini_chunk = file.read(mini_chunk_size)
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        return sorted(set(chunk_boundaries))

    @staticmethod
    def _pretokenize_chunk(args: tuple) -> dict:
        """Worker: read one file chunk and return its word_counts dict."""
        input_path, start, end, special_tokens, pat = args
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

        if special_tokens:
            split_pat = '|'.join(re.escape(st) for st in sorted(special_tokens, key=len, reverse=True))
            parts = re.split(split_pat, chunk)
        else:
            parts = [chunk]

        word_counts: dict = defaultdict(int)
        for part in parts:
            for token in regex.findall(pat, part):
                word = tuple(bytes([b]) for b in token.encode('utf-8'))
                word_counts[word] += 1
        return dict(word_counts)

    def train(self, input_path: str) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        split_token = self.special_tokens[0].encode('utf-8') if self.special_tokens else b"\n"

        with open(input_path, "rb") as f:
            boundaries = self._find_chunk_boundaries(f, self.num_processes, split_token)

        chunk_args = [
            (input_path, start, end, self.special_tokens, self.PAT)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]

        with Pool(processes=len(chunk_args)) as pool:
            chunk_counts = pool.map(self._pretokenize_chunk, chunk_args)

        # Merge word counts from all chunks
        word_counts: dict = defaultdict(int)
        for counts in chunk_counts:
            for word, count in counts.items():
                word_counts[word] += count

        # Initialize vocab: indices 0-255 are single bytes, then special tokens
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, st in enumerate(self.special_tokens):
            vocab[256 + i] = st.encode('utf-8')

        # Build initial pair counts
        pair_counts: dict = defaultdict(int)
        for word, count in word_counts.items():
            for i in range(len(word) - 1):
                pair_counts[(word[i], word[i + 1])] += count

        num_merges = self.vocab_size - 256 - len(self.special_tokens)
        merges: list[tuple[bytes, bytes]] = []
        next_id = 256 + len(self.special_tokens)

        for _ in range(num_merges):
            if not pair_counts:
                break

            # Pick best pair: highest count, break ties lexicographically
            best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))

            new_token = best_pair[0] + best_pair[1]
            merges.append(best_pair)
            vocab[next_id] = new_token
            next_id += 1

            # Merge best_pair in every word, updating pair counts incrementally
            new_words: dict = {}
            for word, count in word_counts.items():
                if best_pair[0] not in word:
                    new_words[word] = new_words.get(word, 0) + count
                    continue

                new_word: list = []
                i = 0
                while i < len(word):
                    if (i + 1 < len(word)
                            and word[i] == best_pair[0]
                            and word[i + 1] == best_pair[1]):
                        # Update left-neighbor pair
                        if new_word:
                            pair_counts[(new_word[-1], best_pair[0])] -= count
                            pair_counts[(new_word[-1], new_token)] += count
                        # Update right-neighbor pair (crosses the merge boundary)
                        if i + 2 < len(word):
                            pair_counts[(best_pair[1], word[i + 2])] -= count
                            pair_counts[(new_token, word[i + 2])] += count
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1

                t = tuple(new_word)
                new_words[t] = new_words.get(t, 0) + count

            del pair_counts[best_pair]
            word_counts = new_words

        return vocab, merges

class Tokenizer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=None):
        self.vocab = vocab
        self.inverse_vocab: dict[bytes, int] = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens: list[str] = special_tokens or []
        self.special_token_set: set[str] = set(self.special_tokens)
        # Build merge rank lookup: (bytes, bytes) -> rank (lower = applied first)
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {pair: i for i, pair in enumerate(merges)}
        # Cache of pretoken bytes -> encoded ids, since the same word recurs constantly in natural text
        self._bpe_cache: dict[bytes, list[int]] = {}
        # Ensure special tokens are in vocab
        for st in self.special_tokens:
            st_bytes = st.encode('utf-8')
            if st_bytes not in self.inverse_vocab:
                next_id = max(self.vocab.keys()) + 1
                self.vocab[next_id] = st_bytes
                self.inverse_vocab[st_bytes] = next_id

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath) as f:
            vocab_data = json.load(f)
        vocab = {int(k): bytes(v) for k, v in vocab_data.items()}
        merges = None
        with open(merges_filepath) as f:
            merges = [(bytes(a), bytes(b)) for a, b in json.load(f)]
        return cls(vocab, merges, special_tokens)

    def _bpe_encode(self, token_bytes: bytes) -> list[int]:
        cached = self._bpe_cache.get(token_bytes)
        if cached is not None:
            return cached

        pieces = [bytes([b]) for b in token_bytes]
        while len(pieces) > 1:
            best_rank = float('inf')
            best_idx = -1
            for i in range(len(pieces) - 1):
                rank = self.merge_ranks.get((pieces[i], pieces[i + 1]), float('inf'))
                if rank < best_rank:
                    best_rank = rank
                    best_idx = i
            if best_idx == -1:
                break
            pieces = pieces[:best_idx] + [pieces[best_idx] + pieces[best_idx + 1]] + pieces[best_idx + 2:]

        ids = [self.inverse_vocab[p] for p in pieces]
        self._bpe_cache[token_bytes] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        ids: list[int] = []
        if self.special_tokens:
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = '(' + '|'.join(re.escape(st) for st in sorted_special) + ')'
            parts = re.split(special_pattern, text)
        else:
            parts = [text]
        for part in parts:
            if not part:
                continue
            if part in self.special_token_set:
                ids.append(self.inverse_vocab[part.encode('utf-8')])
            else:
                for token in regex.findall(self.PAT, part):
                    ids.extend(self._bpe_encode(token.encode('utf-8')))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    @staticmethod
    def _encode_chunk(args: tuple) -> np.ndarray:
        """Worker: read one file chunk and encode it to token ids."""
        tokenizer, input_path, start, end = args
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk_text = f.read(end - start).decode("utf-8", errors="ignore")
        return np.array(tokenizer.encode(chunk_text), dtype=np.int32)

    def encode_file_parallel(self, input_path: str, num_processes: int = os.cpu_count() or 1) -> np.ndarray:
        """Encode a large file in parallel, splitting on special-token boundaries to preserve order."""
        split_token = self.special_tokens[0].encode('utf-8') if self.special_tokens else b"\n"
        with open(input_path, "rb") as f:
            boundaries = BPETrainer._find_chunk_boundaries(f, num_processes, split_token)

        chunk_args = [
            (self, input_path, start, end)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]

        with Pool(processes=len(chunk_args)) as pool:
            chunk_arrays = pool.map(self._encode_chunk, chunk_args)

        return np.concatenate(chunk_arrays)

    def decode(self, ids: list[int]) -> str:
        return b''.join(self.vocab[i] for i in ids).decode('utf-8', errors='replace')

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weight =nn.Parameter(torch.empty(out_features, in_features, device=device, dtype = dtype))
        std = math.sqrt(2/(in_features+out_features))
        a, b = -3*std, 3*std
        nn.init.trunc_normal_(self.weight, mean=0., std=std, a=a, b=b)
    def forward(self, x: torch.Tensor):
        return x@self.weight.T
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, **factory_kwargs))
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.rsqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * rms * self.weight).to(in_dtype)
    
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.w1 = Linear(d_model, d_ff, **factory_kwargs)
        self.w3 = Linear(d_model, d_ff, **factory_kwargs)
        self.w2 = Linear(d_ff, d_model, **factory_kwargs)

    def forward(self, x):
        swish = self.w1(x) * torch.sigmoid(self.w1(x))
        gate = self.w3(x)
        return self.w2(swish * gate)
        #return self.w2(F.silu(self.w1(x)) * self.w3(x))
    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        inv_freq = 1./theta**(torch.arange(0, d_k, 2, device=device)/d_k)
        ps = torch.arange(max_seq_len, device=device)
        angs = torch.outer(ps, inv_freq)
        self.d_k = d_k
        self.register_buffer('cos', torch.cos(angs))   # [..., max_seq_len, d_k//2]
        self.register_buffer('sin', torch.sin(angs))   # [..., max_seq_len, d_k//2]
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        token_positions = token_positions.to(x.device) # defensive practice
        sin = self.sin[token_positions]                # [..., seq_len, d_k//2]
        cos = self.cos[token_positions]                # [..., seq_len, d_k//2]
        x_even, x_odd = x[...,0::2], x[...,1::2]
        rope = torch.empty_like(x)
        rope[...,0::2], rope[...,1::2] = x_even * cos - x_odd * sin, x_odd * cos + x_even * sin
        return rope
    
def softmax(x: torch.Tensor, dim: int):
    x_max = x.max(dim=dim, keepdim=True).values
    exp = torch.exp(x - x_max)
    return exp / exp.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(Q, K, V, mask):
    d_k = Q.shape[-1]
    KT = K.transpose(-2,-1)
    S = Q@KT/math.sqrt(d_k)
    S = S.masked_fill(~mask, float('-inf'))
    A = softmax(S, dim = -1)
    return A@V

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope=None, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, **factory_kwargs)
        self.k_proj = Linear(d_model, d_model, **factory_kwargs)
        self.v_proj = Linear(d_model, d_model, **factory_kwargs)
        self.output_proj = Linear(d_model, d_model, **factory_kwargs)
        self.rope = rope
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        seq_len = x.shape[1]
        Q = self._split_heads(self.q_proj(x))
        K = self._split_heads(self.k_proj(x))
        if self.rope is not None:
            assert token_positions is not None
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        V = self._split_heads(self.v_proj(x))
        mask = torch.tril(torch.ones(1, 1, seq_len, seq_len, device=x.device)).bool()
        y = scaled_dot_product_attention(Q, K, V, mask)
        return self.output_proj(self._merge_heads(y))
    def _split_heads(self, t):
        bsz, seq_len = t.shape[:-1]
        return t.view(bsz, seq_len, self.num_heads, self.d_head).transpose(1, 2)
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        d_head = d_model // num_heads
        self.ln1 = RMSNorm(d_model, eps, device, dtype)
        self.ln2 = RMSNorm(d_model, eps, device, dtype) 
        self.rope = RotaryPositionalEmbedding(theta, d_head, max_seq_len, device)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, self.rope, device, dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
    def forward(self, x):
        token_positions = torch.arange(x.shape[1])
        y = x + self.attn(self.ln1(x), token_positions)
        y = y + self.ffn(self.ln2(y))
        return y

class TransformerLM(nn.Module): # are context_length and max_seq_len the same thing?
    def __init__(self, d_model, num_heads, d_ff, theta, context_length, vocab_size, num_layers, eps: float = 1e-5, device=None):
        super().__init__()
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, theta) 
            for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, eps, device)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size)
    def forward(self, x):
        x = self.token_embeddings(x)
        for i in range(self.num_layers):
            x = self.layers[i](x)
        x = self.ln_final(x)
        x = self.lm_head (x)
        return x
    
# use for hw only
def cross_entropy_(inputs, targets):
    logits = inputs - inputs.max(dim=-1, keepdim=True).values
    # also can use torch.logsumexp
    log_probs = logits - torch.log(torch.sum(torch.exp(logits), dim=-1, keepdim=True)) # B*T*V
    return -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean()   # squeeze(-1) can be removed

def cross_entropy(logits, targets):
    x = torch.log_softmax(logits, dim=-1)
    logps = x.gather(index=targets.unsqueeze(dim=-1), dim=-1)
    return -logps.mean()
 
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, weight_decay, eps):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps":eps}
        super().__init__(params, defaults)
    # my version: no in-place m & v update 
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None: continue
                state = self.state[p]  # Get state associated with p
                g = p.grad.data        # Get the gradient of loss with respect to p
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                m = b1*m + (1-b1)*g
                v = b2*v + (1-b2)*g**2
                t = state.get("t", 0) + 1
                alpha = lr*(math.sqrt(1-b2**t))/(1-b1**t)
                p.data -= alpha*m/(v.sqrt()+eps)
                p.data -= lr*weight_decay*p.data 
                state["m"] = m
                state["v"] = v
                state["t"] = t
        return loss
    # optimized version: in-place m & v update [no need to state["m/v"] = m/v]
    # def step_(self, closure: Optional[Callable] = None):
    #     loss = None if closure is None else closure()
    #     for group in self.param_groups:
    #         lr = group["lr"]
    #         beta1, beta2 = group["betas"]
    #         weight_decay = group["weight_decay"]
    #         eps = group["eps"]
    #         for p in group["params"]:
    #             if p.grad is None: continue
    #             grad = p.grad.data
    #             state = self.state[p]
    #             if len(state) == 0:
    #                 state["t"] = 0
    #                 state["m"] = torch.zeros_like(p.data)
    #                 state["v"] = torch.zeros_like(p.data)
    #             m = state["m"]
    #             v = state["v"]
    #             state["t"] += 1
    #             t = state["t"]
    #             m.mul_(beta1).add_(grad, alpha=1 - beta1)
    #             v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    #             step_size = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
    #             p.data.addcdiv_(m, v.sqrt().add_(eps), value=-step_size)
    #             p.data.add_(p.data, alpha=-lr * weight_decay)
    #     return loss

def get_cosine_lr(
    n: int,
    max_lr: float,
    min_lr: float,
    tw: int,
    tc: int,
):
    if n < tw: return max_lr * n / tw
    if n > tc: return min_lr
    decay_ratio = (n - tw) / (tc - tw)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def clip_gradient(params, max_norm):
    gs = [p.grad for p in params if p.grad is not None]
    norm = torch.sqrt(sum((g**2).sum() for g in gs))
    clip_coef = min(1.0, max_norm / (norm + 1e-6))
    for g in gs:
        g.mul_(clip_coef)
    

def data_loader(dataset, batch_size: int, context_length: int, device: str):
    smpls = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    # Broadcasting addition: a[:, None] + b[None, :]
    indices = smpls[:, None] + np.arange(context_length + 1)[None, :]  # (batch_size, context_length+1)
    chunk = dataset[indices]                                           # (batch_size, context_length+1)
    x = torch.tensor(chunk[:, :-1], dtype=torch.long, device=device)
    y = torch.tensor(chunk[:, 1:], dtype=torch.long, device=device)
    return x, y

def save_checkpoint(model, optimizer, iteration, out):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }, out)
    # no return

def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
