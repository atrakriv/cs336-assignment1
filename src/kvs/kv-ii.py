import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ============================================================
# KV Cache
# ============================================================

@dataclass
class KVCache:
    key: torch.Tensor
    value: torch.Tensor


# ============================================================
# Causal Self Attention
# ============================================================

class CausalSelfAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: KVCache | None = None,
        use_cache: bool = False,
    ):
        """
        x:
            [B, T_new, D]

        past_kv:
            key:   [B, H, T_cached, D_head]
            value: [B, H, T_cached, D_head]
        """

        B, T_new, D = x.shape

        # ----------------------------------------------------
        # 1. Compute Q, K, V for ONLY the input tokens
        # ----------------------------------------------------

        qkv = self.qkv(x)
        # [B, T_new, 3D]

        qkv = qkv.view(
            B,
            T_new,
            3,
            self.n_heads,
            self.head_dim,
        )

        qkv = qkv.permute(2, 0, 3, 1, 4)
        # [3, B, H, T_new, D_head]

        q, k_new, v_new = qkv[0], qkv[1], qkv[2]

        # each:
        # [B, H, T_new, D_head]

        # ----------------------------------------------------
        # 2. Append new K/V to cache
        # ----------------------------------------------------

        if past_kv is not None:

            k = torch.cat(
                [past_kv.key, k_new],
                dim=2,
            )

            v = torch.cat(
                [past_kv.value, v_new],
                dim=2,
            )

            T_cached = past_kv.key.shape[2]

        else:

            k = k_new
            v = v_new

            T_cached = 0

        T_total = k.shape[2]

        # ----------------------------------------------------
        # 3. Attention scores
        # ----------------------------------------------------

        attn_scores = q @ k.transpose(-2, -1)

        attn_scores = (
            attn_scores / (self.head_dim ** 0.5)
        )

        # [B, H, T_new, T_total]

        # ----------------------------------------------------
        # 4. Causal mask
        #
        # Query positions are:
        #
        # T_cached,
        # T_cached + 1,
        # ...
        #
        # Key positions are:
        #
        # 0, 1, ..., T_total - 1
        # ----------------------------------------------------

        query_positions = torch.arange(
            T_cached,
            T_cached + T_new,
            device=x.device,
        ).view(T_new, 1)

        key_positions = torch.arange(
            T_total,
            device=x.device,
        ).view(1, T_total)

        causal_mask = (
            key_positions <= query_positions
        )

        # [T_new, T_total]

        attn_scores = attn_scores.masked_fill(
            ~causal_mask.view(
                1,
                1,
                T_new,
                T_total,
            ),
            float("-inf"),
        )

        # ----------------------------------------------------
        # 5. Softmax
        # ----------------------------------------------------

        attn_probs = F.softmax(
            attn_scores,
            dim=-1,
        )

        # ----------------------------------------------------
        # 6. Weighted sum of values
        # ----------------------------------------------------

        out = attn_probs @ v

        # [B, H, T_new, D_head]

        # ----------------------------------------------------
        # 7. Merge heads
        # ----------------------------------------------------

        out = out.transpose(1, 2).contiguous()

        # [B, T_new, H, D_head]

        out = out.view(
            B,
            T_new,
            D,
        )

        out = self.out_proj(out)

        # ----------------------------------------------------
        # 8. Return updated cache
        # ----------------------------------------------------

        new_kv = None

        if use_cache:
            new_kv = KVCache(
                key=k,
                value=v,
            )

        return out, new_kv


# ============================================================
# Transformer Block
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: int = 4,
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)

        self.attn = CausalSelfAttention(
            d_model,
            n_heads,
        )

        self.ln2 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(
                d_model,
                mlp_ratio * d_model,
            ),
            nn.GELU(),
            nn.Linear(
                mlp_ratio * d_model,
                d_model,
            ),
        )

    def forward(
        self,
        x,
        past_kv=None,
        use_cache=False,
    ):

        attn_out, new_kv = self.attn(
            self.ln1(x),
            past_kv=past_kv,
            use_cache=use_cache,
        )

        x = x + attn_out

        x = x + self.mlp(
            self.ln2(x)
        )

        return x, new_kv


# ============================================================
# Tiny GPT
# ============================================================

class TinyGPT(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers

        self.token_emb = nn.Embedding(
            vocab_size,
            d_model,
        )

        self.pos_emb = nn.Embedding(
            max_seq_len,
            d_model,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model,
                n_heads,
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False,
        )

    def forward(
        self,
        input_ids,
        past_kvs=None,
        use_cache=False,
    ):
        """
        input_ids:
            [B, T_new]

        past_kvs:
            one KVCache per transformer layer
        """

        B, T_new = input_ids.shape

        # ----------------------------------------------------
        # Determine current position
        # ----------------------------------------------------

        if past_kvs is None:

            past_len = 0

            past_kvs = [
                None
                for _ in range(self.n_layers)
            ]

        else:

            past_len = (
                past_kvs[0].key.shape[2]
            )

        # ----------------------------------------------------
        # Position IDs
        #
        # Without cache:
        #   0,1,2,3,...
        #
        # With cache of length 5:
        #   next token gets position 5
        # ----------------------------------------------------

        positions = torch.arange(
            past_len,
            past_len + T_new,
            device=input_ids.device,
        )

        positions = positions.unsqueeze(0)

        # ----------------------------------------------------
        # Embeddings
        # ----------------------------------------------------

        x = (
            self.token_emb(input_ids)
            +
            self.pos_emb(positions)
        )

        # ----------------------------------------------------
        # Transformer layers
        # ----------------------------------------------------

        new_past_kvs = []

        for block, past_kv in zip(
            self.blocks,
            past_kvs,
        ):

            x, new_kv = block(
                x,
                past_kv=past_kv,
                use_cache=use_cache,
            )

            if use_cache:
                new_past_kvs.append(new_kv)

        # ----------------------------------------------------
        # LM head
        # ----------------------------------------------------

        x = self.ln_f(x)

        logits = self.lm_head(x)

        if use_cache:
            return logits, new_past_kvs

        return logits, None


# ============================================================
# Generation WITHOUT KV Cache
# ============================================================

@torch.no_grad()
def generate_without_cache(
    model,
    input_ids,
    max_new_tokens,
):

    model.eval()

    generated = input_ids

    for _ in range(max_new_tokens):

        # IMPORTANT:
        #
        # We send the ENTIRE sequence through the model
        # again on every iteration.

        logits, _ = model(
            generated,
            use_cache=False,
        )

        next_token_logits = (
            logits[:, -1, :]
        )

        next_token = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True,
        )

        generated = torch.cat(
            [generated, next_token],
            dim=1,
        )

    return generated


# ============================================================
# Generation WITH KV Cache
# ============================================================

@torch.no_grad()
def generate_with_cache(
    model,
    input_ids,
    max_new_tokens,
):

    model.eval()

    generated = input_ids

    if max_new_tokens == 0:
        return generated

    # --------------------------------------------------------
    # PREFILL
    #
    # Process the entire prompt once.
    #
    # This creates:
    #
    # K_cache
    # V_cache
    #
    # for every transformer layer.
    # --------------------------------------------------------

    logits, past_kvs = model(
        input_ids,
        past_kvs=None,
        use_cache=True,
    )

    next_token_logits = (
        logits[:, -1, :]
    )

    next_token = torch.argmax(
        next_token_logits,
        dim=-1,
        keepdim=True,
    )

    generated = torch.cat(
        [generated, next_token],
        dim=1,
    )

    # --------------------------------------------------------
    # DECODE
    #
    # From now on we send ONLY ONE TOKEN.
    # --------------------------------------------------------

    for _ in range(max_new_tokens - 1):

        logits, past_kvs = model(
            next_token,
            past_kvs=past_kvs,
            use_cache=True,
        )

        next_token_logits = (
            logits[:, -1, :]
        )

        next_token = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True,
        )

        generated = torch.cat(
            [generated, next_token],
            dim=1,
        )

    return generated