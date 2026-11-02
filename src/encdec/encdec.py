from __future__ import annotations

import torch
import torch.nn as nn

from hw1 import (
    Linear, Embedding, RMSNorm, SwiGLU, RotaryPositionalEmbedding,
    scaled_dot_product_attention,
)

class MultiHeadAttention(nn.Module):
    """Self-attention (rope given, x_kv=None) or cross-attention (x_kv = encoder output).
    mask: bool, True = may attend; broadcastable to [bsz, heads, q_len, kv_len]."""
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
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, x_kv: torch.Tensor | None = None):
        kv = x if x_kv is None else x_kv
        Q = self._split_heads(self.q_proj(x))
        K = self._split_heads(self.k_proj(kv))
        V = self._split_heads(self.v_proj(kv))
        if self.rope is not None:  # positions are relative to each sequence, so only for self-attention
            Q = self.rope(Q, torch.arange(x.shape[1], device=x.device))
            K = self.rope(K, torch.arange(kv.shape[1], device=x.device))
        if mask is None:
            mask = torch.ones(Q.shape[-2], K.shape[-2], dtype=torch.bool, device=x.device)
        y = scaled_dot_product_attention(Q, K, V, mask)
        return self.output_proj(self._merge_heads(y))
    def _split_heads(self, t):
        bsz, seq_len = t.shape[:-1]
        return t.view(bsz, seq_len, self.num_heads, self.d_head).transpose(1, 2)
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        d_head = d_model // num_heads
        self.ln1 = RMSNorm(d_model, eps, device, dtype)
        self.ln2 = RMSNorm(d_model, eps, device, dtype)
        self.rope = RotaryPositionalEmbedding(theta, d_head, max_seq_len, device)
        self.attn = MultiHeadAttention(d_model, num_heads, self.rope, device, dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
    def forward(self, x, src_mask=None):
        y = x + self.attn(self.ln1(x), src_mask)   # bidirectional: no causal mask
        y = y + self.ffn(self.ln2(y))
        return y

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        d_head = d_model // num_heads
        self.ln1 = RMSNorm(d_model, eps, device, dtype)
        self.ln2 = RMSNorm(d_model, eps, device, dtype)
        self.ln3 = RMSNorm(d_model, eps, device, dtype)
        self.rope = RotaryPositionalEmbedding(theta, d_head, max_seq_len, device)
        self.self_attn = MultiHeadAttention(d_model, num_heads, self.rope, device, dtype)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, None, device, dtype)  # no RoPE
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
    def forward(self, x, memory, causal_mask, cross_mask=None):
        y = x + self.self_attn(self.ln1(x), causal_mask)
        y = y + self.cross_attn(self.ln2(y), cross_mask, x_kv=memory)
        y = y + self.ffn(self.ln3(y))
        return y

class TransformerEncoderDecoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta, context_length, src_vocab_size, tgt_vocab_size,
                 num_encoder_layers, num_decoder_layers, eps: float = 1e-5, device=None):
        super().__init__()
        self.src_embeddings = Embedding(num_embeddings=src_vocab_size, embedding_dim=d_model)
        self.tgt_embeddings = Embedding(num_embeddings=tgt_vocab_size, embedding_dim=d_model)
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, context_length, theta)
            for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, context_length, theta)
            for _ in range(num_decoder_layers)])
        self.ln_enc = RMSNorm(d_model, eps, device)
        self.ln_final = RMSNorm(d_model, eps, device)
        self.lm_head = Linear(in_features=d_model, out_features=tgt_vocab_size)
    def encode(self, src, src_pad_mask=None):
        # src_pad_mask: [bsz, src_len] bool, True = real token
        src_mask = None if src_pad_mask is None else src_pad_mask[:, None, None, :]
        x = self.src_embeddings(src)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return self.ln_enc(x), src_mask
    def decode(self, tgt, memory, cross_mask=None):
        tgt_len = tgt.shape[1]
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=tgt.device))
        x = self.tgt_embeddings(tgt)
        for layer in self.decoder_layers:
            x = layer(x, memory, causal_mask, cross_mask)
        return self.lm_head(self.ln_final(x))
    def forward(self, src, tgt, src_pad_mask=None):
        memory, cross_mask = self.encode(src, src_pad_mask)
        return self.decode(tgt, memory, cross_mask)