from __future__ import annotations

import torch
import torch.nn as nn

from hw1 import (
    Embedding,
    Linear,
    RMSNorm,
    RotaryPositionalEmbedding,
    SwiGLU,
    scaled_dot_product_attention,
)


class BidirectionalSelfAttention(nn.Module):
    """Same as MultiHeadSelfAttention in hw1.py, but no causal mask and no KV cache."""
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
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None,
                pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        # pad_mask: [bsz, seq_len] bool, True = real token, False = padding
        Q = self._split_heads(self.q_proj(x))
        K = self._split_heads(self.k_proj(x))
        V = self._split_heads(self.v_proj(x))
        if self.rope is not None:
            assert token_positions is not None
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        # every query may attend to every (non-padding) key
        mask = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device)
        if pad_mask is not None:
            mask = pad_mask[:, None, None, :]  # [bsz, 1, 1, seq_len]; broadcasts over heads & queries
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
        self.attn = BidirectionalSelfAttention(d_model, num_heads, self.rope, device, dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
    def forward(self, x, pad_mask=None):
        token_positions = torch.arange(x.shape[1], device=x.device)
        y = x + self.attn(self.ln1(x), token_positions, pad_mask)
        y = y + self.ffn(self.ln2(y))
        return y

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta, context_length, vocab_size, num_layers, eps: float = 1e-5, device=None):
        super().__init__()
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, context_length, theta)
            for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, eps, device)
    def forward(self, x, pad_mask=None):
        # x: [bsz, seq_len] token ids -> [bsz, seq_len, d_model] contextual embeddings
        x = self.token_embeddings(x)
        for i in range(self.num_layers):
            x = self.layers[i](x, pad_mask)
        return self.ln_final(x)

class EncoderForSequenceClassification(nn.Module):
    """Optional head: mean-pool the encoder output over non-padding tokens, then project to classes."""
    def __init__(self, d_model, num_heads, d_ff, theta, context_length, vocab_size, num_layers, num_classes, eps: float = 1e-5, device=None):
        super().__init__()
        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, theta, context_length, vocab_size, num_layers, eps, device)
        self.classifier = Linear(in_features=d_model, out_features=num_classes)
    def forward(self, x, pad_mask=None):
        h = self.encoder(x, pad_mask)
        if pad_mask is None:
            pooled = h.mean(dim=1)
        else:
            m = pad_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        return self.classifier(pooled)