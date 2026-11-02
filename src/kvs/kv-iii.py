import torch
import torch.nn as nn
import torch.nn.functional as F


class KVCache:
    """Helper class to store and update Keys and Values during generation."""
    
    def __init__(self):
        self.key: torch.Tensor | None = None
        self.value: torch.Tensor | None = None

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Appends new key and value states along the sequence dimension (dim=2).
        
        Shape of key_states / value_states: (batch_size, num_heads, seq_len, head_dim)
        """
        if self.key is None:
            self.key = key_states
            self.value = value_states
        else:
            # Concatenate along the sequence dimension (dim 2)
            self.key = torch.cat([self.key, key_states], dim=2)
            self.value = torch.cat([self.value, value_states], dim=2)

        return self.key, self.value

    def reset(self):
        self.key = None
        self.value = None


class MultiHeadAttentionWithKVCache(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        batch_size, seq_len, _ = x.shape

        # Linear projections
        # Shape: (batch_size, seq_len, num_heads, head_dim) -> transpose to (batch_size, num_heads, seq_len, head_dim)
        queries = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Update or create KV Cache
        if use_cache:
            if kv_cache is None:
                kv_cache = KVCache()
            keys, values = kv_cache.update(keys, values)

        # Query sequence length vs Key/Value sequence length
        q_len = queries.size(2)
        kv_len = keys.size(2)

        # Scaled Dot-Product Attention: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply Causal Masking
        if q_len > 1:
            # Prefill phase (q_len == kv_len)
            mask = torch.triu(torch.full((q_len, kv_len), float("-inf"), device=x.device), diagonal=1)
            scores = scores + mask
        # Note: During decoding (q_len == 1), no mask is needed because 
        # the single query token can attend to all past cached key tokens.

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, values)  # (batch_size, num_heads, q_len, head_dim)

        # Reshape back to (batch_size, q_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        output = self.out_proj(output)

        return output, kv_cache