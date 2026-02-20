import torch

from typing import List


class KVCache:
    keys: List[torch.Tensor]
    values: List[torch.Tensor]

    def __init__(self, num_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device, dtype):
        self.keys = [torch.empty(batch_size, max_seq_len, n_kv_heads, head_dim, device=device, dtype=dtype) for _ in range(num_layers)]
        self.values = [torch.empty(batch_size, max_seq_len, n_kv_heads, head_dim, device=device, dtype=dtype) for _ in range(num_layers)]
