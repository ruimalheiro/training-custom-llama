import torch
import torch.nn.functional as F
import json

from torch import nn
from dataclasses import dataclass
from types import SimpleNamespace


@dataclass
class ModelConfig:
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    multiple_of: int
    ffn_dim_multiplier: float
    norm_eps: float
    rope_theta: float
    max_batch_size: int
    max_seq_len: int
    tokenizer: object = None
    vocab_size: int = None
    pad_token_id: int = None
    stop_tokens: object = None
    ignore_index: int = None
    is_moe: bool = False
    moe_num_experts: int = None
    moe_expert_dim: int = None
    moe_top_k: int = None
    moe_load_balancing_coef: float = None
    moe_z_loss_coef: float = None
    moe_compute_stats: bool = False

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=4)

    def to_dict(self):
        return {
            'dim': self.dim,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'n_kv_heads': self.n_kv_heads,
            'vocab_size': self.vocab_size,
            'multiple_of': self.multiple_of,
            'ffn_dim_multiplier': self.ffn_dim_multiplier,
            'norm_eps': self.norm_eps,
            'rope_theta': self.rope_theta,
            'max_batch_size': self.max_batch_size,
            'max_seq_len': self.max_seq_len,
            'pad_token_id': self.pad_token_id,
            'stop_tokens': list(self.stop_tokens),
            'is_moe': self.is_moe,
            'moe_num_experts': self.moe_num_experts,
            'moe_expert_dim': self.moe_expert_dim,
            'moe_top_k': self.moe_top_k,
            'moe_load_balancing_coef': self.moe_load_balancing_coef,
            'moe_z_loss_coef': self.moe_z_loss_coef,
            'moe_compute_stats': self.moe_compute_stats
        }

def precompute_rope_freqs(head_dim, sequence_length, theta=10000.0, device='cpu', dtype=torch.float32):
    ''' Computes the frequencies that will be used for rope (rotary positional ebedding). Without complex numbers.
    '''
    indices = torch.arange(0, head_dim // 2, device=device, dtype=torch.float32)
    normalised_indices = 2.0 * indices / head_dim

    freqs = 1.0 / (theta ** normalised_indices)

    timesteps = torch.arange(sequence_length, device=device, dtype=torch.float32)

    freqs = torch.outer(timesteps, freqs)

    cos = torch.cos(freqs).repeat_interleave(2, dim=-1).to(dtype)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1).to(dtype)

    freqs_sin_cos = torch.stack((cos, sin), dim=-1)

    return freqs_sin_cos

def rotate_half(x):
    ''' 90 degree rotation
    '''
    # get pairs
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    # rotates the pairs like if we multiplied a rotating matrix ([[0, -1], [1, 0]]) by a vector [x, y]
    rotated = torch.stack((-x2, x1), dim=-1)
    return rotated.flatten(-2) # returns flat tensor with all pairs

def apply_rope(xq, xk, freqs):
    ''' Apply the rotary position embeddings. Without complex numbers.
        freqs is expected to have dimension [sequence_length, head_dim, 2]
    '''
    cos = freqs[..., 0][None, :, None, :] # will have dim (1, sequence_length, 1, model_dim)
    sin = freqs[..., 1][None, :, None, :]

    xq = (xq * cos) + rotate_half(xq) * sin
    xk = (xk * cos) + rotate_half(xk) * sin

    return xq, xk

def repeat_kv(x, n_rep):
    ''' Repeat x n_rep times. The idea is to spread these tensors through more attention heads. 
    '''
    bs, slen, n_kv_heads, head_dim = x.shape

    if n_rep == 1:
        return x
        
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        self.head_dim = config.dim // config.n_heads

        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

    def forward(self, x, rope_freqs, mask=None):
        batch_size, sequence_length, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(batch_size, sequence_length, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, sequence_length, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, sequence_length, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rope(xq, xk, freqs=rope_freqs)

        keys = repeat_kv(xk, self.n_heads // self.n_kv_heads)
        values = repeat_kv(xv, self.n_heads // self.n_kv_heads)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        # if mask is not None:
        #     scores = scores + mask
        # scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # output = torch.matmul(scores, values)

        # Torch scaled_dot_product_attention will use flash-attention if possible.
        output = F.scaled_dot_product_attention(
            xq,
            keys,
            values,
            attn_mask=mask,
            is_causal=True
        )

        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, -1)

        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier):
        super(FeedForward, self).__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        a, b = self.w1(x), self.w3(x)
        c = F.silu(a) * b
        return self.w2(c)

class MoEFeedForward(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        multiple_of,
        ffn_dim_multiplier,
        num_experts,
        expert_dim,
        top_k,
        load_balancing_coef,
        z_loss_coef,
        compute_stats=False
    ):
        super(MoEFeedForward, self).__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balancing_coef = load_balancing_coef
        self.z_loss_coef = z_loss_coef
        self.compute_stats = compute_stats

        # stats buffers
        self.register_buffer('acc_top1_counts', torch.zeros(num_experts, dtype=torch.int64), persistent=False)
        self.register_buffer('acc_topk_counts', torch.zeros(num_experts, dtype=torch.int64), persistent=False)
        self.register_buffer('acc_p_sum', torch.zeros(num_experts, dtype=torch.float32), persistent=False)
        self.register_buffer('acc_tokens', torch.zeros((), dtype=torch.int64), persistent=False)

        self.router = nn.Linear(dim, num_experts, bias=False)

        assert expert_dim == dim, 'MoEFeedForward assumes expert_dim == dim.'

        self.experts = nn.ModuleList([
            FeedForward(expert_dim, hidden_dim, multiple_of, ffn_dim_multiplier) for _ in range(num_experts)
        ])

    def reset_stats(self):
        self.acc_top1_counts.zero_()
        self.acc_topk_counts.zero_()
        self.acc_p_sum.zero_()
        self.acc_tokens.zero_()

    def _accumulate_stats(self, top_k_index, topk_counts, p):
        top1 = top_k_index[:, 0]
        n_tokens = top_k_index.size(0)
        top1_counts = torch.bincount(top1, minlength=self.num_experts).to(torch.int64)

        self.acc_top1_counts.add_(top1_counts)
        self.acc_topk_counts.add_(topk_counts)
        self.acc_p_sum.add_(p.to(torch.float32) * n_tokens)
        self.acc_tokens.add_(n_tokens)

    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.reshape(B * S, D)
        y_flat = torch.zeros_like(x_flat)

        logits = self.router(x_flat) # (B*S, E_logits)

        top_k_logits, top_k_index = torch.topk(logits, k=self.top_k, dim=-1) # (B*S, E_logits) / (B*S, E_indexes)

        top_k_probs = F.softmax(top_k_logits, dim=-1, dtype=x_flat.dtype)
        all_probs = F.softmax(logits, dim=-1, dtype=x_flat.dtype)

        # TOKEN LOAD BALANCE
        # f
        total_token_count_possible = B * S * self.top_k

        topk_counts_i64 = torch.bincount(top_k_index.reshape(-1), minlength=self.num_experts).to(torch.int64)
        counts_per_top_k_expert = topk_counts_i64.to(torch.float32)
        f = counts_per_top_k_expert / max(1.0, total_token_count_possible)
        # p
        p = all_probs.mean(dim=0)
        load_balance_loss = self.load_balancing_coef * self.num_experts * torch.sum(f * p)

        # Z LOSS
        z_loss = self.z_loss_coef * torch.mean(torch.logsumexp(logits, dim=-1) ** 2.0)

        aux_loss = load_balance_loss + z_loss

        # COMPUTE STATS
        if self.compute_stats:
            with torch.no_grad():
                self._accumulate_stats(top_k_index, topk_counts_i64, p)

        # PERFORMANCE
        flat_top_k_index = top_k_index.reshape(-1)
        flat_top_k_probs = top_k_probs.reshape(-1)
        flat_token = torch.arange(B * S, device=x_flat.device).repeat_interleave(self.top_k)

        order = torch.argsort(flat_top_k_index)
        flat_top_k_index = flat_top_k_index.index_select(0, order)
        flat_top_k_probs = flat_top_k_probs.index_select(0, order)
        flat_token = flat_token.index_select(0, order)

        counts = counts_per_top_k_expert.int()
        offsets = torch.cumsum(counts, dim=0)
        starts = torch.empty_like(offsets)
        starts[0] = 0
        starts[1:] = offsets[:-1]

        for i in range(self.num_experts):
            s_i = starts[i]
            e_i = offsets[i]
            if s_i == e_i:
                if torch.is_grad_enabled():
                    for p in self.experts[i].parameters(): # hack so the autograd graph considers the expert (otherwise error because of None grad)
                        aux_loss = aux_loss + p.sum() * 0.0
                continue

            token_indexes = flat_token[s_i:e_i]

            tokens_for_expert_e = x_flat.index_select(0, token_indexes)
            expert_activations = self.experts[i](tokens_for_expert_e)
            gate_activations = flat_top_k_probs[s_i:e_i].unsqueeze(-1)

            activations = expert_activations * gate_activations
            y_flat.index_add_(0, token_indexes, activations)

        y_flat = y_flat.reshape(B, S, D)

        return y_flat, aux_loss

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x) * self.weight

class TransformerBlock(nn.Module):
    def __init__(self, layer_id, config):
        super(TransformerBlock, self).__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.attention = Attention(config)
        self.is_moe = config.is_moe

        if config.is_moe:
            self.feed_forward = MoEFeedForward(
                dim=config.dim,
                hidden_dim=4 * config.dim,
                multiple_of=config.multiple_of,
                ffn_dim_multiplier=config.ffn_dim_multiplier,
                num_experts=config.moe_num_experts,
                expert_dim=config.moe_expert_dim,
                top_k=config.moe_top_k,
                load_balancing_coef=config.moe_load_balancing_coef,
                z_loss_coef=config.moe_z_loss_coef
            )
        else:
            self.feed_forward = FeedForward(
                dim=config.dim,
                hidden_dim=4 * config.dim,
                multiple_of=config.multiple_of,
                ffn_dim_multiplier=config.ffn_dim_multiplier
            )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(self, x, rope_freqs, mask=None):
        hidden_state = x + self.attention(self.attention_norm(x), rope_freqs, mask)
        if self.is_moe:
            ff, aux = self.feed_forward(self.ffn_norm(hidden_state))
            output = hidden_state + ff
        else:
            output = hidden_state + self.feed_forward(self.ffn_norm(hidden_state))
            aux = None
        return output, aux

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()

        self.config = config

        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        self.tok_embeddings = nn.Embedding(
            config.vocab_size,
            config.dim,
            padding_idx=config.pad_token_id
        )

        self.layers = nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_id, config))

        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.ignore_index = config.ignore_index

        rope_freqs = precompute_rope_freqs(
            config.dim // config.n_heads,
            config.max_seq_len * 2,
            config.rope_theta
        )

        self.register_buffer('rope_freqs', rope_freqs, persistent=False)

    def build_optimizer_param_groups(self, weight_decay, lora_weight_decay=0.0, freeze_non_lora=True):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        lora_names = [n for n in param_dict if n.endswith('.A') or n.endswith('.B')]
        lora_params = [param_dict[n] for n in lora_names]

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2 and n not in lora_names]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2 and n not in lora_names]

        # disable optimization for other params if lora is set:
        if lora_params:
            if freeze_non_lora:
                for p in decay_params + nodecay_params:
                    p.requires_grad_(False)

            optimizer_groups = [
                {'params': lora_params, 'weight_decay': lora_weight_decay}
            ]
        else:
            optimizer_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        num_lora_params = sum(p.numel() for p in lora_params)

        total_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return SimpleNamespace(
            optimizer_groups=optimizer_groups,
            decay_params=decay_params,
            num_decay_params=num_decay_params,
            nodecay_params=nodecay_params,
            num_nodecay_params=num_nodecay_params,
            lora_params=lora_params,
            num_lora_params=num_lora_params,
            total_trainable_params=total_trainable_params
        )

    def set_moe_stats(self, enabled):
        for m in self.modules():
            if isinstance(m, MoEFeedForward):
                m.compute_stats = enabled

    def enable_moe_stats(self):
        if self.config.is_moe and self.config.moe_compute_stats:
            self.set_moe_stats(True)

    def disable_moe_stats(self):
        if self.config.is_moe and self.config.moe_compute_stats:
            self.set_moe_stats(False)

    def reset_moe_stats(self):
        if self.config.is_moe and self.config.moe_compute_stats:
            for m in self.modules():
                if isinstance(m, MoEFeedForward):
                    m.reset_stats()

    def get_moe_stats(self):
        if not self.config.is_moe or (not self.config.moe_compute_stats):
            return []
        stats = []
        for layer_id, block in enumerate(self.layers):
            if isinstance(block.feed_forward, MoEFeedForward):
                stats.append((layer_id, block.feed_forward))
        return stats

    def forward(
        self,
        input_ids,
        attention_mask=None,
        start_position=0,
        labels=None
    ):
        batch_size, sequence_length = input_ids.shape

        hidden_state = self.tok_embeddings(input_ids)

        rope_freqs = self.rope_freqs
        if rope_freqs.device != hidden_state.device or rope_freqs.dtype != hidden_state.dtype:
            rope_freqs = rope_freqs.to(
                dtype=hidden_state.dtype,
                device=hidden_state.device
            )

        rope_freqs = rope_freqs[start_position : start_position + sequence_length]

        mask = None
        if attention_mask is not None:
            attention_mask = attention_mask.to(hidden_state.device)
            if (attention_mask != 1).any():
                # only create mask if attention_mask is passed
                # if following HF convention, mask is expected to have 1s in the valid tokens and 0s in the masked positions.
                masked_positions = ~attention_mask.bool() # because torch flash attention expects true for masked.
                mask = masked_positions[:, None, None, :] # torch flash attention expects dims [B, H, Q, K]

        aux_loss_total = hidden_state.new_zeros(())
        for layer in self.layers:
            hidden_state, aux = layer(hidden_state, rope_freqs, mask)
            if aux is not None:
                aux_loss_total = aux_loss_total + aux

        hidden_state = self.norm(hidden_state)

        logits = self.output(hidden_state)

        loss = None
        if labels is not None:
            if labels.device != hidden_state.device:
                labels = labels.to(hidden_state.device)
            flat_logits = logits.view(-1, logits.size(-1))  # (batch_size * sequence_length, num_classes)
            flat_labels = labels.view(-1)  # (batch_size * sequence_length)
            ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index)(flat_logits, flat_labels)
            loss = ce + (aux_loss_total / self.n_layers)

        return SimpleNamespace(logits=logits, loss=loss)
