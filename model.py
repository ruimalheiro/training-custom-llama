import math
import torch
import torch.nn.functional as F
import json
import inspect

from torch import nn
from dataclasses import dataclass
from collections import defaultdict


def precompute_freqs_complex_exponential(dim, sequence_length, theta=10000.0):
    ''' Computes the frequencies that will be used for positional encoding and also for rotary embeddings in
    the attention mechanism.
    '''
    # Get the even indices within the embedding dimension and normalises them.
    even_indices = torch.arange(0, dim, 2)[: (dim // 2)].float()
    normalised_even_indices = even_indices / dim

    # Formula for the frequencies.
    freqs = 1.0 / (theta ** normalised_even_indices)

    # Gets an increasing sequence to the size of the input sequence (time steps).
    timesteps = torch.arange(sequence_length, device=freqs.device, dtype=torch.float32)

    # Multiplies each timestep for all values in frequencies to form the frequencies matrix.
    # These will be the angles for the polar function.
    freqs = torch.outer(timesteps, freqs)

    # Creates a mask filled with ones.
    ones = torch.ones_like(freqs)

    # Computes the complex tensor representing the cartesian coordinates that correspond to the polar coordinates (abs "ones" and angles "freqs").
    freqs_complex_exponential = torch.polar(ones, freqs)

    return freqs_complex_exponential

def reshape_for_broadcast(freqs_cis, ndim, shape):
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq, xk, freqs_cis):
    ''' Apply the rotary embeddings.
    '''
    # We start by reshaping the inputs. Their last dimension is the head_dim, so we need to make sure we split the head dim into 2 parts
    # to account for the complex part.
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Ensure freqs_cis has the correct dimensions compatible with broadcasting. E.g (a, 1, b, c, 1)
    # NOTE that xq has shape (batch_size, sequence_length, n_heads, head_dim). Lets assume batch_size = 6, sequence_length = 512, n_heads = 32 and model_dim = 4096 then
    # it will be (6, 512, 32, 4096 / 32) -> (6, 512, 32, 128). xq_complex will be (6, 512, 32, 64, 2). The reason 128 becomes 64, 2 is because each complex number has the real and the imaginary part.
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_complex.ndim, xq_complex.shape)

    # Now we can apply the rotary embeddings and flatten from dimension 3 (so we get the 128 back with 4 dimensions instead of 5.
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(3)

    # Retain the datatypes
    return xq_out.type_as(xq), xk_out.type_as(xk)

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

    def forward(self, x, start_pos, freqs_cis, mask=None):
        batch_size, sequence_length, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(batch_size, sequence_length, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, sequence_length, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, sequence_length, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

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
        output = F.scaled_dot_product_attention(xq, keys, values, scale=(1 / math.sqrt(self.head_dim)), attn_mask=mask)

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
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, layer_id, config):
        super(TransformerBlock, self).__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.attention = Attention(config)
        self.feed_forward = FeedForward(
            dim=config.dim,
            hidden_dim=4 * config.dim,
            multiple_of=config.multiple_of,
            ffn_dim_multiplier=config.ffn_dim_multiplier
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(self, x, start_position, freqs_cis, mask=None):
        hidden_state = x + self.attention(self.attention_norm(x), start_position, freqs_cis, mask)
        output = hidden_state + self.feed_forward(self.ffn_norm(hidden_state))
        return output

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

        self.freqs_cis = precompute_freqs_complex_exponential(
            config.dim // config.n_heads,
            config.max_seq_len * 2,
            config.rope_theta
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        start_position=0,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ):
        ''' Some arguments in the method signature are not used but are required to make this compatible with the trainer from Hugging Face.
        '''
        batch_size, sequence_length = input_ids.shape

        hidden_state = self.tok_embeddings(input_ids)

        if self.freqs_cis.device != hidden_state.device:
            self.freqs_cis = self.freqs_cis.to(hidden_state.device)

        freqs_cis = self.freqs_cis[start_position : start_position + sequence_length]

        mask = None

        if sequence_length > 1:
            mask = torch.full((sequence_length, sequence_length), float('-inf'), device=input_ids.device)
            mask = torch.triu(mask, diagonal=1)

            cached_shift = torch.zeros((sequence_length, start_position), device = input_ids.device)
            mask = torch.hstack([cached_shift, mask]).type_as(hidden_state) 

        for layer in self.layers:
            hidden_state = layer(hidden_state, start_position, freqs_cis, mask)

        hidden_state = self.norm(hidden_state)

        logits = self.output(hidden_state).float()

        loss = None
        if labels is not None:
            logits = logits.view(-1, logits.size(-1))  # (batch_size * sequence_length, num_classes)
            labels = labels.view(-1)  # (batch_size * sequence_length)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits, labels)

        if loss is not None:
            return {'logits': logits, 'loss': loss}
        else:
            return logits

    def configure_adamw_optimizer(self, weight_decay, learning_rate, device, betas=(0.9, 0.999), eps=1e-8, lora_weight_decay=0.0, is_master_process=True):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        lora_names = [n for n in param_dict if n.endswith('.A') or n.endswith('.B')]
        lora_params = [param_dict[n] for n in lora_names]

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2 and n not in lora_names]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2 and n not in lora_names]

        # disable optimization for other params if lora is set:
        if lora_params:
            for p in decay_params + nodecay_params:
                p.requires_grad_(False)

            optim_groups = [
                {'params': lora_params, 'weight_decay': lora_weight_decay}
            ]
        else:
            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        num_lora_params = sum(p.numel() for p in lora_params)

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device

        total_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        if is_master_process:
            print(f'\nOptimizer configuration:')
            print('----------------------------------------')
            print(f'num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters')
            print(f'num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters')
            if lora_params:
                print(f'num lora parameter tensors: {len(lora_params)}, with {num_lora_params:,} parameters')
            print(f'using fused AdamW: {use_fused}')
            print(f'trainable parameters: {total_trainable_params}')

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=list(betas), eps=eps, fused=use_fused)
        return optimizer

    def sample_top_p(self, probs, p):
        ''' Top P - Sorts the tokens from highest probabilities to lowest and calculates cumulative probabilities up to the cumulative >= p.
        '''
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

        next_token = torch.multinomial(probs_sort, num_samples=1)

        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token

    def temperature_and_top_p_sampling(self, logits, temperature, top_p):
        ''' Applies temperature and calculates top P. If temperature is 0 we just get the token with highest logit.
            Penalty should be ~[1.05, 1.2]
        '''
        if temperature > 0:
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = self.sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)
        return next_token

    def apply_repetition_penalty(self, current_tokens, logits, penalty):
        ''' Applies the repetition penalty to reduce the probability of the same exact tokens appear multiple times even if they have strong logit.
            Modification is in-place.
        '''
        if penalty == 1.0:
            return

        # (for each batch index) look at all the token logits that already were generated and apply the penalty:
        for batch_index in range(logits.size(0)):
            tokens = current_tokens[batch_index].unique()
            positive_logits = logits[batch_index, tokens] > 0

            # penalty: if logit is positive, we want to reduce it, if it is negative we want to make it more negative
            logits[batch_index, tokens] = torch.where(
                positive_logits,
                logits[batch_index, tokens] / penalty,
                logits[batch_index, tokens] * penalty
            )

    def apply_no_repeat_ngram(self, current_tokens, logits, ngram_size):
        ''' Applies the no_repeat_ngram strategy. Computes the ngrams of size ngram_size accross the token sequence and bans 
        tokens "ngram_size" for the prefix "ngram_size - 1" (Tokens that would complete an ngram).
            Modification is in-place.
        '''
        if ngram_size <= 1 or current_tokens.size(1) < ngram_size - 1:
            return

        for batch_index in range(logits.size(0)):
            banned = defaultdict(set)
            tokens = current_tokens[batch_index].tolist()

            # generate the bans
            for i in range(len(tokens) - ngram_size + 1):
                slide_index = i + ngram_size - 1
                prefix, banned_token = tuple(tokens[i: slide_index]), tokens[slide_index]

                banned[prefix].add(banned_token)

            # get last prefix
            prefix = tuple(tokens[-(ngram_size - 1):])
            for banned_token in banned.get(prefix, ()):
                logits[batch_index, banned_token] = float('-inf')

    def generate(self, prompt_tokens, max_gen_len, temperature, top_p, repetition_penalty, no_repeat_ngram_size, device):
        batch_size = len(prompt_tokens)

        # Finding the boundaries / limits.
        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        total_len = min(self.config.max_seq_len, max_gen_len + max_prompt_len)

        # Here we assume we receive a batch of multiple tokenized sequences.
        pad_id = self.config.pad_token_id
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
        
        for batch, tokens_list in enumerate(prompt_tokens):
            tokens[batch, : len(tokens_list)] = torch.tensor(tokens_list, dtype=torch.long, device=device)

        # Define stop conditions, input mask and the stop tokens (extracted from the tokenizer)
        eos_reached = torch.tensor([False] * batch_size, device=device)
        input_text_mask = tokens != pad_id
        stop_tokens = torch.tensor(list(self.config.stop_tokens), device=device)

        with torch.no_grad():
            for current_position in range(min_prompt_len, total_len):
                logits = self.forward(tokens[:, :current_position], start_position=0)

                # Apply repetition penalty
                self.apply_repetition_penalty(
                    tokens[:, :current_position],
                    logits[:, -1],
                    penalty=repetition_penalty
                )

                # Apply no repeat ngram
                self.apply_no_repeat_ngram(
                    tokens[:, :current_position],
                    logits[:, -1],
                    ngram_size=no_repeat_ngram_size
                )
                    
                # Temperature and sampling.
                next_token = self.temperature_and_top_p_sampling(logits, temperature, top_p)
                next_token = next_token.reshape(-1)

                # Gets the next token depending on the condition (mask) and appends to tokens.
                next_token = torch.where(
                    input_text_mask[:, current_position], tokens[:, current_position], next_token
                )
                tokens[:, current_position] = next_token

                # # Checks if we reached the eos on all sequences in the batch and updates the current position.
                eos_reached |= (~input_text_mask[:, current_position]) & (torch.isin(next_token, stop_tokens))
                
                if all(eos_reached):
                    break

            # For all the sequences, we extract all tokens up to a stop_token if it exists.
            out_tokens = []
            for i, toks in enumerate(tokens.tolist()):
                start = len(prompt_tokens[i])
                toks = toks[start : len(prompt_tokens[i]) + max_gen_len]

                for stop_token in self.config.stop_tokens:
                    try:
                        eos_idx = toks.index(stop_token)
                        toks = toks[:eos_idx]
                    except ValueError:
                        pass
                out_tokens.append(toks)

        torch.cuda.empty_cache()

        return out_tokens

    def test_dialogue_custom(
        self,
        texts,
        *,
        max_gen_len=256,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.0,
        no_repeat_ngram_size=1,
        full_seq=False,
        device='cpu',
        is_instruct=False,
        return_result=False,
        skip_encoding=False
    ):
        if not isinstance(texts, list):
            texts = [texts]

        tokenizer = self.config.tokenizer

        if not skip_encoding:
            if is_instruct:
                prompt_tokens = [tokenizer.encode_instruct(text) for text in texts]
            else:
                prompt_tokens = [tokenizer.encode(text) for text in texts]
        else:
            prompt_tokens = texts
        
        generation_tokens = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            device=device
        )

        def validate_token(tokens):
            return [token if token < self.config.vocab_size else self.config.pad_token_id for token in tokens]

        results = [tokenizer.decode(validate_token(t)) for t in generation_tokens]

        outputs = []

        if return_result is False:
            print('-------------------------------------------')

        for text, result in zip(texts, results):
            prompt_and_result = text + result
            if is_instruct:
                prompt_and_result = text + ' ' + result
            if return_result is False:
                print(prompt_and_result + '<|FAKE_END|>')
            if full_seq:
                outputs.append(prompt_and_result)
            else:
                outputs.append(result)
        if return_result is False:
            print('-------------------------------------------')
        return outputs

    def get_parameters_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_model_params(self, return_text=False):
        NUMBER_OF_PARAMETERS_LABEL = round(sum(p.numel() for p in self.parameters())/1e6), 'M parameters'
        NUMBER_OF_PARAMETERS_LABEL = str(NUMBER_OF_PARAMETERS_LABEL[0]) + ' ' + NUMBER_OF_PARAMETERS_LABEL[1]
        print(NUMBER_OF_PARAMETERS_LABEL)
        if return_text:
            return NUMBER_OF_PARAMETERS_LABEL


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
            'stop_tokens': list(self.stop_tokens)
        }
