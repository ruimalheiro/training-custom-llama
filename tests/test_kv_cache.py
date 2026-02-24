import torch

from kv_cache import KVCache


@torch.no_grad()
def test_kv_cache_matches_no_cache(model, device, dummy_prompt_tokens, steps=8):
    model.eval()
    torch.manual_seed(0)

    # Single batch item is easiest first
    if isinstance(dummy_prompt_tokens[0], int):
        prompt_tokens = [dummy_prompt_tokens]

    batch_size = len(prompt_tokens)

    pad_id = model.config.pad_token_id
    total_len = min(model.config.max_seq_len, max(len(t) for t in prompt_tokens) + steps)

    tokensA = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
    tokensB = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)

    for i, toks in enumerate(prompt_tokens):
        toks_t = torch.tensor(toks, dtype=torch.long, device=device)
        tokensA[i, :len(toks)] = toks_t
        tokensB[i, :len(toks)] = toks_t

    current_position = len(prompt_tokens[0])

    # build cache
    kv_cache = KVCache(
        num_layers=model.config.n_layers,
        batch_size=batch_size,
        max_seq_len=total_len,
        n_kv_heads=(model.config.n_kv_heads or model.config.n_heads),
        head_dim=model.config.dim // model.config.n_heads,
        device=device,
        dtype=model.tok_embeddings.weight.dtype
    )

    # prefill
    outA = model.forward(tokensA[:, :current_position], kv_cache=None, start_position=0)
    outB = model.forward(tokensB[:, :current_position], kv_cache=kv_cache, start_position=0)

    torch.testing.assert_close(outA['logits'], outB['logits'], rtol=1e-4, atol=1e-4)

    for step in range(steps):
        logitsA = outA['logits'][:, -1]
        logitsB = outB['logits'][:, -1]

        # compare logits before sampling
        try:
            torch.testing.assert_close(logitsA, logitsB, rtol=1e-4, atol=1e-4)
        except AssertionError:
            max_diff = (logitsA - logitsB).abs().max().item()
            raise AssertionError(f'Cache mismatch at decode step {step}, max_abs_diff={max_diff}')

        # force SAME next token on both paths
        next_token = torch.argmax(logitsA, dim=-1)
        tokensA[:, current_position] = next_token
        tokensB[:, current_position] = next_token

        current_position += 1

        # no-cache path - full prefix
        outA = model.forward(tokensA[:, :current_position], kv_cache=None, start_position=0)

        # cache path - one-token decode
        outB = model.forward(
            tokensB[:, current_position-1:current_position],
            kv_cache=kv_cache,
            start_position=current_position-1
        )

    print('KV-cache path matches no-cache path stepwise')
