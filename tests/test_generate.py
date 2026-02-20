import torch

from generate import generate


def test_generate_kv_cache_matches_no_cache_greedy(model, device):
    model.eval()
    torch.manual_seed(0)

    # token ids must be within vocab and not pad/stop ideally
    prompts = [
        [10, 20, 30, 40],
        [11, 22],
        [7, 8, 9]
    ]

    kwargs = dict(
        model=model,
        prompt_tokens=prompts,
        max_gen_len=16,
        temperature=0.0,
        top_p=0.9, # ignored when temperature=0
        repetition_penalty=1.0,
        no_repeat_ngram_size=1,
        device=device,
    )

    out_no_cache = generate(**kwargs, use_kv_cache=False)
    out_cache = generate(**kwargs, use_kv_cache=True)

    assert out_no_cache == out_cache

def test_generate_batched_variable_lengths_smoke(model, device):
    model.eval()
    torch.manual_seed(0)

    prompts = [
        [10, 11, 12, 13, 14],
        [21, 22],
        [31, 32, 33, 34],
    ]

    max_gen_len = 12

    out = generate(
        model=model,
        prompt_tokens=prompts,
        max_gen_len=max_gen_len,
        temperature=0.0,
        top_p=0.9,
        repetition_penalty=1.0,
        no_repeat_ngram_size=1,
        device=device,
        use_kv_cache=True,
    )

    assert isinstance(out, list)
    assert len(out) == len(prompts)

    for gen_tokens in out:
        assert isinstance(gen_tokens, list)
        assert len(gen_tokens) <= max_gen_len

def test_generate_empty_prompt_raises(model, device):
    with torch.no_grad():
        try:
            generate(
                model=model,
                prompt_tokens=[[1, 2, 3], []], # one empty prompt
                max_gen_len=8,
                temperature=0.0,
                top_p=0.9,
                repetition_penalty=1.0,
                no_repeat_ngram_size=1,
                device=device,
                use_kv_cache=True,
            )
            assert False, 'Expected ValueError for empty prompt'
        except ValueError as e:
            assert 'Prompt cannot be of length 0' in str(e)
