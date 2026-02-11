import torch

from collections import defaultdict


def sample_top_p(probs, p):
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

def temperature_and_top_p_sampling(logits, temperature, top_p):
    ''' Applies temperature and calculates top P. If temperature is 0 we just get the token with highest logit.
        Penalty should be ~[1.05, 1.2]
    '''
    if temperature > 0:
        probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
    else:
        next_token = torch.argmax(logits[:, -1], dim=-1)
    return next_token

def apply_repetition_penalty(current_tokens, logits, penalty):
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

def apply_no_repeat_ngram(current_tokens, logits, ngram_size):
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

def generate(model, prompt_tokens, max_gen_len, temperature, top_p, repetition_penalty, no_repeat_ngram_size, device):
        batch_size = len(prompt_tokens)

        # Finding the boundaries / limits.
        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        total_len = min(model.config.max_seq_len, max_gen_len + max_prompt_len)

        # Here we assume we receive a batch of multiple tokenized sequences.
        pad_id = model.config.pad_token_id
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
        
        for batch, tokens_list in enumerate(prompt_tokens):
            tokens[batch, : len(tokens_list)] = torch.tensor(tokens_list, dtype=torch.long, device=device)

        # Define stop conditions, input mask and the stop tokens (extracted from the tokenizer)
        eos_reached = torch.tensor([False] * batch_size, device=device)
        input_text_mask = tokens != pad_id
        stop_tokens = torch.tensor(list(model.config.stop_tokens), device=device)

        with torch.no_grad():
            for current_position in range(min_prompt_len, total_len):
                logits = model.forward(tokens[:, :current_position], start_position=0).logits

                # Apply repetition penalty
                apply_repetition_penalty(
                    tokens[:, :current_position],
                    logits[:, -1],
                    penalty=repetition_penalty
                )

                # Apply no repeat ngram
                apply_no_repeat_ngram(
                    tokens[:, :current_position],
                    logits[:, -1],
                    ngram_size=no_repeat_ngram_size
                )
                    
                # Temperature and sampling.
                next_token = temperature_and_top_p_sampling(logits, temperature, top_p)
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

                for stop_token in model.config.stop_tokens:
                    try:
                        eos_idx = toks.index(stop_token)
                        toks = toks[:eos_idx]
                    except ValueError:
                        pass
                out_tokens.append(toks)

        return out_tokens

def generate_and_decode(
        model,
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
        skip_encoding=False
    ):
        if not isinstance(texts, list):
            texts = [texts]

        tokenizer = model.config.tokenizer

        if not skip_encoding:
            if is_instruct:
                prompt_tokens = [tokenizer.encode_instruct(text) for text in texts]
            else:
                prompt_tokens = [tokenizer.encode(text) for text in texts]
        else:
            prompt_tokens = texts
        
        generation_tokens = generate(
            model=model,
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            device=device
        )

        def validate_token(tokens):
            return [token if token < model.config.vocab_size else model.config.pad_token_id for token in tokens]

        results = [tokenizer.decode(validate_token(t)) for t in generation_tokens]

        outputs = []

        for text, result in zip(texts, results):
            prompt_and_result = text + result
            if is_instruct:
                prompt_and_result = text + ' ' + result
            if full_seq:
                outputs.append(prompt_and_result)
            else:
                outputs.append(result)
        return outputs
