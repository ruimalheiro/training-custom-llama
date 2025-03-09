import json
import torch
import torch.nn.functional as F


def iterate_hellaswag_val_examples(data_root, size=None):
    i = 0
    with open(f'{data_root}/hellaswag_val.jsonl', 'r') as f:
        for line in f:
            example = json.loads(line)
            yield example
            i += 1
            if size and i == size:
                break


def prepare_hellaswag_example(example, tokenizer):
    """
    Sample example from hellaswag (without some of the metadata):
        {
          "ctx": "A man is sitting on a roof. he",
          "label": 3,
          "endings": [
            "is using wrap to wrap a pair of skis.",
            "is ripping level tiles off.",
            "is holding a rubik's cube.",
            "starts pulling up roofing on a roof."
          ]
        }
    """
    context = example['ctx']
    label = example['label'] # Index for the correct completion
    endings = example['endings'] # Candidates - always 4

    context_tokens = tokenizer.encode(context)

    data = {
        'context_tokens': context_tokens,
        'label': label,
        'ending_tokens': []
    }

    tokens_rows = []
    mask_rows = []
    for ending in endings:
        ending_tokens = tokenizer.encode(ending)
        tokens_rows.append(context_tokens + ending_tokens)

        mask_row = torch.cat([torch.zeros(len(context_tokens)), torch.ones(len(ending_tokens))])
        mask_rows.append(mask_row)

        data['ending_tokens'].append(ending_tokens)


    # rows can have different lengths so pick max for row length.
    max_len = max(len(row) for row in tokens_rows)

    # (4 candidates * max length)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tokens_row, mask_row) in enumerate(zip(tokens_rows, mask_rows)):
        tokens[i, :len(tokens_row)] = torch.tensor(tokens_row)
        mask[i, :len(mask_row)] = mask_row.clone().detach()

    return data, tokens, mask, label


def estimate_correct_candidate_selection(tokens, mask, logits):
    # align tokens mask and logits (remove first token in tokens/mask and last logit in logits)
    shift_tokens = (tokens[..., 1:]).contiguous()
    shift_mask = (mask[..., 1:]).contiguous()
    shift_logits = (logits[..., :-1, :]).contiguous()

    # Flatten for cross_entropy
    flat_shift_tokens = shift_tokens.view(-1)
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))

    losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')

    # restore shape same as tokens
    shift_losses = losses.view(tokens.size(0), -1)

    # Apply the mask
    masked_shift_losses = shift_losses.masked_fill(shift_mask == 0, 0)

    # calculate loss for each candidate completion.
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)

    # Pick the one with lowest loss.
    estimated_correct = avg_loss.argmin().item()
    return estimated_correct
