import json
import torch
import torch.nn.functional as F
import torch.distributed as dist


def load_hellaswag_file(path, ddp, is_master_process, size=None):
    # Loads the file and broadcasts to other ranks
    def prepare_line(line):
        example = json.loads(line)
        return {
            'tokens': torch.tensor(example['tokens'], dtype=torch.long),
            'mask': torch.tensor(example['mask'], dtype=torch.long),
            'label': int(example['label']),
            'valid': True
        }

    def create_dummy(shards):
        return {
            'tokens': torch.zeros_like(shards[0][0]['tokens']) if shards and shards[0] else torch.zeros((4, 2), dtype=torch.long),
            'mask': torch.ones_like(shards[0][0]['mask'])   if shards and shards[0] else torch.ones((4, 2), dtype=torch.long),
            'label': -1,
            'valid': False
        }

    world_size = dist.get_world_size() if ddp else 1

    shards = None
    if is_master_process:
        # master builds the shards
        with open(f'{path}/hellaswag_val.jsonl', 'r') as f:
            data = [prepare_line(line) for line in f]
            if size:
                data = data[:size]

        shard_size = (len(data) + world_size - 1) // world_size
        shards = [data[i * shard_size : (i+1) * shard_size] for i in range(world_size)]

        while len(shards) < world_size: # number of shards must be equal to the world size
            shards.append([])

        # need to pad as each shard needs to have same size so all ranks call forward()
        dummy = create_dummy(shards)
        for i in range(world_size):
            target = shard_size - len(shards[i])
            if target > 0:
                shards[i].extend(target * [dummy])
    else:
        shards = [None]

    if ddp:
        # scatter the shards for respective rank
        buffer_obj = [None]
        dist.scatter_object_list(buffer_obj, scatter_object_input_list=shards if is_master_process else None, src=0)
        data = buffer_obj[0]
    else:
        data = shards[0]

    return data

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
