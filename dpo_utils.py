import torch
import torch.nn.functional as F


def dpo_log_probs(model, prompt_ids, resp_ids):
    ''' Computes de log probabilities for each entry in the batch. Prompt + response.

        Assumes dimensions prompt_ids [B, P] and resp_ids [B, R]
    '''
    device = prompt_ids.device

    # Get dimensions
    B, P = prompt_ids.shape
    R = resp_ids.size(1)

    # Here we concatenate prompt + response but remove the last token (once it goes in the model, for each token we predict the next). Lets assume L = size prompt length + response length
    full_input = torch.cat([prompt_ids, resp_ids[:, :-1]], dim=1) # [B, L]
    # The response flattened
    labels = resp_ids.reshape(-1) # [B * R]

    # compute logits
    logits = model(full_input) # [B, L, V]

    # Flatten B * L
    logits_flat = F.log_softmax(logits, dim=-1).view(-1, logits.size(-1)) # [B * L, V]

    L = logits.size(1)
    # Since we have a flat representation and we need to get the logprobs of the response segments, we do:
    # - Generate the base indexes that will contain the response for the first item in the batch. Then repeat this B times E.g: Indexes [3, 4, 5] and B = 2 -> [[3, 4, 5], [3, 4, 5]]
    base_indexes = torch.arange(P - 1, L, device=device).repeat(B)
    # - Generate the offsets by taking advantage of batch size and creating B entries that jump L by L E.g: If L is 2 [[0],[2]]. Finally for each of them we allocate
    # space for the response by repeating R times. E.g if R = 2 -> [[0, 0],[2, 2]]
    offsets = (torch.arange(B, device=device) * L).repeat_interleave(R)

    # Add them together and we have a mask for all the resp positions accross all entries.
    resp_pos = base_indexes + offsets

    # Get log probabilites of the response tokens (labels)
    resp_logp = logits_flat[resp_pos, labels]

    # Restore dimensions and sum accoss axis 1 to get the response log probs for each entry in the batch
    logp_per_sequence = resp_logp.view(B, R).sum(dim=-1)

    return logp_per_sequence


def dpo_loss(
    policy_log_probs_pos,
    policy_log_probs_neg,
    reference_log_probs_pos,
    reference_log_probs_neg,
    beta
):
    ''' Computes the DPO loss.
        For reference, Direct Preference Optimization paper: https://arxiv.org/pdf/2305.18290
    '''
    pos_difference = policy_log_probs_pos - reference_log_probs_pos
    neg_difference = policy_log_probs_neg - reference_log_probs_neg

    return - F.logsigmoid(beta * (pos_difference - neg_difference)).mean()
