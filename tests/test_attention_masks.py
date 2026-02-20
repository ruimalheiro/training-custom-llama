import torch


@torch.no_grad()
def test_padded_batch_matches_individual(model, device):
    model.eval()
    torch.manual_seed(0)

    pad_id = model.config.pad_token_id

    # Two sequences with different lengths
    seq1 = [10, 20, 30, 40, 50]      # len 5
    seq2 = [11, 22, 33]              # len 3

    max_len = max(len(seq1), len(seq2))

    input_ids = torch.full((2, max_len), pad_id, dtype=torch.long, device=device)
    input_ids[0, :len(seq1)] = torch.tensor(seq1, device=device)
    input_ids[1, :len(seq2)] = torch.tensor(seq2, device=device)

    # hugging face attention mask: 1=real token, 0=pad
    attention_mask = torch.zeros((2, max_len), dtype=torch.long, device=device)
    attention_mask[0, :len(seq1)] = 1
    attention_mask[1, :len(seq2)] = 1

    # batched forward (with padding)
    out_batched = model.forward(input_ids, attention_mask=attention_mask, kv_cache=None).logits

    # individual forwards (no padding)
    out_seq1 = model.forward(
        torch.tensor([seq1], dtype=torch.long, device=device),
        attention_mask=None,
        kv_cache=None
    ).logits

    out_seq2 = model.forward(
        torch.tensor([seq2], dtype=torch.long, device=device),
        attention_mask=None,
        kv_cache=None
    ).logits

    # compare only valid positions
    torch.testing.assert_close(out_batched[0, :len(seq1)], out_seq1[0], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out_batched[1, :len(seq2)], out_seq2[0], rtol=1e-4, atol=1e-4)

    print('Padded batch matches individual forwards (masking is correct)')
