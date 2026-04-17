import argparse

from config import config
from engine import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script options')
    parser.add_argument('--pretrain_checkpoint', type=str, default=None, help='Pretrain checkpoint to load.')
    parser.add_argument('--instruct_checkpoint', type=str, default=None, help='Instruct checkpoint to load.')
    parser.add_argument('--dpo_checkpoint', type=str, default=None, help='DPO checkpoint to load.')
    parser.add_argument('--reset-optimizers', action='store_true', help='Reset the optimizers state when loading a checkpoint.')
    parser.add_argument('--start-step', type=int, default=None, help='Starting step number for training.')

    args = parser.parse_args()

    trainer = Trainer(
        config = config,
        args=args
    )
    trainer.train()

# if is_pretraining and should_run(step, hellaswag_every_x_steps, last_step):
#     model.eval()
#     num_correct_norm = 0
#     num_total = 0
#     for example in tqdm(HELLASWAG_DATA, 'HellaSwag validation', unit=' examples', disable=not is_master_process, leave=False):
#         tokens, mask, label, valid = example['tokens'], example['mask'], example['label'], example['valid']
#         tokens = tokens.to(device)
#         mask = mask.to(device)

#         # get the logits
#         with torch.no_grad():
#             with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=use_autocast):
#                 logits = model(tokens)['logits']

#         if valid: # Some examples might be dummy in FSDP
#             predicted_correct = estimate_correct_candidate_selection(tokens, mask, logits)
#             num_total += 1
#             num_correct_norm += int(predicted_correct == label)

#     # Aggregates counts across all GPU processes (in DDP) to compute global accuracy.
#     if ddp:
#         num_total = torch.tensor(num_total, dtype=torch.long, device=device)
#         num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
#         dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
#         dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
#         num_total = num_total.item()
#         num_correct_norm = num_correct_norm.item()
#     acc_norm = num_correct_norm / num_total

#     if is_master_process:
#         logger.info(f'HellaSwag accuracy: {num_correct_norm} / {num_total} = {acc_norm:.4f}', pbar=pbar)
#         wandb.log({'HellaSwag accuracy': acc_norm})
