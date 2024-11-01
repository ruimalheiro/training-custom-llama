import torch
import torch.nn.functional as F
import torch.distributed as dist
import argparse
import time
import math

from torch.distributed import destroy_process_group
from tokenizer import Tokenizer
from model import Transformer, ModelConfig
from model_utils import save_model, load_model, init_multi_gpu, prepare_model_for_ddp, WnbWrapper
from dataloaders import DataLoaderLite, init_data_loaders
from hellaswag_utils import render_example, get_most_likely_row, iterate_examples
from tqdm import tqdm

##############################################################
# export WANDB_API_KEY=<KEY> && export OMP_NUM_THREADS=1 && torchrun --standalone --nproc_per_node 1 train.py

EDU_FINEWEB_PATH = './edu_fineweb10B'
HELLASWAG_PATH = './hellaswag'
CHECKPOINTS_DIR = './checkpoints'
TOKENIZER_CHECKPOINT_PATH = './tokenizer.model'

total_batch_size = 524288

max_lr = 6e-4 * 3
min_lr = max_lr * 0.1
warmup_steps = 715
weight_decay=0.1

max_steps = 19073

validate_every_x_steps = 2
val_steps = 50

generate_every_x_steps = 2
mas_gen_len=32
test_generation_prompts = [
    'I am a language model,',
    'Computers are',
    'Artificial Intelligence is',
    'I like'
]

hellaswag_every_x_steps = 2
hellag_swag_limit = 10


version = 1
wnb_disabled=True
wnb_project_name = f'custom_llama3_pretraining_local_v{version}'

config = ModelConfig(
    dim=768,
    n_layers=16,
    n_heads=16,
    n_kv_heads=8,
    vocab_size=128256,
    multiple_of=1024,
    ffn_dim_multiplier=1.0,
    norm_eps=1e-05,
    rope_theta=500000.0,
    max_batch_size=4,
    max_seq_len=1024
)

##############################################################


parser = argparse.ArgumentParser(description="Training Script")
parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to load.')

args = parser.parse_args()

ddp, ddp_rank, ddp_local_rank, ddp_world_size, is_master_process, device, device_type = init_multi_gpu(seed=42)

tokenizer = Tokenizer(TOKENIZER_CHECKPOINT_PATH)

config.tokenizer = tokenizer
config.pad_token_id = tokenizer.pad_id
config.stop_tokens = tokenizer.stop_tokens

model = Transformer(config)
model.to(device)

torch.set_float32_matmul_precision('high')

model, raw_model = prepare_model_for_ddp(model, ddp_local_rank)

train_loader, val_loader = init_data_loaders(
    batch_size=config.max_batch_size,
    sequence_length=config.max_seq_len,
    is_master_process=is_master_process,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    data_root=EDU_FINEWEB_PATH,
    use_shuffle=True
)

assert total_batch_size % (config.max_batch_size * config.max_seq_len) == 0

grad_accum_steps = total_batch_size // (config.max_batch_size * config.max_seq_len * ddp_world_size)
if is_master_process:
    print(f'total batch size: {total_batch_size}')
    print(f'calculated gradient accumulation steps: {grad_accum_steps}')

print(f'I am GPU: {ddp_rank}')

optimizer = raw_model.configure_optimizers(weight_decay=weight_decay, learning_rate=max_lr, device=device, is_master_process=is_master_process)

start_step = 0
if args.checkpoint is not None:
    model, optimizer, start_step = load_model(CHECKPOINTS_DIR, args.checkpoint, raw_model, optimizer, is_master_process=is_master_process)

wnb = WnbWrapper(disabled=wnb_disabled, is_master_process=is_master_process)
wnb.init(wnb_project_name, config={
    'batch_size': config.max_batch_size,
    'sequence_length': config.max_seq_len,
    'min_learning_rate': min_lr,
    'max_learning_rate': max_lr,
})

def get_lr(it):
    # cosine lr scheduler
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <=1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

for step in tqdm(range(start_step, max_steps), initial=start_step, total=max_steps, desc='Training'):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    if step > 0 and step % validate_every_x_steps == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_acc = 0.0
            val_loss_steps = val_steps
            for _ in tqdm(range(val_loss_steps), 'Validating'):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    result = model(x, labels=y)
                    logits, loss = result['logits'], result['loss']
                loss  = loss / val_loss_steps
                val_loss_acc += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_acc, op=dist.ReduceOp.AVG)
        if is_master_process:
            print(f'validation loss: {val_loss_acc.item():.4f}')
            save_model(CHECKPOINTS_DIR, raw_model, config, step, val_loss_acc, optimizer)
            wnb.log({'Validation Loss': val_loss_acc.item()})

    if step > 0 and step % hellaswag_every_x_steps == 0 or last_step:
        model.eval()
        num_correct_norm = 0
        num_total = 0
        for i, example in tqdm(enumerate(iterate_examples(HELLASWAG_PATH, 'val', size=hellag_swag_limit)), 'Hellaswag validation', unit=' examples'):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example, tokenizer)
            tokens = tokens.to(device)
            mask = mask.to(device)

            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits = model(tokens)
                    
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)

        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if is_master_process:
            print(f'HellaSwag accuracy: {num_correct_norm} / {num_total} = {acc_norm:.4f} ')
            wnb.log({'HellaSwag accuracy': acc_norm})

    if step > 0 and step % generate_every_x_steps == 0 or last_step:
        model.eval()
        raw_model.test_dialogue_custom(
            test_generation_prompts,
            max_gen_len=mas_gen_len,
            device=device
        )

    model.train()
    optimizer.zero_grad()
    loss_acc = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            result = model(x, labels=y)
            logits, loss = result['logits'], result['loss']

        loss /= grad_accum_steps
        loss_acc += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()

    if ddp:
        dist.all_reduce(loss_acc, op=dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()

    torch.cuda.synchronize()

    t1 = time.time()
    dt = (t1 - t0)
    tokens_per_sec = (train_loader.B * train_loader.S * grad_accum_steps * ddp_world_size) / (t1 - t0)

    if is_master_process:
        print(f'step: {step:4d} | loss: {loss_acc.item():.4f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}s | tok/sec: {tokens_per_sec:.2f}')
        wnb.log({'Train Loss': loss_acc.item()})

if ddp:
    destroy_process_group()

wnb.finish()
