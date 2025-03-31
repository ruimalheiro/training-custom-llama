import torch
import torch.nn.functional as F
import torch.distributed as dist
import argparse
import time
import math

from torch.distributed import broadcast, barrier, destroy_process_group
from tokenizer import Tokenizer
from model import Transformer, ModelConfig
from model_utils import save_model, load_model, init_multi_gpu, prepare_model_for_ddp, WnbWrapper
from dataloaders import init_data_loaders
from hellaswag_utils import (
    iterate_hellaswag_val_examples,
    prepare_hellaswag_example,
    estimate_correct_candidate_selection
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from config import config

##################################################
### CONFIGURATION ###

# datasets path
dataloader_root_path = config.dataloader_root_path
hellaswag_path = config.hellaswag_path

# save / load path
load_checkpoints_path = config.load_checkpoints_path
save_checkpoints_path = config.save_checkpoints_path
save_checkpoints = config.save_checkpoints

# wnb
wnb_disabled = config.wnb_disabled
wnb_project_name = config.wnb_project_name

# tokenizer model path
tokenizer_checkpoint_path = config.tokenizer_checkpoint_path

# train config
total_batch_size = config.total_batch_size
max_lr = config.max_lr
min_lr = config.min_lr
warmup_steps = config.warmup_steps
weight_decay = config.weight_decay
max_steps = config.max_steps
early_stopping_patience = config.early_stopping_patience
is_instruct_training = config.is_instruct_training
is_model_distillation = config.is_model_distillation
distillation_temperature = config.distillation_temperature
# The teacher model is loader via huggingface API: AutoModelForCausalLM.from_pretrained(teacher_model_checkpoint, ...) so needs to ve a valid checkpoint.
teacher_model_checkpoint = config.teacher_model_checkpoint

# validation
validate_every_x_steps = config.validate_every_x_steps
val_steps = config.val_steps
hellaswag_every_x_steps = config.hellaswag_every_x_steps
hellagswag_number_of_examples = config.hellagswag_number_of_examples
generate_every_x_steps = config.generate_every_x_steps
max_test_gen_len = config.max_test_gen_len

# test prompts
test_pretrain_generation_prompts = config.test_pretrain_generation_prompts
test_instruct_generation_prompts = config.test_instruct_generation_prompts


model_config = ModelConfig(
    dim=config.dim,
    n_layers=config.n_layers,
    n_heads=config.n_heads,
    n_kv_heads=config.n_kv_heads,
    vocab_size=config.vocab_size,
    multiple_of=config.multiple_of,
    ffn_dim_multiplier=config.ffn_dim_multiplier,
    norm_eps=config.norm_eps,
    rope_theta=config.rope_theta,
    max_batch_size=config.max_batch_size,
    max_seq_len=config.max_seq_len
)

##################################################


parser = argparse.ArgumentParser(description='Training Script')
parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to load.')
parser.add_argument('--reset-optimizer', action='store_true', help='Reset the optimizer state when loading a checkpoint.')
parser.add_argument('--start-step', type=int, default=None, help='Starting step number for training.')

args = parser.parse_args()

ddp, ddp_rank, ddp_local_rank, ddp_world_size, is_master_process, device, device_type = init_multi_gpu(seed=42)

tokenizer = Tokenizer(tokenizer_checkpoint_path)

model_config.tokenizer = tokenizer
model_config.pad_token_id = tokenizer.pad_id
model_config.stop_tokens = tokenizer.stop_tokens

model = Transformer(model_config)
model.to(device)

torch.set_float32_matmul_precision('high')

model, raw_model = prepare_model_for_ddp(model, ddp_local_rank)

train_loader, val_loader = init_data_loaders(
    batch_size=model_config.max_batch_size,
    sequence_length=model_config.max_seq_len,
    is_master_process=is_master_process,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    data_root=dataloader_root_path,
    use_shuffle=True
)

assert total_batch_size % (model_config.max_batch_size * model_config.max_seq_len) == 0

grad_accum_steps = total_batch_size // (model_config.max_batch_size * model_config.max_seq_len * ddp_world_size)
if is_master_process:
    print(f'total batch size: {total_batch_size}')
    print(f'calculated gradient accumulation steps: {grad_accum_steps}')

print(f'I am GPU: {ddp_rank} and I am ready to go brrr :)')

teacher_model = None
if is_model_distillation:
    print(f'Loading teacher model on gpu: {ddp_rank}...')
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_checkpoint, cache_dir='./cache').to(device)
    print(f'Finished loading teacher model on gpu: {ddp_rank}...')

optimizer = raw_model.configure_optimizers(weight_decay=weight_decay, learning_rate=max_lr, device=device, is_master_process=is_master_process)

start_step = 0
if args.checkpoint is not None:
    model, optimizer, start_step = load_model(
        load_checkpoints_path,
        args.checkpoint,
        raw_model,
        optimizer=optimizer,
        reset_optimizer=args.reset_optimizer,
        force_start_step=args.start_step,
        is_master_process=is_master_process
    )

wnb = WnbWrapper(disabled=wnb_disabled, is_master_process=is_master_process)
wnb.init(wnb_project_name, config={
    'batch_size': model_config.max_batch_size,
    'sequence_length': model_config.max_seq_len,
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

def distillation_loss(teacher_logits, student_logits, temperature=1.0):
    teacher_probabilities = F.softmax(teacher_logits.view(-1, teacher_logits.size(-1)) / temperature, dim=-1)
    student_log_probabilities = F.log_softmax(student_logits / temperature, dim=-1)

    kl_divergence = F.kl_div(student_log_probabilities, teacher_probabilities, reduction='batchmean') * (temperature ** 2)
    return kl_divergence

test_generation_prompts = test_pretrain_generation_prompts
if is_instruct_training:
    test_generation_prompts = test_instruct_generation_prompts

best_val_loss = float('inf')
epochs_no_improve = 0
abort_if_no_improve = torch.tensor([0], device=device)
for step in tqdm(range(start_step, max_steps), initial=start_step, total=max_steps, desc='Training'):
    if abort_if_no_improve.item() == 1:
        print(f'Rank {ddp_rank} received stop signal.')
        break

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

            if val_loss_acc.item() < best_val_loss:
                best_val_loss = val_loss_acc.item()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f'Validation loss did not improve. Best: {best_val_loss}, Latest: {val_loss_acc.item()} - Attempts left: {early_stopping_patience - epochs_no_improve}')

            if save_checkpoints is True and epochs_no_improve == 0:
                save_model(save_checkpoints_path, raw_model, model_config, step, val_loss_acc, optimizer)
            wnb.log({'Validation Loss': val_loss_acc.item()})

            stop_signal = torch.tensor([0], device=device)
            if epochs_no_improve == early_stopping_patience:
                print(f'The validation loss did not improve for: {early_stopping_patience} - Aborting training...')
                abort_if_no_improve[0] = 1

        if ddp:
            broadcast(abort_if_no_improve, src=0)

    if step > 0 and step % hellaswag_every_x_steps == 0 or last_step:
        model.eval()
        num_correct_norm = 0
        num_total = 0
        for i, example in tqdm(enumerate(iterate_hellaswag_val_examples(hellaswag_path, size=hellagswag_number_of_examples)), 'Hellaswag validation', unit=' examples'):
            # if i % ddp_world_size == ddp_rank (gpu itself), process.
            if i % ddp_world_size != ddp_rank:
                continue

            _, tokens, mask, label = prepare_hellaswag_example(example, tokenizer)
            tokens = tokens.to(device)
            mask = mask.to(device)

            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits = model(tokens)
                    
                predicted_correct = estimate_correct_candidate_selection(tokens, mask, logits)

            num_total += 1
            num_correct_norm += int(predicted_correct == label)

        # Aggregates counts across all GPU processes (in DDP) to compute global accuracy.
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total

        if is_master_process:
            print(f'HellaSwag accuracy: {num_correct_norm} / {num_total} = {acc_norm:.4f}')
            wnb.log({'HellaSwag accuracy': acc_norm})

    if step > 0 and step % generate_every_x_steps == 0 or last_step:
        model.eval()
        raw_model.test_dialogue_custom(
            test_generation_prompts,
            max_gen_len=max_test_gen_len,
            device=device,
            is_instruct=is_instruct_training
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

        if is_model_distillation and teacher_model:
            with torch.no_grad():
                teacher_result = teacher_model(input_ids=x)
                teacher_logits = teacher_result.logits

            loss = distillation_loss(teacher_logits, logits, temperature=distillation_temperature)

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
    barrier()
    destroy_process_group()

wnb.finish()
