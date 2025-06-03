import torch
import torch.nn.functional as F
import torch.distributed as dist
import argparse
import time
import math

from torch.distributed import broadcast, barrier, destroy_process_group
from tokenizer import init_tokenizer
from model import Transformer, ModelConfig
from model_utils import print_model_config, save_model, load_model, init_multi_gpu, prepare_model_for_ddp, WnbWrapper
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

# datasets path / save checkpoints path
if config.is_instruct_training:
    dataloader_root_path = config.instruct_dataloader_root_path
    save_checkpoints_path = config.instruct_save_checkpoints_path
else:
    dataloader_root_path = config.pretrain_dataloader_root_path
    save_checkpoints_path = config.pretrain_save_checkpoints_path

hellaswag_path = config.hellaswag_path

# load path
pretrain_checkpoints_path = config.pretrain_load_checkpoints_path
instruct_checkpoints_path = config.instruct_load_checkpoints_path

# save toggle
save_checkpoints = config.save_checkpoints

# wnb
wnb_enabled = config.wnb_enabled
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
early_stopping_patience_skip_steps = config.early_stopping_patience_skip_steps
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
    multiple_of=config.multiple_of,
    ffn_dim_multiplier=config.ffn_dim_multiplier,
    norm_eps=config.norm_eps,
    rope_theta=config.rope_theta,
    max_batch_size=config.max_batch_size,
    max_seq_len=config.max_seq_len
)

##################################################

# Can be augmented with more useful options.
parser = argparse.ArgumentParser(description='Training Script')
parser.add_argument('--pretrain_checkpoint', type=str, default=None, help='Pretrain checkpoint to load.')
parser.add_argument('--instruct_checkpoint', type=str, default=None, help='Instruct checkpoint to load.')
parser.add_argument('--reset-optimizer', action='store_true', help='Reset the optimizer state when loading a checkpoint.')
parser.add_argument('--start-step', type=int, default=None, help='Starting step number for training.')

args = parser.parse_args()

ddp, ddp_rank, ddp_local_rank, ddp_world_size, is_master_process, device, device_type = init_multi_gpu(seed=42)

tokenizer = init_tokenizer(config.tokenizer_checkpoint_path, config.huggingface_tokenizer)

model_config.tokenizer = tokenizer
model_config.vocab_size = tokenizer.vocab_size
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
    is_instruct_training=is_instruct_training,
    pad_id=model_config.pad_token_id
)

assert total_batch_size % (model_config.max_batch_size * model_config.max_seq_len) == 0

grad_accum_steps = total_batch_size // (model_config.max_batch_size * model_config.max_seq_len * ddp_world_size)

assert total_batch_size == (model_config.max_batch_size * model_config.max_seq_len * ddp_world_size * grad_accum_steps)

total_tokens = train_loader.calculate_max_tokens()
model_params = raw_model.get_parameters_count()
complete_max_steps = math.ceil(total_tokens / total_batch_size)

# max_steps not set
if max_steps == -1:
    max_steps = complete_max_steps

test_generation_prompts = test_pretrain_generation_prompts
if is_instruct_training:
    test_generation_prompts = test_instruct_generation_prompts

load_checkpoints_path = None
checkpoint = None
if args.pretrain_checkpoint:
    load_checkpoints_path = pretrain_checkpoints_path
    checkpoint = args.pretrain_checkpoint
elif args.instruct_checkpoint:
    load_checkpoints_path = instruct_checkpoints_path
    checkpoint = args.instruct_checkpoint

if is_master_process:
    if is_instruct_training:
        print('\nSFT configuration:')
    else:
        print('\nPretrain configuration:')
    print('----------------------------------------')
    print(f'dataloader data path: "{dataloader_root_path}"')
    print(f'hellaswag data path: "{hellaswag_path}"')

    if checkpoint is not None:
        print(f'loading checkpoint data path: "{load_checkpoints_path}"')

    if save_checkpoints:
        print(f'saving checkpoint data path: "{save_checkpoints_path}"')

    if wnb_enabled:
        print(f'weights and biases project name: "{wnb_project_name}"')

    print(f'tokenizer loaded from: "{tokenizer_checkpoint_path}"')

    print(f'total batch size: {total_batch_size}')
    print(f'max learning rate: {max_lr}')
    print(f'min learning rate: {min_lr}')
    print(f'warmup steps: {warmup_steps}')
    print(f'weight decay: {weight_decay}')
    print(f'max steps: {max_steps}')
    if is_instruct_training:
        m_factor = 0.3 # 0.2–0.5 is typical
    else:
        m_factor = 20.0 # Chinchilla
    tokens_required_for_model_size = int(model_params * m_factor)
    steps_needed = math.ceil(tokens_required_for_model_size / total_batch_size)
    tokens_coverage = max_steps * model_config.max_batch_size * model_config.max_seq_len * ddp_world_size * grad_accum_steps
    print(f'model parameter count: {model_params}')
    print(f'number of tokens in the dataset: {total_tokens}')
    print(f'full dataset steps: {complete_max_steps}')
    print(f'heuristic token target [model parameter count * {m_factor}]): {tokens_required_for_model_size}')
    print(f'dataset covers heuristic? {"YES" if total_tokens >= tokens_required_for_model_size else "NO"}')
    print(f'number of steps needed for target: {steps_needed}')
    print(f'configured "max steps" corresponds to {round((max_steps / complete_max_steps) * 100,2)}% of total tokens (~{tokens_coverage})')
    print(f'configured "max steps" covers heuristic? {"YES" if max_steps >= steps_needed else "NO"}')

    print(f'early stopping patience: {early_stopping_patience}')
    if is_instruct_training:
        print(f'using instruct format: {is_instruct_training}')
    if is_model_distillation:
        print(f'performing model distillation: {is_model_distillation}')
        print(f'distillation temperature set to: {distillation_temperature}')
        print(f'teacher model checkpoint: {teacher_model_checkpoint}')

    print('\nEvaluation Config')
    print('----------------------------------------')
    print(f'number of steps between validation: {validate_every_x_steps}')
    print(f'number of validating steps: {val_steps}')
    print(f'number of steps between hellaswag validation: {hellaswag_every_x_steps}')
    print(f'number of hellaswag examples: {hellagswag_number_of_examples}')
    print(f'number of steps between model output generations: {generate_every_x_steps}')
    print(f'max length for the generated text from each prompt: {max_test_gen_len}')
    print(f'generation prompts:')
    for example in test_generation_prompts:
        print(f'=> "{example}"')

    print('\nDerived properties')
    print('----------------------------------------')
    print(f'gradient accumulation steps: {grad_accum_steps}')

    if checkpoint is None:
        print('\nModel config')
        print('----------------------------------------')
        print_model_config(model_config.to_dict())

teacher_model = None
if is_model_distillation:
    print(f'Loading teacher model on gpu: {ddp_rank}...')
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_checkpoint, cache_dir='./cache').to(device)
    print(f'Finished loading teacher model on gpu: {ddp_rank}...')

optimizer = raw_model.configure_optimizers(weight_decay=weight_decay, learning_rate=max_lr, device=device, is_master_process=is_master_process)

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

start_step = 0
best_val_loss = float('inf')
if checkpoint is not None:
    model, optimizer, start_step, best_loss, train_loader_state, val_loader_state = load_model(
        load_checkpoints_path,
        checkpoint,
        raw_model,
        optimizer=optimizer,
        reset_optimizer=args.reset_optimizer,
        force_start_step=args.start_step,
        is_master_process=is_master_process
    )

    if best_loss < best_val_loss and not args.reset_optimizer:
        best_val_loss = best_loss

    if train_loader_state is not None and val_loader_state is not None:
        train_loader.load_state_dict(train_loader_state)
        val_loader.load_state_dict(val_loader_state)

    if is_master_process:
        current_lr = optimizer.param_groups[0]['lr']
        scheduled_lr = get_lr(start_step)
        print(f'LR stored in checkpoint: {current_lr:.4e}')
        print(f'LR that will be applied for step {start_step}: {scheduled_lr:.4e}')

wnb = WnbWrapper(enabled=wnb_enabled, is_master_process=is_master_process)
wnb.init(wnb_project_name, config={
    'batch_size': model_config.max_batch_size,
    'sequence_length': model_config.max_seq_len,
    'min_learning_rate': min_lr,
    'max_learning_rate': max_lr,
})

def distillation_loss(teacher_logits, student_logits, temperature=1.0):
    teacher_probabilities = F.softmax(teacher_logits.view(-1, teacher_logits.size(-1)) / temperature, dim=-1)
    student_log_probabilities = F.log_softmax(student_logits / temperature, dim=-1)

    kl_divergence = F.kl_div(student_log_probabilities, teacher_probabilities, reduction='batchmean') * (temperature ** 2)
    return kl_divergence

if ddp:
    barrier()
print(f'\nGPU: {ddp_rank} is ready.')

tqdm_label = 'Training'
if is_instruct_training:
    tqdm_label = 'Training (SFT)'

if is_model_distillation:
    tqdm_label += ' (Distil)'

epochs_no_improve = 0
abort_if_no_improve = torch.tensor([0], device=device)
early_stopping_patience_skip_steps += start_step
for step in tqdm(range(start_step, max_steps), initial=start_step, total=max_steps, desc=tqdm_label):
    if abort_if_no_improve.item() == 1:
        print(f'Rank {ddp_rank} received stop signal.')
        break

    t0 = time.time()
    last_step = (step == max_steps - 1)

    if step > 0 and step % validate_every_x_steps == 0 or last_step:
        model.eval()

        val_loss_sum = torch.tensor(0.0, device=device)
        val_tok_sum = torch.tensor(0.0, device=device)

        with torch.no_grad():
            for _ in tqdm(range(val_steps), 'Validating'):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    loss = model(x, labels=y)['loss']

                n_valid = (y != -100).sum().float()
                val_loss_sum += loss * n_valid
                val_tok_sum += n_valid

        if ddp:
            dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_tok_sum, op=dist.ReduceOp.SUM)

        val_ce = (val_loss_sum / val_tok_sum).item()

        if is_master_process:
            print(f'validation loss: {val_ce:.4f}')

            if val_ce < best_val_loss:
                best_val_loss = val_ce
                epochs_no_improve = 0

                if save_checkpoints is True:
                    save_model(save_checkpoints_path, raw_model, model_config, step, val_ce, optimizer, train_loader, val_loader)
            else:
                if step > early_stopping_patience_skip_steps:
                    epochs_no_improve += 1
                    print(f'Validation loss did not improve. Best: {best_val_loss}, Latest: {val_ce} - Attempts left: {early_stopping_patience - epochs_no_improve}')
                else:
                    print(f'Validation loss did not improve. Best: {best_val_loss}, Latest: {val_ce} - (Skip phase...) steps left to skip: {early_stopping_patience_skip_steps - step}')

                print('Skipping save checkpoint...')

            wnb.log({'Validation Loss': val_ce})

            stop_signal = torch.tensor([0], device=device)
            if epochs_no_improve == early_stopping_patience:
                print(f'The validation loss did not improve for: {early_stopping_patience} - Aborting training...')
                abort_if_no_improve[0] = 1

        if ddp:
            broadcast(abort_if_no_improve, src=0)

    if not is_instruct_training and step > 0 and step % hellaswag_every_x_steps == 0 or (not is_instruct_training and last_step):
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
            is_instruct=is_instruct_training,
            temperature=0.0,
            top_p=1.0
        )

    model.train()
    optimizer.zero_grad()
    train_loss_local_sum = 0.0
    train_tok_local_sum = 0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            result = model(x, labels=y)
            loss = result['loss']

        loss_scaled = loss / grad_accum_steps
        if is_model_distillation and teacher_model:
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids=x).logits

            loss_distil = distillation_loss(teacher_logits, result['logits'], temperature=distillation_temperature)
            loss_scaled += loss_distil / grad_accum_steps

        n_valid = (y != -100).sum().item()
        train_loss_local_sum += loss.item() * n_valid
        train_tok_local_sum  += n_valid

        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

        loss_scaled.backward()

    train_loss_sum = train_loss_local_sum
    train_tok_sum = train_tok_local_sum

    if ddp:
        loss_sum = torch.tensor(train_loss_local_sum, device=device)
        tok_sum  = torch.tensor(train_tok_local_sum,  device=device)

        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(tok_sum,  op=dist.ReduceOp.SUM)

        train_loss_sum = loss_sum.item()
        train_tok_sum  = tok_sum.item()

    train_ce = train_loss_sum / train_tok_sum

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()

    torch.cuda.synchronize()

    t1 = time.time()
    dt = (t1 - t0)

    if not is_instruct_training:
        tokens_per_sec = (train_loader.B * train_loader.S * grad_accum_steps * ddp_world_size) / dt
    else:
        tokens_per_sec = train_tok_sum  / dt

    if is_master_process:
        print(f'step: {step:4d} | train loss: {train_ce:.4f} | last val loss: {best_val_loss:.4f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}s | tok/sec: {tokens_per_sec:.2f}')
        wnb.log({'Train Loss': train_ce})

if ddp:
    barrier()
    destroy_process_group()

wnb.finish()
