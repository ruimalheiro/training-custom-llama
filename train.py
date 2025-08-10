import torch
import torch.nn.functional as F
import torch.distributed as dist
import argparse
import time
import math
import copy
import json

from pathlib import Path
from tokenizer import init_tokenizer
from dataloaders import init_data_loaders
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from lora import apply_lora
from lr_schedulers import cosine_scheduler
from distillation_utils import distillation_loss
from wnb_utils import WnbWrapper

from config import (
    config,
    TrainingStage
)
from torch.distributed import (
    broadcast,
    barrier,
    destroy_process_group
)
from ddp_utils import (
    init_multi_gpu,
    prepare_model_for_ddp
)
from model import (
    Transformer,
    ModelConfig
)
from model_utils import (
    print_dict,
    save_model,
    load_model
)
from hellaswag_utils import (
    iterate_hellaswag_val_examples,
    prepare_hellaswag_example,
    estimate_correct_candidate_selection
)
from dpo_utils import (
    dpo_log_probs,
    dpo_loss
)


#### CONFIGURATION

# set training stage
training_stage = config.training_stage

is_pretraining = True if training_stage == TrainingStage.PRETRAIN else False
is_instruct_training = True if training_stage == TrainingStage.INSTRUCT else False
is_dpo_training = True if training_stage == TrainingStage.DPO else False

# datasets path / save checkpoints path
if is_pretraining:
    dataloader_root_path = config.pretrain_dataloader_root_path
    save_checkpoints_path = config.pretrain_save_checkpoints_path
elif is_instruct_training:
    dataloader_root_path = config.instruct_dataloader_root_path
    save_checkpoints_path = config.instruct_save_checkpoints_path
elif is_dpo_training:
    dataloader_root_path = config.dpo_dataloader_root_path
    save_checkpoints_path = config.dpo_save_checkpoints_path
else:
    raise ValueError('Invalid training stage')

hellaswag_path = config.hellaswag_path

# load path
pretrain_checkpoints_path = config.pretrain_load_checkpoints_path
instruct_checkpoints_path = config.instruct_load_checkpoints_path
dpo_checkpoints_path = config.dpo_load_checkpoints_path

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
dpo_beta = config.dpo_beta
is_model_distillation = config.is_model_distillation
distillation_temperature = config.distillation_temperature
# The teacher model is loaded via huggingface API: AutoModelForCausalLM.from_pretrained(teacher_model_checkpoint, ...) so needs to be a valid checkpoint.
teacher_model_checkpoint = config.teacher_model_checkpoint
lora_enabled = config.lora_enabled
lora_rank = config.lora_rank
lora_alpha = config.lora_alpha
lora_dropout = config.lora_dropout
lora_target_modules = config.lora_target_modules

# validation
validate_every_x_steps = config.validate_every_x_steps
val_steps = config.val_steps
hellaswag_every_x_steps = config.hellaswag_every_x_steps
hellaswag_number_of_examples = config.hellaswag_number_of_examples
generate_every_x_steps = config.generate_every_x_steps
max_test_gen_len = config.max_test_gen_len

# test prompts
test_prompts_data = json.loads(Path(config.test_prompts_path).read_text())

test_generation_prompts = test_prompts_data[training_stage.value]

# Init the tokenizer
tokenizer = init_tokenizer(config.tokenizer_checkpoint_path, config.huggingface_tokenizer)

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
    max_seq_len=config.max_seq_len,
    # tokenizer aux config
    tokenizer = tokenizer,
    vocab_size = tokenizer.vocab_size,
    pad_token_id = tokenizer.pad_id,
    stop_tokens = tokenizer.stop_tokens
)

# Extra metadata to store when saving a checkpoint
extra_checkpoint_metadata = {
    'training_stage': training_stage.value,
    'lora_enabled': lora_enabled
}

#### SCRIPT OPTIONS
parser = argparse.ArgumentParser(description='Script options')
parser.add_argument('--pretrain_checkpoint', type=str, default=None, help='Pretrain checkpoint to load.')
parser.add_argument('--instruct_checkpoint', type=str, default=None, help='Instruct checkpoint to load.')
parser.add_argument('--dpo_checkpoint', type=str, default=None, help='DPO checkpoint to load.')
parser.add_argument('--reset-optimizer', action='store_true', help='Reset the optimizer state when loading a checkpoint.')
parser.add_argument('--start-step', type=int, default=None, help='Starting step number for training.')

args = parser.parse_args()

#### INIT DISTRIBUTED DATA PARALLEL (DDP)
ddp, ddp_rank, ddp_local_rank, ddp_world_size, is_master_process, device, device_type = init_multi_gpu(seed=42)

#### INIT WnB wrapper
wnb = WnbWrapper(enabled=wnb_enabled, is_master_process=is_master_process)
wnb.init(wnb_project_name, config={
    'batch_size': model_config.max_batch_size,
    'sequence_length': model_config.max_seq_len,
    'min_learning_rate': min_lr,
    'max_learning_rate': max_lr,
})

#### LOAD CHECKPOINT
load_checkpoints_path = None
checkpoint = None
if args.pretrain_checkpoint:
    load_checkpoints_path = pretrain_checkpoints_path
    checkpoint = args.pretrain_checkpoint
elif args.instruct_checkpoint:
    load_checkpoints_path = instruct_checkpoints_path
    checkpoint = args.instruct_checkpoint
elif args.dpo_checkpoint:
    load_checkpoints_path = dpo_checkpoints_path
    checkpoint = args.dpo_checkpoint

# defaults
loaded_model_state = None
loaded_optimizer_state = None
start_step = 0
best_val_loss = float('inf')
loaded_train_loader_state = None
loaded_val_loader_state = None
is_lora_checkpoint = False

if checkpoint is not None:
    (
        loaded_model_state,
        loaded_optimizer_state,
        start_step,
        best_loss,
        loaded_train_loader_state,
        loaded_val_loader_state,
        loaded_extra_checkpoint_metadata
    ) = load_model(
        load_checkpoints_path,
        checkpoint,
        reset_optimizer=args.reset_optimizer,
        force_start_step=args.start_step,
        is_master_process=is_master_process
    )

    if best_loss < best_val_loss and not args.reset_optimizer:
        best_val_loss = best_loss

    if loaded_extra_checkpoint_metadata.get('training_stage', None) != training_stage:
        if is_master_process:
            print('** WARNING: Training stage has chanded **')
        if not args.start_step:
            if is_master_process:
                print('ignoring stored start step...')
            start_step = 0
        if loaded_train_loader_state is not None and loaded_val_loader_state is not None:
            if is_master_process:
                print('ignoring stored metada for dataset...')
            loaded_train_loader_state = None
            loaded_val_loader_state = None
        if loaded_optimizer_state is not None:
            if is_master_process:
                print('ignoring stored state of optimizer...')
            loaded_optimizer_state = None
        if is_master_process:
            print('\n')

    is_lora_checkpoint = loaded_extra_checkpoint_metadata.get('lora_enabled', False)

#### INIT DATA LOADERS
train_loader, val_loader = init_data_loaders(
    batch_size=model_config.max_batch_size,
    sequence_length=model_config.max_seq_len,
    is_master_process=is_master_process,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    data_root=dataloader_root_path,
    pad_id=model_config.pad_token_id,
    training_stage=training_stage
)

if loaded_train_loader_state is not None and loaded_val_loader_state is not None:
    train_loader.load_state_dict(loaded_train_loader_state)
    val_loader.load_state_dict(loaded_val_loader_state)

#### INIT MODEL AND TRAINING SETUP
model = Transformer(model_config)

if checkpoint and loaded_model_state:
    if is_lora_checkpoint:
        apply_lora(
            model,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=lora_target_modules,
            device=device,
            is_master_process=is_master_process
        )
    model.load_state_dict(loaded_model_state)
    if is_master_process:
        print('\nModel loading')
        print('----------------------------------------')
        print('Model checkpoint loaded and ready')

if lora_enabled and not is_lora_checkpoint:
    apply_lora(
        model,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        target_modules=lora_target_modules,
        device=device,
        is_master_process=is_master_process
    )

model.to(device)

torch.set_float32_matmul_precision('high')

#### BATCH SIZE ASSERTIONS

# NOTE: total_batch_size is the total batch size in tokens. The model max_batch_size is the number of sequences per device during forward pass (micro batches).
# The total batch size must be a multiple of (max_batch_size * max_seq_len * ddp_world_size). This is needed for the gradient accumulation steps to be calculated correctly.
assert total_batch_size % (model_config.max_batch_size * model_config.max_seq_len * ddp_world_size) == 0, 'total_batch_size must be divisible by (max_batch_size * max_seq_len * ddp_world_size)'

# Gradient accumulation steps
grad_accum_steps = total_batch_size // (model_config.max_batch_size * model_config.max_seq_len * ddp_world_size)

# Final check to validate previous calculations.
assert total_batch_size == (model_config.max_batch_size * model_config.max_seq_len * ddp_world_size * grad_accum_steps)

#### INIT OPTIMIZER
optimizer = model.configure_adamw_optimizer(
    weight_decay=weight_decay,
    learning_rate=max_lr,
    betas=(0.9, 0.95),
    eps=1e-8,
    device=device,
    is_master_process=is_master_process
)
if loaded_optimizer_state is not None:
    assert type(loaded_optimizer_state) == dict
    optimizer.load_state_dict(loaded_optimizer_state)

    # This is to ensure the optimiser state respect the device for all params
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device=p.device, dtype=p.dtype)

    if is_master_process:
        print('optimizer state loaded and ready')

model, raw_model = prepare_model_for_ddp(model, ddp_local_rank)

# Model distillation setup
teacher_model = None
if is_model_distillation:
    print(f'Loading teacher model on gpu: {ddp_rank}...')
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_checkpoint, cache_dir='./cache').to(device)
    print(f'Finished loading teacher model on gpu: {ddp_rank}...')

# DPO (Direct Preference Optimization) reference model setup
if is_dpo_training:
    print(f'Preparing DPO reference model...')
    dpo_ref_model = copy.deepcopy(model).eval().to(device)
    for p in dpo_ref_model.parameters():
        p.requires_grad = False
    print(f'Finished preparing DPO reference model')

#### CONFIG SUMMARY

total_tokens = train_loader.calculate_max_tokens()
model_params = model.get_parameters_count()
complete_max_steps = math.ceil(total_tokens / total_batch_size)

if is_master_process:
    current_lr = optimizer.param_groups[0]['lr']
    scheduled_lr = cosine_scheduler(start_step, min_lr, max_lr, warmup_steps, max_steps)
    print(f'LR stored in checkpoint: {current_lr:.4e}')
    print(f'LR that will be applied for step {start_step}: {scheduled_lr:.4e}')
    print(f'\n{training_stage.upper()} configuration:')
    print('----------------------------------------')
    print(f'dataloader data path: "{dataloader_root_path}"')
    print(f'HellaSwag data path: "{hellaswag_path}"')

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

    if is_pretraining or is_instruct_training:
        # For pretraining according to the Chinchilla paper ~20.0 is reasonable. For instruct: ~0.2 to ~0.5 is reasonable
        m_factor = 20.0 if is_pretraining else 0.3
        tokens_required_for_model_size = int(model_params * m_factor)
        steps_needed = math.ceil(tokens_required_for_model_size / total_batch_size)
        tokens_coverage = max_steps * model_config.max_batch_size * model_config.max_seq_len * ddp_world_size * grad_accum_steps
        print(f'model parameter count: {model_params}')
        print(f'number of tokens in the dataset: {total_tokens}')
        print(f'full dataset steps: {complete_max_steps}')
        print(f'heuristic token target [model parameter count * {m_factor}]: {tokens_required_for_model_size}')
        print(f'dataset covers heuristic? {"YES" if total_tokens >= tokens_required_for_model_size else "NO"}')
        print(f'number of steps needed for target: {steps_needed}')
        print(f'configured "max steps" corresponds to {round((max_steps / complete_max_steps) * 100,2)}% of total tokens (~{tokens_coverage})')
        print(f'configured "max steps" covers heuristic? {"YES" if max_steps >= steps_needed else "NO"}')

    if is_dpo_training:
        print(f'DPO beta: {dpo_beta}')

    print(f'early stopping patience: {early_stopping_patience}')

    if is_model_distillation:
        print(f'performing model distillation: {is_model_distillation}')
        print(f'distillation temperature set to: {distillation_temperature}')
        print(f'teacher model checkpoint: {teacher_model_checkpoint}')

    print('\nEvaluation Config')
    print('----------------------------------------')
    print(f'number of steps between validation: {validate_every_x_steps}')
    print(f'number of validating steps: {val_steps}')
    print(f'number of steps between HellaSwag validation: {hellaswag_every_x_steps}')
    print(f'number of HellaSwag examples: {hellaswag_number_of_examples}')
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
        print_dict(model_config.to_dict())

#### TRAINING LOOP
if ddp:
    barrier(device_ids=[ddp_local_rank])
print(f'\nGPU: {ddp_rank} is ready.')

tqdm_label = f'Training ({training_stage.value})'

# max_steps not set
if max_steps == -1:
    max_steps = complete_max_steps

epochs_no_improve = 0
abort_if_no_improve = torch.tensor([0], device=device)
early_stopping_patience_skip_steps += start_step
for step in tqdm(range(start_step, max_steps), initial=start_step, total=max_steps, desc=tqdm_label):
    if abort_if_no_improve.item() == 1:
        print(f'Rank {ddp_rank} received stop signal.')
        break

    last_step = (step == max_steps - 1)

    if step > 0 and step % validate_every_x_steps == 0 or last_step:
        model.eval()

        val_loss_sum = torch.tensor(0.0, device=device)
        val_tok_sum = torch.tensor(0.0, device=device)

        dpo_metrics = None
        with torch.no_grad():
            for _ in tqdm(range(val_steps), 'Validating'):
                if is_dpo_training:
                    # x, y, z = prompt, chosen, rejected
                    x, y, z = val_loader.next_batch()
                    x, y, z = x.to(device), y.to(device), z.to(device)

                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        policy_log_probs_pos = dpo_log_probs(model, x, y)
                        policy_log_probs_neg = dpo_log_probs(model, x, z)
                        reference_log_probs_pos = dpo_log_probs(dpo_ref_model, x, y)
                        reference_log_probs_neg = dpo_log_probs(dpo_ref_model, x, z)

                    loss, dpo_metrics = dpo_loss(
                        policy_log_probs_pos,
                        policy_log_probs_neg,
                        reference_log_probs_pos,
                        reference_log_probs_neg,
                        dpo_beta
                    )
                    n_valid = torch.tensor(x.size(0), device=device) # Assume 1 valid example as the entire triple.
                else:
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
            wnb_metrics = {'Validation Loss': val_ce}
            if is_dpo_training:
                print(dpo_metrics['str'])
                wnb_metrics.update(dpo_metrics['wnb'])
            wnb.log(wnb_metrics)

            if val_ce < best_val_loss:
                best_val_loss = val_ce
                epochs_no_improve = 0

                if save_checkpoints is True:
                    save_model(save_checkpoints_path, raw_model, model_config, step, val_ce, optimizer, train_loader, val_loader, extra_checkpoint_metadata)
            else:
                if step > early_stopping_patience_skip_steps:
                    epochs_no_improve += 1
                    print(f'Validation loss did not improve. Best: {best_val_loss}, Latest: {val_ce} - Attempts left: {early_stopping_patience - epochs_no_improve}')
                else:
                    print(f'Validation loss did not improve. Best: {best_val_loss}, Latest: {val_ce} - (Skip phase...) steps left to skip: {early_stopping_patience_skip_steps - step}')

                print('Skipping save checkpoint...')

            stop_signal = torch.tensor([0], device=device)
            if epochs_no_improve == early_stopping_patience:
                print(f'The validation loss did not improve for: {early_stopping_patience} - Aborting training...')
                abort_if_no_improve[0] = 1

        if ddp:
            broadcast(abort_if_no_improve, src=0)

    if is_pretraining and (step > 0 and step % hellaswag_every_x_steps == 0 or last_step):
        model.eval()
        num_correct_norm = 0
        num_total = 0
        for i, example in tqdm(enumerate(iterate_hellaswag_val_examples(hellaswag_path, size=hellaswag_number_of_examples)), 'HellaSwag validation', unit=' examples'):
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
    t0 = time.time()
    train_loss_local_sum = 0.0
    train_loss_local_token_sum = 0
    train_local_token_sum = 0
    dpo_metrics = None
    for micro_step in range(grad_accum_steps):
        if is_dpo_training:
            # x, y, z = prompt, chosen, rejected
            x, y, z = train_loader.next_batch()
            x, y, z = x.to(device), y.to(device), z.to(device)

            train_local_token_sum += 4 * x.numel() + 2 * y.numel() + 2 * z.numel()

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                policy_log_probs_pos = dpo_log_probs(model, x, y)
                policy_log_probs_neg = dpo_log_probs(model, x, z)

            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    reference_log_probs_pos = dpo_log_probs(dpo_ref_model, x, y)
                    reference_log_probs_neg = dpo_log_probs(dpo_ref_model, x, z)

            loss, dpo_metrics = dpo_loss(
                policy_log_probs_pos,
                policy_log_probs_neg,
                reference_log_probs_pos,
                reference_log_probs_neg,
                dpo_beta
            )

            loss_scaled = loss / grad_accum_steps

            n_valid = x.size(0) # Assume 1 valid example as the entire triple.
        else:
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            train_local_token_sum += x.numel()

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                result = model(x, labels=y)
                loss = result['loss']

            loss_scaled = loss / grad_accum_steps
            if is_model_distillation and teacher_model:
                train_local_token_sum += x.numel()

                with torch.no_grad():
                    # NOTE: The vocabularies must match otherwise there will be an error.
                    teacher_logits = teacher_model(input_ids=x).logits

                loss_distil = distillation_loss(teacher_logits, result['logits'], temperature=distillation_temperature)
                loss_scaled += loss_distil / grad_accum_steps

            n_valid = (y != -100).sum().item()

        train_loss_local_sum += loss.item() * n_valid
        train_loss_local_token_sum  += n_valid

        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

        loss_scaled.backward()

    train_loss_sum = train_loss_local_sum
    train_loss_token_sum = train_loss_local_token_sum
    train_token_sum = train_local_token_sum

    if ddp:
        reduce_vec = torch.tensor([train_loss_local_sum, train_loss_local_token_sum, train_token_sum], device=device, dtype=torch.float64)
        dist.all_reduce(reduce_vec, op=dist.ReduceOp.SUM)

        train_loss_sum = float(reduce_vec[0].item())
        train_loss_token_sum = float(reduce_vec[1].item())
        train_token_sum = float(reduce_vec[2].item())

    train_avg_loss = train_loss_sum / train_loss_token_sum

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = cosine_scheduler(step, min_lr, max_lr, warmup_steps, max_steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()

    torch.cuda.synchronize()

    t1 = time.time()
    dt = (t1 - t0)
    tokens_per_sec = train_token_sum / dt

    if is_master_process:
        print(f'step: {step:4d} | train loss: {train_avg_loss:.4f} | last val loss: {best_val_loss:.4f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}s | tok/sec: {tokens_per_sec:.2f}')
        wnb_metrics = {'Train Loss': train_avg_loss}
        if is_dpo_training:
            print(dpo_metrics['str'])
            wnb_metrics.update(dpo_metrics['wnb'])
        wnb.log(wnb_metrics)

if ddp:
    barrier(device_ids=[ddp_local_rank])
    destroy_process_group()

wnb.finish()
