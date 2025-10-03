import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
import argparse
import math
import copy
import json
import inspect

from config import (
    config,
    TrainingStage,
    TrainingPrecision
)
os.environ['HF_HOME'] = config.hf_home
os.environ['HF_DATASETS_CACHE'] = f'{config.hf_home}/datasets'
os.environ['HF_HUB_CACHE'] = f'{config.hf_home}/hub'

from pathlib import Path
from tokenizer import init_tokenizer
from dataloaders import init_data_loaders
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from lora import apply_lora
from lr_schedulers import cosine_scheduler
from distillation_utils import distillation_loss
from wandb_utils import WandbWrapper
from contextlib import nullcontext
from torch.distributed.fsdp import MixedPrecision

from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler
)
from torch.distributed import (
    broadcast,
    destroy_process_group
)
from ddp_utils import (
    init_multi_gpu,
    prepare_model_for_ddp,
    prepare_model_for_fsdp
)
from model import (
    Transformer,
    ModelConfig
)
from model_utils import (
    print_dict,
    save_checkpoint,
    load_checkpoint,
    load_model_state,
    load_optimizer_state,
    clip_grad_norm
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
max_number_checkpoints = config.max_number_checkpoints

# wandb
wandb_enabled = config.wandb_enabled
wandb_project_name = config.wandb_project_name

# tokenizer model path
tokenizer_checkpoint_path = config.tokenizer_checkpoint_path

# value to mask the padded tokens
ignore_index = config.ignore_index

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
use_torch_compile = config.use_torch_compile
use_fsdp = config.use_fsdp
fsdp_sharding_strategy = config.fsdp_sharding_strategy

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

# model config

assert config.dim % config.n_heads == 0, f'"dim" ({config.dim}) must be divisible by "n_heads" ({config.n_heads})'
assert config.n_kv_heads <= config.n_heads, f'"n_kv_heads" ({config.n_kv_heads}) must be less or equal to "n_heads" ({config.n_heads})'
assert config.n_heads % config.n_kv_heads == 0, f'"n_heads" ({config.n_heads}) must be divisible by n_kv_heads" ({config.n_kv_heads})'

model_config = ModelConfig(
    dim=config.dim,
    n_layers=config.n_layers,
    n_heads=config.n_heads,
    n_kv_heads=config.n_kv_heads,
    multiple_of=config.multiple_of,
    ffn_dim_multiplier=config.ffn_dim_multiplier,
    norm_eps=config.norm_eps,
    is_rope_cis=config.is_rope_cis,
    rope_theta=config.rope_theta,
    max_batch_size=config.max_batch_size,
    max_seq_len=config.max_seq_len,
    # tokenizer aux config
    tokenizer = tokenizer,
    vocab_size = tokenizer.vocab_size,
    pad_token_id = tokenizer.pad_id,
    stop_tokens = tokenizer.stop_tokens,
    ignore_index = ignore_index
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

# print helper
def log(content, force=False):
    if is_master_process or force:
        print(content)

#### TRAINING PRECISION, AUTOCAST AND SCALER
if config.training_precision == TrainingPrecision.BF16:
    use_autocast = True
    scaler = None
    model_dtype = torch.float32
    autocast_dtype = torch.bfloat16
    fsdp_mp = MixedPrecision(
        param_dtype=torch.float32,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16
    ) if use_fsdp else None
elif config.training_precision == TrainingPrecision.FP16:
    use_autocast = True
    scaler = torch.amp.GradScaler(device_type) # need gradscaler when fp16
    model_dtype = torch.float32
    autocast_dtype = torch.float16
    fsdp_mp = MixedPrecision(
        param_dtype=torch.float32,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16
    ) if use_fsdp else None
elif config.training_precision == TrainingPrecision.FP32:
    use_autocast = False
    scaler = None
    model_dtype = torch.float32
    autocast_dtype = torch.float32
    fsdp_mp = None
else:
    raise ValueError('Invalid training precision')

#### INIT WANDB wrapper
wandb = WandbWrapper(enabled=wandb_enabled, is_master_process=is_master_process)
wandb.init(wandb_project_name, config={
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
    ) = load_checkpoint(
        load_checkpoints_path,
        checkpoint,
        reset_optimizer=args.reset_optimizer,
        force_start_step=args.start_step,
        is_master_process=is_master_process
    )

    if best_loss < best_val_loss and not args.reset_optimizer:
        best_val_loss = best_loss

    if loaded_extra_checkpoint_metadata.get('training_stage', None) != training_stage:
        log('** WARNING: Training stage has chanded **')
        if not args.start_step:
            log('ignoring stored start step...')
            start_step = 0
        if loaded_train_loader_state is not None and loaded_val_loader_state is not None:
            log('ignoring stored metada for dataset...')
            loaded_train_loader_state = None
            loaded_val_loader_state = None
        if loaded_optimizer_state is not None:
            log('ignoring stored state of optimizer...')
            loaded_optimizer_state = None
        log('\n')

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
    load_model_state(model, loaded_model_state, use_fsdp)
    log('\nModel loading')
    log('----------------------------------------')
    log('Model checkpoint loaded and ready')

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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

#### BATCH SIZE ASSERTIONS

# NOTE: total_batch_size is the total batch size in tokens. The model max_batch_size is the number of sequences per device during forward pass (micro batches).
# The total batch size must be a multiple of (max_batch_size * max_seq_len * ddp_world_size). This is needed for the gradient accumulation steps to be calculated correctly.
assert total_batch_size % (model_config.max_batch_size * model_config.max_seq_len * ddp_world_size) == 0, 'total_batch_size must be divisible by (max_batch_size * max_seq_len * ddp_world_size)'

# Gradient accumulation steps
grad_accum_steps = total_batch_size // (model_config.max_batch_size * model_config.max_seq_len * ddp_world_size)

# Final check to validate previous calculations.
assert total_batch_size == (model_config.max_batch_size * model_config.max_seq_len * ddp_world_size * grad_accum_steps)

# COUNT PARAMS
model_params = model.get_parameters_count()

# DPO (Direct Preference Optimization) reference model setup
dpo_ref_model = None
if is_dpo_training:
    log(f'Preparing DPO reference model...', True)
    dpo_ref_model = copy.deepcopy(model).eval()
    for p in dpo_ref_model.parameters():
        p.requires_grad = False
    log(f'Finished preparing DPO reference model', True)

#### COMPILE
if use_torch_compile:
    model.compile()
    if is_dpo_training and dpo_ref_model:
        dpo_ref_model.compile()

#### PREPARE OPTIMIZER OPTIMAL PARAM GROUPS
optimizer_param_groups = model.build_optimizer_param_groups(weight_decay=weight_decay, is_master_process=is_master_process)

#### PREPARE DDP / FSDP
# for FSDP no need to move as that would actually cost more VRAM, instead let FSDP initialization alocate the shard to the device id (ddp_local_rank).
if use_fsdp and dist.is_initialized():
    log('\nWrapping the model in preparation for FSDP')
    model, raw_model = prepare_model_for_fsdp(model, ddp_local_rank, fsdp_mp, fsdp_sharding_strategy)
    if is_dpo_training and dpo_ref_model:
        dpo_ref_model, _ = prepare_model_for_fsdp(dpo_ref_model, ddp_local_rank, fsdp_mp, fsdp_sharding_strategy)
else:
    # move to gpu
    model.to(device=device, dtype=model_dtype)
    if is_dpo_training and dpo_ref_model:
        dpo_ref_model.to(device, dtype=model_dtype)

    log('\nWrapping the model in preparation for DDP')
    model, raw_model = prepare_model_for_ddp(model, ddp_local_rank)
    if is_dpo_training and dpo_ref_model:
        dpo_ref_model, _ = prepare_model_for_ddp(dpo_ref_model, ddp_local_rank)

#### INIT OPTIMIZER
fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_available and 'cuda' in device

optimizer = torch.optim.AdamW(
    params=optimizer_param_groups,
    lr=max_lr,
    betas=[0.9, 0.95],
    eps=1e-8,
    fused=use_fused
)
if loaded_optimizer_state is not None:
    assert type(loaded_optimizer_state) == dict
    load_optimizer_state(optimizer, model if use_fsdp else raw_model, loaded_optimizer_state, use_fsdp)

    # This is to ensure the optimiser state respect the device for all params
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device=p.device, dtype=p.dtype)

    log('optimizer state loaded and ready')

# Model distillation setup
teacher_model = None
if is_model_distillation:
    log(f'Loading teacher model on gpu: {ddp_rank}...', True)
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_checkpoint, token=config.hf_token).to(device, dtype=model_dtype).eval()
    log(f'Finished loading teacher model on gpu: {ddp_rank}...', True)

#### CONFIG SUMMARY
total_tokens = train_loader.calculate_max_tokens()
complete_max_steps = math.ceil(total_tokens / total_batch_size)

if is_master_process:
    print(f'\n{training_stage.upper()} configuration:')
    print('----------------------------------------')
    current_lr = optimizer.param_groups[0]['lr']
    scheduled_lr = cosine_scheduler(start_step, min_lr, max_lr, warmup_steps, max_steps)
    print(f'using fused AdamW: {use_fused}')
    print(f'LR set in the optimizer: {current_lr:.4e}')
    print(f'(scheduler) LR that will be applied for step {start_step}: {scheduled_lr:.4e}')
    print(f'dataloader data path: "{dataloader_root_path}"')
    print(f'HellaSwag data path: "{hellaswag_path}"')

    if checkpoint is not None:
        print(f'loading checkpoint data path: "{load_checkpoints_path}"')

    if save_checkpoints:
        print(f'saving checkpoint data path: "{save_checkpoints_path}"')

    if wandb_enabled:
        print(f'weights and biases project name: "{wandb_project_name}"')

    print(f'tokenizer loaded from: "{tokenizer_checkpoint_path}"')
    print(f'training precision: {config.training_precision.value}')
    print(f'parameter dtype: {model_dtype}')
    print(f'using autocast: {use_autocast}')
    if use_autocast:
        print(f'autocast dtype: {autocast_dtype}')
    print(f'total batch size: {total_batch_size}')
    print(f'max learning rate: {max_lr}')
    print(f'min learning rate: {min_lr}')
    print(f'warmup steps: {warmup_steps}')
    print(f'weight decay: {weight_decay}')
    print(f'max steps: {max_steps}')
    print(f'using torch compile: {use_torch_compile}')
    print(f'Using FSDP: {use_fsdp}')
    if use_fsdp:
        print(f'FSDP sharding strategy: {fsdp_sharding_strategy.value}')

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
    dist.barrier(device_ids=[ddp_local_rank])

log(f'\nGPU: {ddp_local_rank} is ready.', True)

tqdm_label = f'Training ({training_stage.value})'

# max_steps not set
if max_steps == -1:
    max_steps = complete_max_steps

epochs_no_improve = 0
abort_if_no_improve = torch.tensor([0], device=device)
early_stopping_patience_skip_steps += start_step

def trace_handler(prof):
    if config.torch_profiler_tensorboard_enabled:
        tensor_board = tensorboard_trace_handler(
            dir_name=config.torch_profiler_tensorboard_log_path,
            worker_name='rank0' if is_master_process else None,
        )
        tensor_board(prof)
    else:
        log(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10), True) # TODO can be configured.

torch_profiler_enabled = config.torch_profiler_enabled and is_master_process
if torch_profiler_enabled:
    log('\nWARN: Torch profiler is enabled!\n')
torch_profiler_context = (
    profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA
        ],
        record_shapes=True,
        profile_memory=True,
        schedule=schedule(
            skip_first=config.torch_profiler_schedule_skip_first,
            wait=config.torch_profiler_schedule_wait,
            warmup=config.torch_profiler_schedule_warmup,
            active=config.torch_profiler_schedule_active,
            repeat=config.torch_profiler_schedule_repeat
        ),
        on_trace_ready=trace_handler
    ) if torch_profiler_enabled else nullcontext()
)
with torch_profiler_context as prof:
    for step in tqdm(range(start_step, max_steps), initial=start_step, total=max_steps, desc=tqdm_label, disable=not is_master_process):
        if abort_if_no_improve.item() == 1:
            log(f'Rank {ddp_rank} received stop signal.', True)
            break

        last_step = (step == max_steps - 1)

        if (step > 0 and step % validate_every_x_steps == 0) or last_step:
            model.eval()

            val_loss_sum = torch.tensor(0.0, device=device)
            val_tok_sum = torch.tensor(0.0, device=device)

            dpo_metrics = None
            with torch.no_grad():
                for _ in tqdm(range(val_steps), 'Validating', disable=not is_master_process):
                    if is_dpo_training:
                        # x, y, z = prompt, chosen, rejected
                        x, y, z = val_loader.next_batch()
                        x, y, z = x.to(device, non_blocking=True), y.to(device, non_blocking=True), z.to(device, non_blocking=True)

                        with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=use_autocast):
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
                        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                        with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=use_autocast):
                            loss = model(x, labels=y)['loss']

                        n_valid = (y != ignore_index).sum().float()
                    val_loss_sum += loss * n_valid
                    val_tok_sum += n_valid

            if ddp:
                dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_tok_sum, op=dist.ReduceOp.SUM)

            val_ce = (val_loss_sum / val_tok_sum).item()

            if is_master_process:
                log(f'\nvalidation loss: {val_ce:.4f}')
                wandb_metrics = {'Validation Loss': val_ce}
                if is_dpo_training:
                    log(dpo_metrics['str'])
                    wandb_metrics.update(dpo_metrics['wandb'])
                wandb.log(wandb_metrics)

            if val_ce < best_val_loss:
                best_val_loss = val_ce
                epochs_no_improve = 0

                if save_checkpoints is True:
                    save_checkpoint(
                        save_checkpoints_path,
                        model if use_fsdp else raw_model,
                        model_config,
                        step,
                        val_ce,
                        optimizer,
                        train_loader,
                        val_loader,
                        extra_checkpoint_metadata,
                        max_number_checkpoints,
                        is_master_process,
                        use_fsdp
                    )
            else:
                if step > early_stopping_patience_skip_steps:
                    epochs_no_improve += 1
                    log(f'Validation loss did not improve. Best: {best_val_loss}, Latest: {val_ce} - Attempts left: {early_stopping_patience - epochs_no_improve}')
                else:
                    log(f'Validation loss did not improve. Best: {best_val_loss}, Latest: {val_ce} - (Skip phase...) steps left to skip: {early_stopping_patience_skip_steps - step}')

                log('Skipping save checkpoint...')

            if epochs_no_improve == early_stopping_patience:
                log(f'The validation loss did not improve for: {early_stopping_patience} - Aborting training...')
                abort_if_no_improve[0] = 1

            if ddp:
                broadcast(abort_if_no_improve, src=0)

        if is_pretraining and ((step > 0 and step % hellaswag_every_x_steps == 0) or last_step):
            model.eval()
            num_correct_norm = 0
            num_total = 0
            for i, example in tqdm(enumerate(iterate_hellaswag_val_examples(hellaswag_path, size=hellaswag_number_of_examples)), 'HellaSwag validation', unit=' examples', disable=not is_master_process):
                # if i % ddp_world_size == ddp_rank (gpu itself), process.
                if i % ddp_world_size != ddp_rank:
                    continue

                _, tokens, mask, label = prepare_hellaswag_example(example, tokenizer)
                tokens = tokens.to(device)
                mask = mask.to(device)

                # get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=use_autocast):
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
                log(f'HellaSwag accuracy: {num_correct_norm} / {num_total} = {acc_norm:.4f}')
                wandb.log({'HellaSwag accuracy': acc_norm})

        if (step > 0 and step % generate_every_x_steps == 0) or last_step:
            model.eval()
            raw_model.test_dialogue_custom(
                test_generation_prompts,
                max_gen_len=max_test_gen_len,
                device=device,
                is_instruct=is_instruct_training,
                temperature=0.0,
                top_p=1.0
            )

        torch.cuda.reset_peak_memory_stats()

        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss_local_sum = torch.tensor(0.0, device=device)
        train_loss_local_token_sum = torch.tensor(0.0, device=device)
        train_local_token_sum = torch.tensor(0.0, device=device)
        dpo_metrics = None
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)

        t0.record()
        for micro_step in range(grad_accum_steps):
            if is_dpo_training:
                # x, y, z = prompt, chosen, rejected
                x, y, z = train_loader.next_batch()
                x, y, z = x.to(device, non_blocking=True), y.to(device, non_blocking=True), z.to(device, non_blocking=True)

                train_local_token_sum += 4 * x.numel() + 2 * y.numel() + 2 * z.numel()

                with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=use_autocast):
                    policy_log_probs_pos = dpo_log_probs(model, x, y)
                    policy_log_probs_neg = dpo_log_probs(model, x, z)

                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=use_autocast):
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
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

                train_local_token_sum += x.numel()

                with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=use_autocast):
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

                n_valid = (y != ignore_index).sum()

            if not torch.is_tensor(n_valid):
                n_valid = torch.tensor(n_valid, device=device, dtype=loss.dtype)

            train_loss_local_sum += loss.detach() * n_valid
            train_loss_local_token_sum  += n_valid

            if ddp and not use_fsdp: # require_backward_grad_sync is not used with FSDP
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

            if scaler:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

        train_loss_sum = train_loss_local_sum
        train_loss_token_sum = train_loss_local_token_sum
        train_token_sum = train_local_token_sum

        if ddp:
            reduce_vec = torch.stack([
                train_loss_local_sum.float(),
                train_loss_local_token_sum.float(),
                train_token_sum.float()
            ])
            dist.all_reduce(reduce_vec, op=dist.ReduceOp.SUM)

            train_loss_sum, train_loss_token_sum, train_token_sum = reduce_vec.tolist()

        train_avg_loss = train_loss_sum / train_loss_token_sum

        if scaler:
            scaler.unscale_(optimizer) # due to fp16, optimizer gradients are inflated so need to unscale before clipping.
        norm = clip_grad_norm(model, 1.0, use_fsdp and ddp)

        lr = cosine_scheduler(step, min_lr, max_lr, warmup_steps, max_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if scaler:
            # The dynamic range in fp16 is low and this handles NaNs/infs which might occur more.
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        t1.record()
        t1.synchronize()
        dt = t0.elapsed_time(t1) / 1000.0
        tokens_per_sec = train_token_sum / dt

        peak_allocated_mb = torch.cuda.max_memory_allocated(ddp_local_rank) / 1024**2
        peak_reserved_mb = torch.cuda.max_memory_reserved(ddp_local_rank) / 1024**2
        current_allocated_mb = torch.cuda.memory_allocated(ddp_local_rank) / 1024**2
        current_reserved_mb = torch.cuda.memory_reserved(ddp_local_rank) / 1024**2

        if is_master_process:
            log(f'step: {step:4d} | train loss: {train_avg_loss:.4f} | val loss: {best_val_loss:.4f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}s | tok/sec: {int(tokens_per_sec)} | alloc/res MiB: (peak) {peak_allocated_mb:.0f} / {peak_reserved_mb:.0f} (curr) {current_allocated_mb:.0f} / {current_reserved_mb:.0f}')
            wandb_metrics = {
                'Train Loss': train_avg_loss,
                'Learning rate': lr,
                'Norm': norm,
                'Step time (seconds)': dt,
                'Tokens (per sec)': tokens_per_sec,
                'Peak Alloc MiB': peak_allocated_mb,
                'Peak Reserved MiB': peak_reserved_mb,
                'Alloc MiB': current_allocated_mb,
                'Reserved MiB': current_reserved_mb
            }
            if is_dpo_training:
                log(dpo_metrics['str'])
                wandb_metrics.update(dpo_metrics['wandb'])
            wandb.log(wandb_metrics)

        if torch_profiler_enabled:
            prof.step()

if ddp:
    dist.barrier(device_ids=[ddp_local_rank])
    destroy_process_group()

wandb.finish()
