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
from types import SimpleNamespace

from torch.optim import AdamW
from torch.distributed import (
    broadcast,
    destroy_process_group
)
from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler
)
from ddp_utils import (
    init_multi_gpu,
    prepare_model_for_ddp,
    prepare_model_for_fsdp,
    get_model
)
from model import (
    Transformer,
    ModelConfig
)
from generate import generate_and_decode
from model_utils import (
    print_dict,
    get_parameters_count,
    save_checkpoint,
    load_checkpoint,
    load_model_state,
    load_optimizer_state,
    clip_grad_norm,
    log_workload_summary
)
from hellaswag_utils import (
    load_hellaswag_file,
    estimate_correct_candidate_selection
)
from dpo_utils import (
    dpo_log_probs,
    dpo_loss
)
from logger import logger


#### CONFIGURATION

# set training stage
training_stage = config.training_stage

is_pretraining = config.is_pretraining
is_instruct_training = config.is_instruct_training
is_dpo_training = config.is_dpo_training

# datasets path / save checkpoints path
dataloader_root_path = config.dataloader_root_path
save_checkpoints_path = config.save_checkpoints_path
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
seed = config.seed
device_type = config.device_type.value
total_batch_size = config.total_batch_size
max_lr = config.max_lr
min_lr = config.min_lr
warmup_steps = config.warmup_steps
weight_decay = config.weight_decay
max_steps = config.max_steps
adamw_betas = config.adamw_betas
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

# precision, autocast and scaler
use_autocast = config.use_autocast
scaler = config.scaler
model_dtype = config.model_dtype
autocast_dtype = config.autocast_dtype
fsdp_mp = config.fsdp_mp

# validation
validate_every_x_steps = config.validate_every_x_steps
val_steps = config.val_steps
hellaswag_every_x_steps = config.hellaswag_every_x_steps
hellaswag_number_of_examples = config.hellaswag_number_of_examples
generate_every_x_steps = config.generate_every_x_steps
max_test_gen_len = config.max_test_gen_len

# pre asserts model config
assert config.dim % config.n_heads == 0, f'"dim" ({config.dim}) must be divisible by "n_heads" ({config.n_heads})'
assert config.n_kv_heads <= config.n_heads, f'"n_kv_heads" ({config.n_kv_heads}) must be less or equal to "n_heads" ({config.n_heads})'
assert config.n_heads % config.n_kv_heads == 0, f'"n_heads" ({config.n_heads}) must be divisible by n_kv_heads" ({config.n_kv_heads})'

# test prompts
test_prompts_data = json.loads(Path(config.test_prompts_path).read_text())

test_generation_prompts = test_prompts_data[training_stage.value]

# Init the tokenizer
tokenizer = init_tokenizer(config.tokenizer_checkpoint_path, config.huggingface_tokenizer)

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
ddp, ddp_rank, ddp_local_rank, ddp_world_size, is_master_process, device = init_multi_gpu(seed, device_type)

# SETUP LOG HELPER
logger.set_master(is_master_process)

#### INIT WANDB wrapper
wandb = WandbWrapper(enabled=wandb_enabled, is_master_process=is_master_process)
wandb.init(
    wandb_project_name,
    job_name=config.wandb_run_name,
    config={
        'batch_size': config.max_batch_size,
        'sequence_length': config.max_seq_len,
        'min_learning_rate': min_lr,
        'max_learning_rate': max_lr
    }
)

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
val_ce = float('inf')
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

    if loaded_extra_checkpoint_metadata.get('training_stage', None) != training_stage.value:
        logger.info('** WARNING: Training stage has chanded **')
        if not args.start_step:
            logger.info('ignoring stored start step...')
            start_step = 0
        if loaded_train_loader_state is not None and loaded_val_loader_state is not None:
            logger.info('ignoring stored metada for dataset...')
            loaded_train_loader_state = None
            loaded_val_loader_state = None
        if loaded_optimizer_state is not None:
            logger.info('ignoring stored state of optimizer...')
            loaded_optimizer_state = None
        logger.info('\n')

    is_lora_checkpoint = loaded_extra_checkpoint_metadata.get('lora_enabled', False)

#### INIT DATA LOADERS
train_loader, val_loader = init_data_loaders(
    batch_size=config.max_batch_size,
    sequence_length=config.max_seq_len,
    is_master_process=is_master_process,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    data_root=dataloader_root_path,
    pad_id=tokenizer.pad_id,
    training_stage=training_stage
)

if loaded_train_loader_state is not None and loaded_val_loader_state is not None:
    train_loader.load_state_dict(loaded_train_loader_state)
    val_loader.load_state_dict(loaded_val_loader_state)

#### HellaSwag data
HELLASWAG_DATA = load_hellaswag_file(hellaswag_path, ddp, is_master_process, size=hellaswag_number_of_examples)

run_hellaswag_eval = False if hellaswag_every_x_steps == -1 else True

#### INIT MODEL AND TRAINING SETUP
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
    tokenizer=tokenizer,
    vocab_size=tokenizer.vocab_size,
    pad_token_id=tokenizer.pad_id,
    stop_tokens=tokenizer.stop_tokens,
    ignore_index=ignore_index,
    # moe
    is_moe=config.is_moe,
    moe_num_experts=config.moe_num_experts,
    moe_expert_dim=config.moe_expert_dim,
    moe_top_k=config.moe_top_k,
    moe_load_balancing_coef=config.moe_load_balancing_coef,
    moe_z_loss_coef=config.moe_z_loss_coef,
    moe_compute_stats=config.moe_compute_stats
)

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
    load_model_state(model, loaded_model_state)
    logger.info('\nModel loading')
    logger.info('----------------------------------------')
    logger.info('Model checkpoint loaded and ready')

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

torch.backends.cuda.matmul.fp32_precision = 'tf32'
torch.backends.cudnn.conv.fp32_precision = 'tf32'

#### BATCH SIZE ASSERTIONS

# NOTE: total_batch_size is the total batch size in tokens. The model max_batch_size is the number of sequences per device during forward pass (micro batches).
# The total batch size must be a multiple of (max_batch_size * max_seq_len * ddp_world_size). This is needed for the gradient accumulation steps to be calculated correctly.
assert total_batch_size % (model_config.max_batch_size * model_config.max_seq_len * ddp_world_size) == 0, 'total_batch_size must be divisible by (max_batch_size * max_seq_len * ddp_world_size)'

# Gradient accumulation steps
grad_accum_steps = total_batch_size // (model_config.max_batch_size * model_config.max_seq_len * ddp_world_size)

# Final check to validate previous calculations.
assert total_batch_size == (model_config.max_batch_size * model_config.max_seq_len * ddp_world_size * grad_accum_steps)

# COUNT PARAMS
model_params_counts = get_parameters_count(model)

# DPO (Direct Preference Optimization) reference model setup
dpo_ref_model = None
if is_dpo_training:
    logger.info(f'Preparing DPO reference model...', True)
    dpo_ref_model = copy.deepcopy(model).eval()
    for p in dpo_ref_model.parameters():
        p.requires_grad = False
    logger.info(f'Finished preparing DPO reference model', True)

#### COMPILE
if use_torch_compile:
    model.compile()
    if is_dpo_training and dpo_ref_model:
        dpo_ref_model.compile()

#### PREPARE DDP / FSDP
# for FSDP no need to move as that would actually cost more VRAM, instead let FSDP initialization alocate the shard to the device id (ddp_local_rank).
if use_fsdp and dist.is_initialized():
    logger.info('\nFSDP')
    logger.info('----------------------------------------')
    logger.info('Wrapping the model in preparation for FSDP')
    model = prepare_model_for_fsdp(model, ddp_local_rank, fsdp_mp)
    if is_dpo_training and dpo_ref_model:
        dpo_ref_model = prepare_model_for_fsdp(dpo_ref_model, ddp_local_rank, fsdp_mp)
else:
    # move to gpu
    model.to(device=device, dtype=model_dtype)
    if is_dpo_training and dpo_ref_model:
        dpo_ref_model.to(device, dtype=model_dtype)

    if dist.is_initialized():
        logger.info('\nDDP')
        logger.info('----------------------------------------')
        logger.info('Wrapping the model in preparation for DDP')
        model = prepare_model_for_ddp(model, ddp_local_rank)
        if is_dpo_training and dpo_ref_model:
            dpo_ref_model = prepare_model_for_ddp(dpo_ref_model, ddp_local_rank)

#### PREPARE OPTIMIZER OPTIMAL PARAM GROUPS
param_groups = get_model(model).build_optimizer_param_groups(weight_decay=weight_decay)

logger.info(f'\nOptimizer param group configuration:')
logger.info('----------------------------------------')
logger.info(f'num decayed parameter tensors: {len(param_groups.decay_params)}, with {param_groups.num_decay_params:,} parameters')
logger.info(f'num non-decayed parameter tensors: {len(param_groups.nodecay_params)}, with {param_groups.num_nodecay_params:,} parameters')
if param_groups.lora_params:
    logger.info(f'num lora parameter tensors: {len(param_groups.lora_params)}, with {param_groups.num_lora_params:,} parameters')
logger.info(f'trainable parameters: {param_groups.total_trainable_params:,}')

#### INIT OPTIMIZER
fused_available = 'fused' in inspect.signature(AdamW).parameters
use_fused = fused_available and 'cuda' in device

optimizer = AdamW(
    params=param_groups.optimizer_groups,
    lr=max_lr,
    betas=adamw_betas,
    eps=1e-8,
    fused=use_fused
)
if loaded_optimizer_state is not None:
    assert type(loaded_optimizer_state) == dict
    load_optimizer_state(optimizer, model, loaded_optimizer_state)

    # This is to ensure the optimiser state respect the device for all params
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device=p.device, dtype=p.dtype)

    logger.info('optimizer state loaded and ready')

# Model distillation setup
teacher_model = None
if is_model_distillation:
    logger.info(f'Loading teacher model on gpu: {ddp_rank}...', True)
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_checkpoint, token=config.hf_token).to(device, dtype=model_dtype).eval()
    logger.info(f'Finished loading teacher model on gpu: {ddp_rank}...', True)

#### TRAINING LOOP
total_tokens = train_loader.calculate_max_tokens()
complete_max_steps = math.ceil(total_tokens / total_batch_size)

# Workload summary
if is_master_process:
    log_workload_summary(SimpleNamespace(
        checkpoint=checkpoint,
        load_checkpoints_path=load_checkpoints_path,
        optimizers=[('adamw', optimizer)],
        start_step=start_step,
        model_params_counts=model_params_counts,
        ddp_world_size=ddp_world_size,
        grad_accum_steps=grad_accum_steps,
        total_tokens=total_tokens,
        complete_max_steps=complete_max_steps,
        test_generation_prompts=test_generation_prompts,
        model_config=model_config.to_dict(),
        **config.model_dump(),
    ))

if ddp:
    dist.barrier()

logger.info(f'\nGPU: {ddp_local_rank} is ready.', True)

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
        logger.info(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10), True) # TODO can be configured.

torch_profiler_enabled = config.torch_profiler_enabled and is_master_process
if torch_profiler_enabled:
    logger.info('\nWARN: Torch profiler is enabled!\n')
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
    pbar = tqdm(
        range(start_step, max_steps),
        initial=start_step,
        total=max_steps,
        desc=tqdm_label,
        disable=not is_master_process,
        dynamic_ncols=True,
    )
    for step in pbar:
        if abort_if_no_improve.item() == 1:
            logger.info(f'Rank {ddp_rank} received stop signal.', True)
            break

        last_step = (step == max_steps - 1)

        if (step > 0 and step % validate_every_x_steps == 0) or last_step:
            model.eval()
            get_model(model).enable_moe_stats()
            get_model(model).reset_moe_stats()

            val_loss_sum = torch.tensor(0.0, device=device)
            val_tok_sum = torch.tensor(0.0, device=device)

            dpo_metrics = None
            with torch.no_grad():
                for _ in tqdm(range(val_steps), 'Validating', disable=not is_master_process, leave=False):
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
                            loss = model(x, labels=y).loss

                        n_valid = (y != ignore_index).sum().float()
                    val_loss_sum += loss * n_valid
                    val_tok_sum += n_valid

            if ddp:
                dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_tok_sum, op=dist.ReduceOp.SUM)

            val_ce = (val_loss_sum / val_tok_sum).item()

            # MOE METRICS
            moe_metrics = None

            if config.is_moe and config.moe_compute_stats:
                moe_layer_stats = get_model(model).get_moe_stats()

                for layer_id, moe in moe_layer_stats:
                    if ddp:
                        dist.all_reduce(moe.acc_top1_counts, op=dist.ReduceOp.SUM)
                        dist.all_reduce(moe.acc_topk_counts, op=dist.ReduceOp.SUM)
                        dist.all_reduce(moe.acc_p_sum, op=dist.ReduceOp.SUM)
                        dist.all_reduce(moe.acc_tokens, op=dist.ReduceOp.SUM)

                if is_master_process:
                    moe_metrics = {}

                    effs = []
                    max_shares = []
                    dead_counts = []
                    cvs1 = []
                    effs_k = []
                    max_shares_k = []
                    dead_counts_k = []
                    cvsk = []
                    pmean_maxes = []
                    for layer_id, moe in moe_layer_stats:
                        # top-1 utilization
                        c1 = moe.acc_top1_counts.float()
                        p1 = c1 / c1.sum().clamp(min=1.0)
                        cv = (c1.std(unbiased=False) / (c1.mean() + 1e-9)).item() # coef of var

                        # entropy + effective experts (from top-1 usage)
                        entropy = -(p1 * (p1 + 1e-9).log()).sum()
                        eff_experts = torch.exp(entropy).item()
                        max_share = p1.max().item()
                        dead = int((moe.acc_top1_counts == 0).sum().item())

                        effs.append((layer_id, eff_experts))
                        max_shares.append((layer_id, max_share))
                        dead_counts.append((layer_id, dead))
                        cvs1.append((layer_id, cv))

                        # top-k utilization (optional)
                        ck = moe.acc_topk_counts.float()
                        pk = ck / ck.sum().clamp(min=1.0)
                        cvk = (ck.std(unbiased=False) / (ck.mean() + 1e-9)).item()

                        # entropy + effective experts (from top-k usage)
                        entropy_k = -(pk * (pk + 1e-9).log()).sum()
                        eff_experts_k = torch.exp(entropy_k).item()
                        max_share_k = pk.max().item()
                        dead_k = int((moe.acc_topk_counts == 0).sum().item())

                        effs_k.append((layer_id, eff_experts_k))
                        max_shares_k.append((layer_id, max_share_k))
                        dead_counts_k.append((layer_id, dead_k))
                        cvsk.append((layer_id, cvk))

                        # mean router probs (token-weighted)
                        p_mean = (moe.acc_p_sum / moe.acc_tokens.clamp(min=1)).to(torch.float32)
                        pmean_max = float(p_mean.max().item())

                        pmean_maxes.append((layer_id, pmean_max))

                        moe_metrics.update({
                            f'moe/layer_{layer_id}/eff_experts_top1': eff_experts,
                            f'moe/layer_{layer_id}/max_share_top1': max_share,
                            f'moe/layer_{layer_id}/dead_experts_top1': dead,
                            f'moe/layer_{layer_id}/cv_top1': cv,
                            f'moe/layer_{layer_id}/eff_experts_topk': eff_experts_k,
                            f'moe/layer_{layer_id}/max_share_topk': max_share_k,
                            f'moe/layer_{layer_id}/dead_experts_topk': dead_k,
                            f'moe/layer_{layer_id}/cv_topk': cvk,
                            f'moe/layer_{layer_id}/max_p_mean': pmean_max,
                        })

                    # Summary stats
                    eff_vals = [v for _, v in effs]
                    ms_vals = [v for _, v in max_shares]
                    dead_vals = [v for _, v in dead_counts]
                    cv1_vals = [v for _, v in cvs1]

                    worst_eff_layer, worst_eff = min(effs, key=lambda x: x[1])
                    worst_ms_layer, worst_ms = max(max_shares, key=lambda x: x[1])
                    worst_dead_layer, worst_dead = max(dead_counts, key=lambda x: x[1])
                    worst_cv1_layer, worst_cv1 = max(cvs1, key=lambda x: x[1])

                    eff_vals_k = [v for _, v in effs_k]
                    ms_vals_k = [v for _, v in max_shares_k]
                    dead_vals_k = [v for _, v in dead_counts_k]
                    cvk_vals = [v for _, v in cvsk]

                    worst_eff_layer_k, worst_eff_k = min(effs_k, key=lambda x: x[1])
                    worst_ms_layer_k, worst_ms_k = max(max_shares_k, key=lambda x: x[1])
                    worst_dead_layer_k, worst_dead_k = max(dead_counts_k, key=lambda x: x[1])
                    worst_cvk_layer, worst_cvk = max(cvsk, key=lambda x: x[1])

                    pmm_vals = [v for _, v in pmean_maxes]

                    moe_metrics.update({
                        # top 1
                        'moe_summary_top1/eff_experts_mean': float(sum(eff_vals) / len(eff_vals)),
                        'moe_summary_top1/eff_experts_min': float(min(eff_vals)),
                        'moe_summary_top1/max_share_mean': float(sum(ms_vals) / len(ms_vals)),
                        'moe_summary_top1/max_share_max': float(max(ms_vals)),
                        'moe_summary_top1/dead_experts_sum': int(sum(dead_vals)),
                        'moe_summary_top1/cv_mean': float(sum(cv1_vals) / len(cv1_vals)),
                        'moe_summary_top1/cv_max': float(max(cv1_vals)),
                        # top 1 worst
                        'moe_summary_top1_worst/eff_layer': int(worst_eff_layer),
                        'moe_summary_top1_worst/eff_value': float(worst_eff),
                        'moe_summary_top1_worst/max_share_layer': int(worst_ms_layer),
                        'moe_summary_top1_worst/max_share_value': float(worst_ms),
                        'moe_summary_top1_worst/dead_layer': int(worst_dead_layer),
                        'moe_summary_top1_worst/dead_value': int(worst_dead),
                        'moe_summary_top1_worst/cv_layer': int(worst_cv1_layer),
                        'moe_summary_top1_worst/cv_value': float(worst_cv1),
                        # top k
                        'moe_summary_topk/eff_experts_mean': float(sum(eff_vals_k) / len(eff_vals_k)),
                        'moe_summary_topk/eff_experts_min': float(min(eff_vals_k)),
                        'moe_summary_topk/max_share_mean': float(sum(ms_vals_k) / len(ms_vals_k)),
                        'moe_summary_topk/max_share_max': float(max(ms_vals_k)),
                        'moe_summary_topk/dead_experts_sum': int(sum(dead_vals_k)),
                        'moe_summary_topk/cv_mean': float(sum(cvk_vals) / len(cvk_vals)),
                        'moe_summary_topk/cv_max': float(max(cvk_vals)),
                        # top k worst
                        'moe_summary_topk_worst/eff_layer': int(worst_eff_layer_k),
                        'moe_summary_topk_worst/eff_value': float(worst_eff_k),
                        'moe_summary_topk_worst/max_share_layer': int(worst_ms_layer_k),
                        'moe_summary_topk_worst/max_share_value': float(worst_ms_k),
                        'moe_summary_topk_worst/dead_layer': int(worst_dead_layer_k),
                        'moe_summary_topk_worst/dead_value': int(worst_dead_k),
                        'moe_summary_topk_worst/cv_layer': int(worst_cvk_layer),
                        'moe_summary_topk_worst/cv_value': float(worst_cvk),
                        # general
                        'moe_summary/max_p_mean_mean': float(sum(pmm_vals) / len(pmm_vals)),
                        'moe_summary/max_p_mean_max': float(max(pmm_vals)),

                    })

            get_model(model).disable_moe_stats()

            if is_master_process:
                logger.info(f'\nValidation loss: {val_ce:.4f}', pbar=pbar)
                wandb_metrics = {'Validation Loss': val_ce}
                if is_dpo_training:
                    logger.info(dpo_metrics['str'])
                    wandb_metrics.update(dpo_metrics['wandb'])
                if moe_metrics:
                    wandb_metrics.update(moe_metrics)
                wandb.log(wandb_metrics)

            if val_ce < best_val_loss:
                best_val_loss = val_ce
                epochs_no_improve = 0

                if save_checkpoints is True:
                    save_checkpoint(
                        save_checkpoints_path,
                        model,
                        model_config,
                        step,
                        val_ce,
                        optimizer,
                        train_loader,
                        val_loader,
                        extra_checkpoint_metadata,
                        max_number_checkpoints,
                        is_master_process,
                        pbar
                    )
            else:
                if step > early_stopping_patience_skip_steps:
                    epochs_no_improve += 1
                    logger.info(f'Validation loss did not improve. Best: {best_val_loss}, Latest: {val_ce} - Attempts left: {early_stopping_patience - epochs_no_improve}', pbar=pbar)
                else:
                    logger.info(f'Validation loss did not improve. Best: {best_val_loss}, Latest: {val_ce} - (Skip phase...) steps left to skip: {early_stopping_patience_skip_steps - step}', pbar=pbar)

                logger.info('Skipping save checkpoint...', pbar=pbar)

            if epochs_no_improve == early_stopping_patience:
                logger.info(f'The validation loss did not improve for: {early_stopping_patience} - Aborting training...', pbar=pbar)
                abort_if_no_improve[0] = 1

            if ddp:
                broadcast(abort_if_no_improve, src=0)

        if is_pretraining and run_hellaswag_eval and ((step > 0 and step % hellaswag_every_x_steps == 0) or last_step):
            model.eval()
            num_correct_norm = 0
            num_total = 0
            for example in tqdm(HELLASWAG_DATA, 'HellaSwag validation', unit=' examples', disable=not is_master_process, leave=False):
                tokens, mask, label, valid = example['tokens'], example['mask'], example['label'], example['valid']
                tokens = tokens.to(device)
                mask = mask.to(device)

                # get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=use_autocast):
                        logits = model(tokens).logits

                if valid: # Some examples might be dummy in FSDP
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
                logger.info(f'HellaSwag accuracy: {num_correct_norm} / {num_total} = {acc_norm:.4f}', pbar=pbar)
                wandb.log({'HellaSwag accuracy': acc_norm})

        if (step > 0 and step % generate_every_x_steps == 0) or last_step:
            model.eval()
            logger.info('-----------------------------------------------', pbar=pbar)
            for text in generate_and_decode(
                model=get_model(model),
                texts=test_generation_prompts,
                max_gen_len=max_test_gen_len,
                full_seq=True,
                device=device,
                is_instruct=is_instruct_training,
                temperature=0.0,
                top_p=1.0,
                use_kv_cache=True
            ):
                logger.info(text, pbar=pbar)
            logger.info('-----------------------------------------------', pbar=pbar)

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
                    loss = result.loss

                loss_scaled = loss / grad_accum_steps
                if is_model_distillation and teacher_model:
                    train_local_token_sum += x.numel()

                    with torch.no_grad():
                        # NOTE: The vocabularies must match otherwise there will be an error.
                        teacher_logits = teacher_model(input_ids=x).logits

                    loss_distil = distillation_loss(teacher_logits, result.logits, temperature=distillation_temperature)
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
        norm = clip_grad_norm(model, 1.0)

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
            logger.info(f'{step:4d} | train: {train_avg_loss:.4f} | val (last/best): {val_ce:.4f} / {best_val_loss:.4f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}s | tok/s: {int(tokens_per_sec)} | mem MiB: {current_allocated_mb:.0f} / {current_reserved_mb:.0f} (peak) {peak_allocated_mb:.0f}', pbar=pbar)
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
                logger.info(dpo_metrics['str'], pbar=pbar)
                wandb_metrics.update(dpo_metrics['wandb'])
            wandb.log(wandb_metrics)

        if torch_profiler_enabled:
            prof.step()

wandb.finish()

if ddp:
    dist.barrier()
    destroy_process_group()
