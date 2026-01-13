import os
import json
import torch
import torch.distributed as dist
import math

from collections import OrderedDict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    StateDictOptions
)
from torch.optim import AdamW
from lr_schedulers import cosine_scheduler


def print_dict(config):
    print(json.dumps(config, indent=4))

def manage_checkpoints(directory, current_step, max_files):
    # List all checkpoint files
    checkpoints = [os.path.join(directory, file) for file in os.listdir(directory) if file.startswith('model_')]
    # Extract steps from filenames and pair them
    steps_files = [(int(file.split('_')[-1].split('.')[0]), file) for file in checkpoints]
    # Sort list by step numbers
    steps_files.sort()

    # save only the last max_files checkpoints- This is to avoid running out of disk space
    if len(steps_files) > max_files:
        cutoff_index = max(0, len(steps_files) - max_files)
        cutoff_step = steps_files[cutoff_index][0]

        # Delete files
        for step, file in steps_files:
            if step < cutoff_step:
                os.remove(file)
                print(f'Deleted old checkpoint: {file}')

def state_to_cpu(obj):
    # helper to move items from the state to cpu to avoid using more vram
    if torch.is_tensor(obj):
        return obj.cpu()
    if isinstance(obj, dict):
        return {k: state_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = type(obj)
        return t(state_to_cpu(v) for v in obj)
    return obj

def save_checkpoint(
    checkpoint_dir,
    model,
    config,
    step,
    val_loss_accum,
    optimizer,
    train_loader,
    val_loader,
    extra_metadata,
    max_number_checkpoints,
    is_master_process,
    use_fsdp=False
):
    if use_fsdp and dist.is_initialized():
        # ensures we materialize both model / optimizer fully before we attempt to save
        options = StateDictOptions(full_state_dict=True)
        model_state_dict = get_model_state_dict(model, options=options)
        optimizer_state_dict = FSDP.optim_state_dict(model, optimizer)
        dist.barrier(device_ids=[torch.cuda.current_device()]) # we need all ranks to be sync and participate in the above
    else:
        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()

    model_state_dict = state_to_cpu(model_state_dict)
    optimizer_state_dict = state_to_cpu(optimizer_state_dict)

    if is_master_process:
        checkpoint_path = os.path.join(checkpoint_dir, f'model_{step}.pt')
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'model': model_state_dict,
            'step': step,
            'config': config.to_dict(),
            'optimizer': optimizer_state_dict,
            'val_loss': val_loss_accum,
            'train_dl': train_loader.state_dict(),
            'val_dl': val_loader.state_dict(),
            'metadata': extra_metadata
        }

        torch.save(checkpoint, checkpoint_path)
        print(f'Saved checkpoint: {checkpoint_path}')

        manage_checkpoints(checkpoint_dir, current_step=step, max_files=max_number_checkpoints)

def load_checkpoint(
    checkpoint_dir,
    checkpoint,
    reset_optimizer=False,
    force_start_step=None,
    is_master_process=True
):
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
    state = torch.load(checkpoint_path, map_location='cpu')

    step = state['step'] + 1
    loss = state['val_loss']

    model_state = state['model']
    assert type(model_state) in {OrderedDict, dict}

    optimizer_state = None
    if not reset_optimizer:
        optimizer_state = state['optimizer']
        assert type(optimizer_state) == dict

    if force_start_step is not None:
        step = force_start_step

    train_dl_state = state.get('train_dl', None)
    val_dl_state = state.get('val_dl',   None)

    metadata = state.get('metadata', {})

    if is_master_process:
        print('\nModel checkpoint loading')
        print('----------------------------------------')
        print(f'model state loaded from checkpoint: "{checkpoint}"')
        if optimizer_state is not None:
            print(f'optimizer state loaded from checkpoint')

        if train_dl_state is not None and val_dl_state is not None:
            print('Dataloaders state loaded')
            _valid_keys = ['current_shard', 'current_position', 'epoch']

            print('--Train Loader state:')
            print({key: train_dl_state[key] for key in _valid_keys if key in train_dl_state})

            print('--Val Loader state:')
            print({key: val_dl_state[key] for key in _valid_keys if key in val_dl_state})

        try:
            print('\nModel config')
            print('----------------------------------------')
            print_dict(state['config'])
        except:
            print('Error printing the config. Potential serialization problem. Should be resolved in the next save attempt.')
        if step > 0:
            print(f'\nResuming from step: {step}')
        else:
            print(f'\nStarting from step: 0')
        print(f'Last calculated loss: {loss}')

        print('\nExtra metadata stored in the checkpoint:')
        print_dict(metadata)
    
    # Delete large state file to free memory
    del state
    if torch.cuda.is_available():
        if is_master_process:
            print('\nClearing cuda cache...\n')
        torch.cuda.empty_cache()
    
    return model_state, optimizer_state, step, loss, train_dl_state, val_dl_state, metadata

def load_model_state(model, checkpoint_state_dict, use_fsdp=False):
    if use_fsdp and dist.is_initialized():
        options = StateDictOptions(full_state_dict=True)
        set_model_state_dict(model=model, model_state_dict=checkpoint_state_dict, options=options)
    else:
        model.load_state_dict(checkpoint_state_dict)

def load_optimizer_state(optimizer, model, checkpoint_state_dict, use_fsdp):
    if use_fsdp and dist.is_initialized():
        optimizer_state_dict = FSDP.optim_state_dict_to_load(
            optim=optimizer,
            optim_state_dict=checkpoint_state_dict,
            model=model
        )
        optimizer.load_state_dict(optimizer_state_dict)
    else:
        optimizer.load_state_dict(checkpoint_state_dict)

def clip_grad_norm(model, max_norm, is_fsdp=False):
    if is_fsdp:
        return torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_(model, max_norm)
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

def log_workload_summary(c):
    print(f'\n{c.training_stage.upper()} configuration:')
    print('----------------------------------------')
    print('Optimizers:')
    for name, optimizer in c.optimizers:
        print(f'{name}:')
        if isinstance(optimizer, AdamW):
            current_lr, betas = optimizer.param_groups[0]['lr'], optimizer.param_groups[0]['betas']
            scheduled_lr = cosine_scheduler(c.start_step, c.min_lr, c.max_lr, c.warmup_steps, c.max_steps)
            print(f'    using fused AdamW: {optimizer.defaults["fused"]}')
            print(f'    LR set in the optimizer: {current_lr:.4e}')
            print(f'    betas for AdamW: {c.adamw_betas}')
            print(f'    (scheduler) LR that will be applied for step {c.start_step}: {scheduled_lr:.4e}')
        print('--')
    print(f'dataloader data path: "{c.dataloader_root_path}"')
    print(f'HellaSwag data path: "{c.hellaswag_path}"')

    if c.checkpoint is not None:
        print(f'loading checkpoint data path: "{c.load_checkpoints_path}"')

    if c.save_checkpoints:
        print(f'saving checkpoint data path: "{c.save_checkpoints_path}"')

    if c.wandb_enabled:
        print(f'weights and biases project name: "{c.wandb_project_name}"')

    print(f'tokenizer loaded from: "{c.tokenizer_checkpoint_path}"')
    print(f'tokenizer vocab size: {c.model_config["vocab_size"]}')
    print(f'training precision: {c.training_precision.value}')
    print(f'parameter dtype: {c.model_dtype}')
    print(f'using autocast: {c.use_autocast}')
    if c.use_autocast:
        print(f'autocast dtype: {c.autocast_dtype}')
    print(f'total batch size: {c.total_batch_size}')
    print(f'max learning rate: {c.max_lr}')
    print(f'min learning rate: {c.min_lr}')
    print(f'warmup steps: {c.warmup_steps}')
    print(f'weight decay: {c.weight_decay}')
    print(f'max steps: {c.max_steps}')
    print(f'using torch compile: {c.use_torch_compile}')
    print(f'Using FSDP: {c.use_fsdp}')
    if c.use_fsdp:
        print(f'FSDP sharding strategy: {c.fsdp_sharding_strategy.value}')

    if c.is_pretraining or c.is_instruct_training:
        # For pretraining according to the Chinchilla paper ~20.0 is reasonable. For instruct: ~0.2 to ~0.5 is reasonable
        m_factor = 20.0 if c.is_pretraining else 0.3
        tokens_required_for_model_size = int(c.model_params_counts * m_factor)
        steps_needed = math.ceil(tokens_required_for_model_size / c.total_batch_size)
        tokens_per_step = c.max_batch_size * c.max_seq_len * c.ddp_world_size * c.grad_accum_steps
        tokens_coverage = c.max_steps * tokens_per_step
        dataset_fraction = tokens_coverage / c.total_tokens

        print(f'model parameter count: {c.model_params_counts:,}')
        print(f'number of tokens in the dataset: {c.total_tokens:,}')
        print(f'full dataset steps: {c.complete_max_steps}')
        print(f'heuristic token target [model parameter count * {m_factor}]: {tokens_required_for_model_size:,}')
        print(f'dataset covers heuristic? {"YES" if c.total_tokens >= tokens_required_for_model_size else "NO"}')
        print(f'number of steps needed for target: {steps_needed}')
        print(f'tokens per step: {tokens_per_step:,}')
        print(f'tokens processed in this run: {tokens_coverage:,}')
        print(f'fraction of dataset processed: {dataset_fraction*100:.2f}%')
        print(f'configured "max steps" covers heuristic? {"YES" if c.max_steps >= steps_needed else "NO"}')

    if c.is_dpo_training:
        print(f'DPO beta: {c.dpo_beta}')

    print(f'early stopping patience: {c.early_stopping_patience}')

    if c.is_model_distillation:
        print(f'performing model distillation: {c.is_model_distillation}')
        print(f'distillation temperature set to: {c.distillation_temperature}')
        print(f'teacher model checkpoint: {c.teacher_model_checkpoint}')

    print('\nDerived properties')
    print('----------------------------------------')
    print(f'gradient accumulation steps: {c.grad_accum_steps}')

    if c.checkpoint is None:
        print('\nModel config')
        print('----------------------------------------')
        print_dict(c.model_config)

    print('\nEvaluation Config')
    print('----------------------------------------')
    print(f'number of steps between validation: {c.validate_every_x_steps}')
    print(f'number of validating steps: {c.val_steps}')
    print(f'number of steps between HellaSwag validation: {c.hellaswag_every_x_steps}')
    print(f'number of HellaSwag examples: {c.hellaswag_number_of_examples}')
    print(f'number of steps between model output generations: {c.generate_every_x_steps}')
    print(f'max length for the generated text from each prompt: {c.max_test_gen_len}')
    print(f'generation prompts:')
    for example in c.test_generation_prompts:
        print(f'=> "{example}"')
