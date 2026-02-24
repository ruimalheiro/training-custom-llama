import torch
import os
import torch.distributed as dist

from collections import OrderedDict
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions
)
from logger import logger


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

def manage_checkpoints(directory, current_step, max_files, pbar=None):
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
                logger.info(f'Deleted old checkpoint: {file}', pbar)

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
    pbar=None
):
    if dist.is_initialized():
        # ensures we materialize both model / optimizer fully before we attempt to save
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        dist.barrier()
        model_state_dict = get_model_state_dict(model, options=options)
        optimizer_state_dict = get_optimizer_state_dict(model, optimizer, options=options)
        dist.barrier() # we need all ranks to be sync and participate in the above
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
        logger.info(f'Saved checkpoint: {checkpoint_path}', pbar)

        manage_checkpoints(checkpoint_dir, current_step=step, max_files=max_number_checkpoints, pbar=pbar)

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
        logger.info('\nModel checkpoint loading')
        logger.info('----------------------------------------')
        logger.info(f'model state loaded from checkpoint: "{checkpoint}"')
        if optimizer_state is not None:
            logger.info(f'optimizer state loaded from checkpoint')

        if train_dl_state is not None and val_dl_state is not None:
            logger.info('Dataloaders state loaded')
            _valid_keys = ['current_shard', 'current_position', 'epoch']

            logger.info('--Train Loader state:')
            logger.info({key: train_dl_state[key] for key in _valid_keys if key in train_dl_state})

            logger.info('--Val Loader state:')
            logger.info({key: val_dl_state[key] for key in _valid_keys if key in val_dl_state})

        try:
            logger.info('\nModel config')
            logger.info('----------------------------------------')
            logger.info(state['config'], is_json=True)
        except:
            logger.info('Error printing the config. Potential serialization problem. Should be resolved in the next save attempt.')
        if step > 0:
            logger.info(f'\nResuming from step: {step}')
        else:
            logger.info(f'\nStarting from step: 0')
        logger.info(f'Last calculated loss: {loss}')

        logger.info('\nExtra metadata stored in the checkpoint:')
        logger.info(metadata, is_json=True)
    
    # Delete large state file to free memory
    del state
    if torch.cuda.is_available():
        if is_master_process:
            logger.info('\nClearing cuda cache...\n')
        torch.cuda.empty_cache()
    
    return model_state, optimizer_state, step, loss, train_dl_state, val_dl_state, metadata

def load_model_state(model, checkpoint_state_dict):
    if dist.is_initialized():
        options = StateDictOptions(full_state_dict=True)
        set_model_state_dict(
            model=model,
            model_state_dict=checkpoint_state_dict,
            options=options
        )
    else:
        model.load_state_dict(checkpoint_state_dict)

def load_optimizer_state(optimizer, model, checkpoint_state_dict):
    if dist.is_initialized():
        options = StateDictOptions(full_state_dict=True)
        set_optimizer_state_dict(
            model=model,
            optimizers=optimizer,
            optim_state_dict=checkpoint_state_dict,
            options=options
        )
    else:
        optimizer.load_state_dict(checkpoint_state_dict)
