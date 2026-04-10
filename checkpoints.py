import torch
import os
import torch.distributed as dist
import json
import math

from collections import OrderedDict
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions
)
from logger import logger
from dataclasses import dataclass, field
from typing import Any


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

def manage_checkpoints(directory, max_files, pbar=None):
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
    model_config,
    config,
    step,
    last_val_loss,
    best_val_loss,
    optimizers,
    train_loader,
    val_loader,
    extra_metadata,
    max_number_checkpoints,
    is_master_process,
    pbar=None
):
    optimizer_state = {'adamw': None, 'muon': None}
    if dist.is_initialized():
        # ensures we materialize both model / optimizers fully before we attempt to save
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        dist.barrier()
        model_state_dict = get_model_state_dict(model, options=options)

        if optimizers.adamw:
            optimizer_state['adamw'] = get_optimizer_state_dict(model, optimizers.adamw, options=options)
        if optimizers.muon:
            optimizer_state['muon'] = get_optimizer_state_dict(model, optimizers.muon, options=options)

        dist.barrier() # we need all ranks to be sync and participate in the above
    else:
        model_state_dict = model.state_dict()

        if optimizers.adamw:
            optimizer_state['adamw'] = optimizers.adamw.state_dict()
        if optimizers.muon:
            optimizer_state['muon'] = optimizers.muon.state_dict()

    model_state_dict = state_to_cpu(model_state_dict)

    if optimizer_state['adamw']:
        optimizer_state['adamw'] = state_to_cpu(optimizer_state['adamw'])
    if optimizer_state['muon']:
        optimizer_state['muon'] = state_to_cpu(optimizer_state['muon'])

    if is_master_process:
        checkpoint_path = os.path.join(checkpoint_dir, f'model_{step}.pt')
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'model': model_state_dict,
            'step': step,
            'model_config': model_config.to_dict(),
            'config': config.to_summary_dict(include_model_config=False),
            'optimizers': optimizer_state,
            'last_val_loss': float(last_val_loss),
            'best_val_loss': float(best_val_loss),
            'train_dl': train_loader.state_dict(),
            'val_dl': val_loader.state_dict(),
            'metadata': extra_metadata
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f'Saved checkpoint: {checkpoint_path}', pbar)

        manage_checkpoints(checkpoint_dir, max_files=max_number_checkpoints, pbar=pbar)

@dataclass
class CheckpointData:
    path: str | None = None
    checkpoint_name: str | None = None
    model_state: dict[str, Any] | None = None
    optimizers_state: dict[str, Any] | None = None
    start_step: int = 0
    last_val_loss: float = float('inf')
    best_val_loss: float = float('inf')
    train_loader_state: Any = None
    val_loader_state: Any = None
    is_lora_checkpoint: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            'path': self.path,
            'checkpoint_name': self.checkpoint_name,
            'start_step': self.start_step,
            'last_val_loss': self.last_val_loss if not math.isinf(self.last_val_loss) else None,
            'best_val_loss': self.best_val_loss if not math.isinf(self.best_val_loss) else None,
            'is_lora_checkpoint': self.is_lora_checkpoint,
            'metadata': self.metadata
        }

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=4)

def load_checkpoint(
    checkpoint_dir,
    checkpoint,
    reset_optimizers=False,
    force_start_step=None,
    is_master_process=True
):
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    step = state['step'] + 1
    last_val_loss = state['last_val_loss']
    best_val_loss = state['best_val_loss']

    model_state = state['model']
    assert type(model_state) in {OrderedDict, dict}

    optimizers_state = None
    if not reset_optimizers:
        optimizers_state = state['optimizers']
        if optimizers_state['adamw']:
            assert type(optimizers_state['adamw']) == dict
        if optimizers_state['muon']:
            assert type(optimizers_state['muon']) == dict

    if force_start_step is not None:
        step = force_start_step

    train_dl_state = state.get('train_dl', None)
    val_dl_state = state.get('val_dl',   None)

    metadata = state.get('metadata', {})

    if is_master_process:
        logger.info('\nModel checkpoint loading')
        logger.info('----------------------------------------')
        logger.info(f'model state loaded from checkpoint: "{checkpoint}"')
        if optimizers_state is not None:
            logger.info(f'optimizers state loaded from checkpoint')
            if optimizers_state['adamw']:
                logger.info('-- loaded state for adamW')
            if optimizers_state['muon']:
                logger.info('-- loaded state for Muon')

        if train_dl_state is not None and val_dl_state is not None:
            logger.info('Dataloaders state loaded')
            _valid_keys = ['current_shard', 'current_position', 'epoch']

            logger.info('--Train Loader state:')
            logger.info({key: train_dl_state[key] for key in _valid_keys if key in train_dl_state})

            logger.info('--Val Loader state:')
            logger.info({key: val_dl_state[key] for key in _valid_keys if key in val_dl_state})

        try:
            logger.info('\nLoaded config')
            logger.info('----------------------------------------')
            logger.info(state['config'], is_json=True)

            logger.info('\nLoaded model config')
            logger.info('----------------------------------------')
            logger.info(state['model_config'], is_json=True)
        except:
            logger.info('Error printing the config. Potential serialization problem. Should be resolved in the next save attempt.')
        if step > 0:
            logger.info(f'\nResuming from step: {step}')
        else:
            logger.info(f'\nStarting from step: 0')
        logger.info(f'Last calculated loss: {last_val_loss:.4f}')
        logger.info(f'Last calculated best loss: {best_val_loss:.4f}')

        logger.info('\nExtra metadata stored in the checkpoint:')
        logger.info(metadata, is_json=True)
    
    # Delete large state file to free memory
    del state
    if torch.cuda.is_available():
        if is_master_process:
            logger.info('\nClearing cuda cache...\n')
        torch.cuda.empty_cache()

    return CheckpointData(
        path=checkpoint_dir,
        checkpoint_name=checkpoint,
        model_state=model_state,
        optimizers_state=optimizers_state,
        start_step=step,
        last_val_loss=last_val_loss,
        best_val_loss=best_val_loss,
        train_loader_state=train_dl_state,
        val_loader_state=val_dl_state,
        is_lora_checkpoint=metadata.get('lora_enabled', False),
        metadata=metadata
    )

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
