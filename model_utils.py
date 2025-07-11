import torch
import os
import json
import time
import wandb

from datetime import datetime
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict


def print_model_config(config):
    print(json.dumps(config, indent=4))

def load_model(
    checkpoint_dir,
    checkpoint,
    reset_optimizer=False,
    force_start_step=None,
    wait_time=5,
    is_master_process=True
):
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
    state = torch.load(checkpoint_path, map_location='cpu')

    step = state['step'] + 1
    loss = state['val_loss']

    model_state = state['model']
    assert type(model_state) == OrderedDict

    optimizer_state = None
    if not reset_optimizer:
        optimizer_state = state['optimizer']
        assert type(optimizer_state) == dict

    if force_start_step is not None:
        step = force_start_step

    train_dl_state = state.get('train_dl', None)
    val_dl_state = state.get('val_dl',   None)
    lora_is_set = state.get('lora_enabled', False)

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
            print_model_config(state['config'])
        except:
            print('Error printing the config. Potential serialization problem. Should be resolved in the next save attempt.')
        if step > 0:
            print(f'\nResuming from step: {step}')
        else:
            print(f'\nStarting from step: 0')
        print(f'Last calculated loss: {loss}')
        if lora_is_set is True:
            print('\nLoRA is present in this checkpoint')
    
    # Delete large state file to free memory
    del state
    if torch.cuda.is_available():
        if is_master_process:
            print('Clearing cache...\n')
        torch.cuda.empty_cache()
        time.sleep(wait_time)
    
    return model_state, optimizer_state, step, loss, train_dl_state, val_dl_state, lora_is_set

def save_model(
    checkpoint_dir,
    model,
    config,
    step,
    val_loss_accum,
    optimizer,
    train_loader,
    val_loader,
    lora_enabled
):
    checkpoint_path = os.path.join(checkpoint_dir, f'model_{step}.pt')
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'model': model.state_dict(),
        'step': step,
        'config': config.to_dict(),
        'optimizer': optimizer.state_dict(),
        'val_loss': val_loss_accum,
        'train_dl': train_loader.state_dict(),
        'val_dl': val_loader.state_dict(),
        'lora_enabled': lora_enabled
    }
    torch.save(checkpoint, checkpoint_path)
    print(f'Saved model: {checkpoint_path}')

    manage_checkpoints(checkpoint_dir, current_step=step, max_files=2)


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


def init_multi_gpu(seed=None):
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available()

        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])

        torch.cuda.set_device(ddp_local_rank)
        init_process_group(backend='nccl')

        device = f'cuda:{ddp_local_rank}'
        is_master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        device = 'cpu'
        is_master_process = True
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'

    device_type = 'cuda' if device.startswith('cuda') else 'cpu'

    if is_master_process:
        print(f'\nDevice setup:')
        print('----------------------------------------')
        print(f'Using device type: {device_type}')
        if ddp_rank:
            print(f'DDP rank: {ddp_rank}')
        if ddp_local_rank:
            print(f'DDP local rank: {ddp_local_rank}')
        if ddp_world_size:
            print(f'DDP world size: {ddp_world_size}')
        print('\n')

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, is_master_process, device, device_type


def prepare_model_for_ddp(model, ddp_local_rank):
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model
    return model, raw_model


class WnbWrapper():
    def __init__(self, enabled=True, is_master_process=True):
        self.WANDB = False
        self.is_master_process = is_master_process

        if enabled and self.is_master_process:
            WANDB_API_KEY = os.getenv('WANDB_API_KEY')
            if WANDB_API_KEY is not None:
                wandb.login(key=WANDB_API_KEY)
                self.WANDB = True
                print('Wandb enabled.')

            
    def init(self, project_name, *, job_name=None, config=None):
        if not self.WANDB:
            return
        
        if not job_name:
            job_start_time = datetime.now().strftime('%Y-%m-%d %H:%M')
            job_name = f'run_{job_start_time}'

        if config is None:
            wandb.init(project=project_name, name=job_name)
        else:
            wandb.init(
                project=project_name,
                name=job_name,
                config=config
            )

    def log(self, data):
        if not self.WANDB:
            return
        wandb.log(data)

    def finish(self):
        if not self.WANDB:
            return
        wandb.finish()
