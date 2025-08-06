import torch
import os
import json

from collections import OrderedDict


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

def save_model(
    checkpoint_dir,
    model,
    config,
    step,
    val_loss_accum,
    optimizer,
    train_loader,
    val_loader,
    extra_metadata
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
        'metadata': extra_metadata
    }
    torch.save(checkpoint, checkpoint_path)
    print(f'Saved checkpoint: {checkpoint_path}')

    manage_checkpoints(checkpoint_dir, current_step=step, max_files=2)

def load_model(
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
    assert type(model_state) == OrderedDict

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
