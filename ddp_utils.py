import os
import torch

from functools import partial
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
from torch.distributed.fsdp import fully_shard
from model import TransformerBlock


os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')

def init_multi_gpu(seed, device_type):
    ddp = int(os.environ.get('RANK', -1)) != -1

    assert torch.cuda.is_available()

    if ddp:
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])

        torch.cuda.set_device(ddp_local_rank)
        init_process_group(backend='nccl', device_id=torch.device(f'cuda:{ddp_local_rank}'))

        device = f'cuda:{ddp_local_rank}'
        is_master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1

        device = 'cuda'
        is_master_process = True

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

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, is_master_process, device


def prepare_model_for_ddp(model, ddp_local_rank):
    ''' More details for the following config:
        https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
    '''
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        model = DDP(
            model,
            device_ids=[ddp_local_rank],
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            bucket_cap_mb=32,
            static_graph=True,
            find_unused_parameters=False
        )
    return model

def prepare_model_for_fsdp(model, ddp_local_rank, fsdp_precision):
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        for block in model.layers:
            fully_shard(block, reshard_after_forward=True, mp_policy=fsdp_precision)
        fully_shard(model, reshard_after_forward=False, mp_policy=fsdp_precision)
    return model

def get_model(model):
    if hasattr(model, 'module'):
        return model.module
    return model
