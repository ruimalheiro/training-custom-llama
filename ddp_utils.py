import os
import torch

from functools import partial
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from model import TransformerBlock


os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')

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

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, is_master_process, device, device_type


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
    raw_model = model.module if ddp else model
    return model, raw_model

def prepare_model_for_fsdp(model, ddp_local_rank, fsdp_precision, sharding='zero3'):
    if sharding == 'zero3':
        strategy = ShardingStrategy.FULL_SHARD
    elif sharding == 'zero2':
        strategy = ShardingStrategy.SHARD_GRAD_OP
    else:
        strategy = ShardingStrategy.NO_SHARD

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock}
    )

    raw_model = model

    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=strategy,
            mixed_precision=fsdp_precision,
            device_id=torch.device(f'cuda:{ddp_local_rank}'),
            sync_module_states=True, # weights broadcasting at initialization..
            use_orig_params=True # The default (false) would flatten the params and breaks our adam configuration as we sepparate decay vs non decay params.
        )
    return model, raw_model
