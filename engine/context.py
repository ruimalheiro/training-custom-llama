import torch

from dataclasses import dataclass
from typing import Any
from torch.distributed.fsdp import MixedPrecisionPolicy


@dataclass(frozen=True)
class DistributedContext:
    ddp: bool
    ddp_rank: int
    ddp_local_rank: int
    ddp_world_size: int
    use_fsdp: bool
    is_master_process: bool

@dataclass(frozen=True)
class DeviceContext:
    device_type: str
    device: torch.device

@dataclass(frozen=True)
class PrecisionContext:
    use_autocast: bool
    scaler: torch.amp.GradScaler | None
    model_dtype: torch.dtype
    autocast_dtype: torch.dtype
    fsdp_mp: MixedPrecisionPolicy | None

@dataclass(frozen=True)
class TrainerContext:
    distributed: DistributedContext
    device: DeviceContext
    precision: PrecisionContext
    grad_accum_steps: int
