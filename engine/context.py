import torch

from dataclasses import dataclass, asdict
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

    def to_dict(self):
        return {
            'use_autocast': self.use_autocast,
            'scaler': str(self.scaler),
            'model_dtype': str(self.model_dtype),
            'autocast_dtype': str(self.autocast_dtype),
            'fsdp_mp': str(self.fsdp_mp)
        }

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=4)

@dataclass(frozen=True)
class TrainerContext:
    distributed: DistributedContext
    device: DeviceContext
    precision: PrecisionContext
    grad_accum_steps: int

    def to_dict(self):
        return {
            'distributed': asdict(self.distributed),
            'device': asdict(self.device),
            'precision': self.precision.to_dict(),
            'grad_accum_steps': self.grad_accum_steps
        }

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=4)
