import torch

from dataclasses import dataclass
from typing import Any

@dataclass
class TrainerContext:
    device: torch.device
    device_type: str
    use_autocast: bool
    autocast_dtype: torch.dtype
    grad_accum_steps: int
    scaler: Any
    ddp: bool
    use_fsdp: bool
    is_master_process: bool
