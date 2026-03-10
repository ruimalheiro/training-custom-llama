import torch

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class TaskStepOutput:
    loss: torch.Tensor
    n_valid: torch.Tensor
    tokens_processed: int = 0
    loss_for_backward: Optional[torch.Tensor] = None
    console_logs: list[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

class BaseTask:
    name: str = 'base'

    def setup(self, config, ctx, **kwargs):
        """Useful for any custom setup"""
        self.config = config
        self.ctx = ctx
        return self

    def train_micro_step(self, model, batch) -> TaskStepOutput:
        raise NotImplementedError

    @torch.no_grad()
    def validation_step(self, model, batch) -> TaskStepOutput:
        raise NotImplementedError
