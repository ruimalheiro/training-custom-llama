from dataclasses import dataclass
from enum import Enum


class StepType(str, Enum):
    TRAIN = 'train'
    VAL = 'val'
    HELLASWAG = 'hellaswag'
    WINOGRANDE = 'winogrande'

@dataclass(frozen=True)
class StepMetrics:
    step_type: StepType
    norm: float = None
    dt: float = None
    tokens_per_sec: int = None
    lrs: dict[str, float] = None
    accuracy: float = None
