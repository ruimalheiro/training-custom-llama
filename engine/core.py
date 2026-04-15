from dataclasses import dataclass


@dataclass
class TrainerState:
    start_step: int = 0
    current_step: int = 0
    max_steps: int = 0
    best_val_loss: float = float('inf')
    last_val_loss: float = float('inf')
    num_evals_no_improve: int = 0
    should_stop: bool = False
