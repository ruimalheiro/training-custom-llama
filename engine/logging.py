import math

from config import TrainingStage, TrainConfig
from model import ModelConfig
from checkpoints import CheckpointData
from engine.context import TrainerContext
from engine.optim import OptimizerPlan
from engine.core import TrainerState
from metrics import (
    MemoryUsageMetrics,
    StepType,
    StepMetrics,
)
from logger import logger


def prepare_workload_summary(
    *,
    config: TrainConfig,
    model_config: ModelConfig,
    checkpoint_data: CheckpointData,
    trainer_ctx: TrainerContext,
    optimizer_plan: OptimizerPlan,
    trainer_state: TrainerState,
    model_params_count: int,
    model_trainable_params_count: int,
    total_tokens: int
):
    if config.training_stage == TrainingStage.PRETRAIN or config.training_stage == TrainingStage.INSTRUCT:
        # For pretraining according to the Chinchilla paper ~20.0 is reasonable. For instruct: ~0.2 to ~0.5 is reasonable
        m_factor = 20.0 if config.training_stage == TrainingStage.PRETRAIN else 0.3
        tokens_required_for_model_size = int(model_params_count * m_factor)
        steps_needed = math.ceil(tokens_required_for_model_size / config.total_batch_size)
        tokens_per_step = config.total_batch_size
        tokens_coverage = trainer_state.max_steps * tokens_per_step
        dataset_fraction = tokens_coverage / total_tokens if total_tokens > 0 else None

        derived = {
            'model_params_count': model_params_count,
            'model_trainable_params_count': model_trainable_params_count,
            'total_tokens': total_tokens,
            'm_factor': m_factor,
            'tokens_required_for_model_size': tokens_required_for_model_size,
            'dataset_covers_heuristic': (total_tokens >= tokens_required_for_model_size),
            'steps_needed_for_target': steps_needed,
            'tokens_per_step': tokens_per_step,
            'tokens_processed_this_run': tokens_coverage,
            'dataset_fraction_processed': dataset_fraction,
            'max_steps_covers_heuristic': (trainer_state.max_steps >= steps_needed)
        }
    else:
        derived = {
            'model_params_count': model_params_count,
            'model_trainable_params_count': model_trainable_params_count,
            'total_tokens': total_tokens
        }

    summary = {
        'config': config.to_summary_dict(include_model_config=False),
        'model_config': model_config.to_dict(),
        'checkpoint_data': checkpoint_data.to_dict() if checkpoint_data else None,
        'trainer_ctx': trainer_ctx.to_dict(),
        'optimizer_plan': optimizer_plan.to_dict(),
        'derived_properties': derived
    }

    return summary

def prepare_train_step_log(
    *,
    step_metrics: StepMetrics,
    trainer_state: TrainerState,
    aggregated_metrics: dict[str, float],
    memory_usage_metrics: MemoryUsageMetrics,
    console_logs: list[str]
):
    if step_metrics.step_type != StepType.TRAIN:
        raise ValueError(f'Invalid step type for logging: {step_metrics.step_type.value}')

    step = trainer_state.current_step
    last_val_loss = trainer_state.last_val_loss
    best_val_loss = trainer_state.best_val_loss

    train_loss = aggregated_metrics['Train Loss']

    adam_lr = step_metrics.lrs.get('adamw_lr', None)
    muon_lr = step_metrics.lrs.get('muon_lr', None)
    lr_console_message = f'lr (adamw): {adam_lr:.4e}'
    if muon_lr:
        lr_console_message = f'lr (adamw/muon): {adam_lr:.4e} / {muon_lr:.4e}'
    norm = step_metrics.norm
    dt = step_metrics.dt
    tokens_per_sec = step_metrics.tokens_per_sec

    current_allocated_mb = memory_usage_metrics.current_allocated_mb
    current_reserved_mb = memory_usage_metrics.current_reserved_mb
    peak_allocated_mb = memory_usage_metrics.peak_allocated_mb
    peak_reserved_mb = memory_usage_metrics.peak_reserved_mb

    console_log = (
        f'{step:4d} | '
        f'train loss: {train_loss:.4f} | '
        f'val (last/best): {last_val_loss:.4f} / {best_val_loss:.4f} | '
        f'{lr_console_message} | '
        f'norm: {norm:.4f} | '
        f'dt: {dt:.2f}s | '
        f'tok/s: {int(tokens_per_sec)}'
        f'\n       mem MiB current alloc/res: {current_allocated_mb:.0f} / {current_reserved_mb:.0f} | '
        f'peak alloc/res: {peak_allocated_mb:.0f} / {peak_reserved_mb:.0f}'
    )

    wandb_metrics = dict(aggregated_metrics)
    wandb_metrics.update({
        'Learning rate (adamw)': adam_lr,
        'Learning rate (muon)': muon_lr,
        'Norm': norm,
        'Step time (seconds)': dt,
        'Tokens (per sec)': tokens_per_sec,
        'Peak Alloc MiB': peak_allocated_mb,
        'Peak Reserved MiB': peak_reserved_mb,
        'Alloc MiB': current_allocated_mb,
        'Reserved MiB': current_reserved_mb
    })

    console_logs = [console_log, *console_logs]

    return console_logs, wandb_metrics

def prepare_val_step_log(
    *,
    step_metrics: StepMetrics,
    trainer_state: TrainerState,
    aggregated_metrics: dict[str, float],
    moe_metrics: dict[str, int | float],
    console_logs: list[str]
):
    if step_metrics.step_type != StepType.VAL:
        raise ValueError(f'Invalid step type for logging: {step_metrics.step_type.value}')

    step = trainer_state.current_step

    console_log = (
        f'{step:4d} | '
        f'val loss: {trainer_state.last_val_loss:.4f}'
    )

    wandb_metrics = {'Validation Loss': trainer_state.last_val_loss}
    wandb_metrics.update(aggregated_metrics)
    if moe_metrics:
        wandb_metrics.update(moe_metrics)

    console_logs = [console_log, *console_logs]

    return console_logs, wandb_metrics

def prepare_val_step_no_improve_log(
    *,
    early_stopping_patience: int,
    early_stopping_patience_skip_steps: int,
    trainer_state: TrainerState,
    skip_phase=False
):
    step = trainer_state.current_step
    base_msg = f'validation loss did not improve. Best: {trainer_state.best_val_loss}, Latest: {trainer_state.last_val_loss}'
    if skip_phase:
        msg = logger.warning_wrapper(f'{base_msg} - (Skip phase...) steps left to skip: {early_stopping_patience_skip_steps - trainer_state.current_step}')
        console_log = f'{step:4d} | {msg}'
    else:
        msg = logger.warning_wrapper(f'{base_msg} - Attempts left: {early_stopping_patience - trainer_state.num_val_runs_no_improve}')
        console_log = f'{step:4d} | {msg}'

    console_logs = [console_log]

    if trainer_state.should_stop:
        msg = logger.warning_wrapper(f'validation loss did not improve for: {early_stopping_patience} patience steps - Aborting training...')
        console_logs.append(f'{step:4d} | {msg}')

    return console_logs

def prepare_hellaswag_log(
    *,
    step_metrics: StepMetrics,
    trainer_state: TrainerState
):
    if step_metrics.step_type != StepType.HELLASWAG:
        raise ValueError(f'Invalid step type for logging: {step_metrics.step_type.value}')

    step = trainer_state.current_step

    console_log = (
        f'{step:4d} | '
        f'hellaswag accuracy: {step_metrics.accuracy:.4f}'
    )

    wandb_metrics = {'HellaSwag accuracy': step_metrics.accuracy}
    console_logs = [console_log]

    return console_logs, wandb_metrics
