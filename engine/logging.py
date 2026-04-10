import math

from config import TrainingStage, TrainConfig
from model import ModelConfig
from checkpoints import CheckpointData
from engine.context import TrainerContext
from engine.optim import OptimizerPlan
from engine.core import TrainerState


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
