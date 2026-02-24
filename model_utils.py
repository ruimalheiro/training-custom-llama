import torch
import math

from torch.distributed.tensor import DTensor
from torch.optim import AdamW
from lr_schedulers import cosine_scheduler
from logger import logger


def get_parameters_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def clip_grad_norm(model, max_norm):
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    if isinstance(norm, DTensor):
        return norm.to_local()
    return norm

def log_workload_summary(c):
    logger.info(f'\n{c.training_stage.upper()} configuration:')
    logger.info('----------------------------------------')
    logger.info('Optimizers:')
    for name, optimizer in c.optimizers:
        logger.info(f'{name}:')
        if isinstance(optimizer, AdamW):
            current_lr, betas = optimizer.param_groups[0]['lr'], optimizer.param_groups[0]['betas']
            scheduled_lr = cosine_scheduler(c.start_step, c.min_lr, c.max_lr, c.warmup_steps, c.max_steps)
            logger.info(f'    using fused AdamW: {optimizer.defaults["fused"]}')
            logger.info(f'    LR set in the optimizer: {current_lr:.4e}')
            logger.info(f'    betas for AdamW: {c.adamw_betas}')
            logger.info(f'    (scheduler) LR that will be applied for step {c.start_step}: {scheduled_lr:.4e}')
        logger.info('--')
    logger.info(f'dataloader data path: "{c.dataloader_root_path}"')
    logger.info(f'HellaSwag data path: "{c.hellaswag_path}"')

    if c.checkpoint is not None:
        logger.info(f'loading checkpoint data path: "{c.load_checkpoints_path}"')

    if c.save_checkpoints:
        logger.info(f'saving checkpoint data path: "{c.save_checkpoints_path}"')

    if c.wandb_enabled:
        logger.info(f'weights and biases project name: "{c.wandb_project_name}"')

    logger.info(f'tokenizer loaded from: "{c.tokenizer_checkpoint_path}"')
    logger.info(f'tokenizer vocab size: {c.model_config["vocab_size"]}')
    logger.info(f'training precision: {c.training_precision.value}')
    logger.info(f'parameter dtype: {c.model_dtype}')
    logger.info(f'using autocast: {c.use_autocast}')
    if c.use_autocast:
        logger.info(f'autocast dtype: {c.autocast_dtype}')
    logger.info(f'total batch size: {c.total_batch_size}')
    logger.info(f'max learning rate: {c.max_lr}')
    logger.info(f'min learning rate: {c.min_lr}')
    logger.info(f'warmup steps: {c.warmup_steps}')
    logger.info(f'weight decay: {c.weight_decay}')
    logger.info(f'max steps: {c.max_steps}')
    logger.info(f'using torch compile: {c.use_torch_compile}')
    logger.info(f'Using FSDP: {c.use_fsdp}')

    if c.is_pretraining or c.is_instruct_training:
        # For pretraining according to the Chinchilla paper ~20.0 is reasonable. For instruct: ~0.2 to ~0.5 is reasonable
        m_factor = 20.0 if c.is_pretraining else 0.3
        tokens_required_for_model_size = int(c.model_params_counts * m_factor)
        steps_needed = math.ceil(tokens_required_for_model_size / c.total_batch_size)
        tokens_per_step = c.max_batch_size * c.max_seq_len * c.ddp_world_size * c.grad_accum_steps
        tokens_coverage = c.max_steps * tokens_per_step
        dataset_fraction = tokens_coverage / c.total_tokens

        logger.info(f'model parameter count: {c.model_params_counts:,}')
        logger.info(f'number of tokens in the dataset: {c.total_tokens:,}')
        logger.info(f'full dataset steps: {c.complete_max_steps}')
        logger.info(f'heuristic token target [model parameter count * {m_factor}]: {tokens_required_for_model_size:,}')
        logger.info(f'dataset covers heuristic? {"YES" if c.total_tokens >= tokens_required_for_model_size else "NO"}')
        logger.info(f'number of steps needed for target: {steps_needed}')
        logger.info(f'tokens per step: {tokens_per_step:,}')
        logger.info(f'tokens processed in this run: {tokens_coverage:,}')
        logger.info(f'fraction of dataset processed: {dataset_fraction*100:.2f}%')
        logger.info(f'configured "max steps" covers heuristic? {"YES" if c.max_steps >= steps_needed else "NO"}')

    if c.is_dpo_training:
        logger.info(f'DPO beta: {c.dpo_beta}')

    logger.info(f'early stopping patience: {c.early_stopping_patience}')

    if c.is_model_distillation:
        logger.info(f'performing model distillation: {c.is_model_distillation}')
        logger.info(f'distillation temperature set to: {c.distillation_temperature}')
        logger.info(f'teacher model checkpoint: {c.teacher_model_checkpoint}')

    logger.info('\nDerived properties')
    logger.info('----------------------------------------')
    logger.info(f'gradient accumulation steps: {c.grad_accum_steps}')

    if c.checkpoint is None:
        logger.info('\nModel config')
        logger.info('----------------------------------------')
        logger.info(c.model_config, is_json=True)

    logger.info('\nEvaluation Config')
    logger.info('----------------------------------------')
    logger.info(f'number of steps between validation: {c.validate_every_x_steps}')
    logger.info(f'number of validating steps: {c.val_steps}')
    logger.info(f'number of steps between HellaSwag validation: {c.hellaswag_every_x_steps}')
    logger.info(f'number of HellaSwag examples: {c.hellaswag_number_of_examples}')
    logger.info(f'number of steps between model output generations: {c.generate_every_x_steps}')
    logger.info(f'max length for the generated text from each prompt: {c.max_test_gen_len}')
    logger.info(f'generation prompts:')
    for example in c.test_generation_prompts:
        logger.info(f'=> "{example}"')
