import torch

from transformers import AutoModelForCausalLM
from tasks.base import (
    BaseTask,
    TaskStepOutput,
    TaskAssets
)
from distillation_utils import distillation_loss
from logger import logger


class CausalTask(BaseTask):
    name: str = 'causal'

    def setup(self, config, ctx, **kwargs):
        super().setup(config, ctx, **kwargs)
        return self

    def build_assets(self, tokenizer, model):
        config = self.config
        if not config.is_model_distillation:
            return TaskAssets()
        ddp_rank = self.ctx.distributed.ddp_rank
        device = self.ctx.device.device
        model_dtype = self.ctx.precision.model_dtype
        logger.info(f'Loading teacher model on gpu: {ddp_rank}...', True)
        teacher_model = AutoModelForCausalLM.from_pretrained(config.teacher_model_checkpoint, token=config.hf_token)
        if teacher_model.vocab_size != tokenizer.vocab_size:
            logger.warn(f'The sizes of the vocabularies for the teacher model and the tokenizer do not match: {teacher_model.vocab_size} != {tokenizer.vocab_size}\nResizing the vocab of the teacher model to match the tokenizer... NOTE: This can potentially cause issues.')
            teacher_model.resize_token_embeddings(tokenizer.vocab_size)
        teacher_model = teacher_model.to(device, dtype=model_dtype).eval()
        logger.info(f'Finished loading teacher model on gpu: {ddp_rank}...', True)
        return TaskAssets(teacher_model=teacher_model)

    def train_micro_step(self, model, batch, assets: TaskAssets):
        device = self.ctx.device.device
        device_type = self.ctx.device.device_type
        autocast_dtype = self.ctx.precision.autocast_dtype
        use_autocast = self.ctx.precision.use_autocast
        grad_accum_steps = self.ctx.grad_accum_steps

        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        tokens_processed = x.numel()

        with torch.autocast(
            device_type=device_type,
            dtype=autocast_dtype,
            enabled=use_autocast
        ):
            result = model(x, labels=y)
            loss = result['loss']

        loss_for_backward = loss

        metrics = {
            'Train Loss': loss.detach()
        }

        if self.config.is_model_distillation:
            tokens_processed += x.numel()

            with torch.no_grad():
                teacher_logits = assets.teacher_model(input_ids=x)['logits']

            loss_distil = distillation_loss(
                teacher_logits,
                result['logits'],
                temperature=self.config.distillation_temperature
            )
            loss_for_backward = loss_for_backward + loss_distil

            metrics['Train Loss'] = loss_for_backward.detach()
            metrics['Train Distill Loss'] = loss_distil.detach()

        loss_for_backward = loss_for_backward / grad_accum_steps

        n_valid = (y != self.config.ignore_index).sum()

        return TaskStepOutput(
            tokens_processed=tokens_processed,
            n_valid=n_valid,
            loss=loss,
            loss_for_backward=loss_for_backward,
            metrics=metrics
        )

    @torch.no_grad()
    def validation_step(self, model, batch, assets: TaskAssets):
        device = self.ctx.device.device
        device_type = self.ctx.device.device_type
        autocast_dtype = self.ctx.precision.autocast_dtype
        use_autocast = self.ctx.precision.use_autocast

        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=use_autocast):
            loss = model(x, labels=y)['loss']
        
        n_valid = (y != self.config.ignore_index).sum().float()

        return TaskStepOutput(
            n_valid=n_valid,
            loss=loss
        )
