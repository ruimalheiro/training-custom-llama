import torch

from tasks.base import BaseTask, TaskStepOutput
from distillation_utils import distillation_loss


class CausalTask(BaseTask):
    name: str = 'causal'

    def setup(self, config, ctx, *, teacher_model=None, **kwargs):
        super().setup(config, ctx, **kwargs)

        self.teacher_model = teacher_model

        return self

    def train_micro_step(self, model, batch):
        device = self.ctx.device
        device_type = self.ctx.device_type
        autocast_dtype = self.ctx.autocast_dtype
        use_autocast = self.ctx.use_autocast
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

        if self.teacher_model is not None:
            tokens_processed += x.numel()

            with torch.no_grad():
                teacher_logits = self.teacher_model(input_ids=x)['logits']

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
    def validation_step(self, model, batch):
        device = self.ctx.device
        device_type = self.ctx.device_type
        autocast_dtype = self.ctx.autocast_dtype
        use_autocast = self.ctx.use_autocast

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
