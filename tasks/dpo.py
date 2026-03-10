import torch

from tasks.base import BaseTask, TaskStepOutput
from dpo_utils import (
    dpo_log_probs,
    dpo_loss
)


class DPOTask(BaseTask):
    name: str = 'dpo'

    def setup(self, config, ctx, *, dpo_ref_model, **kwargs):
        super().setup(config, ctx, **kwargs)

        assert dpo_ref_model is not None, 'DPOTask requires a reference model'

        self.dpo_ref_model = dpo_ref_model

        return self

    def train_micro_step(self, model, batch):
        device = self.ctx.device
        device_type = self.ctx.device_type
        autocast_dtype = self.ctx.autocast_dtype
        use_autocast = self.ctx.use_autocast
        grad_accum_steps = self.ctx.grad_accum_steps
        dpo_beta = self.config.dpo_beta

        # x, y, z = prompt, chosen, rejected
        x, y, z = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        z = z.to(device, non_blocking=True)

        tokens_processed = 4 * x.numel() + 2 * y.numel() + 2 * z.numel()

        with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=use_autocast):
            policy_log_probs_pos = dpo_log_probs(model, x, y)
            policy_log_probs_neg = dpo_log_probs(model, x, z)

        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=use_autocast):
                reference_log_probs_pos = dpo_log_probs(self.dpo_ref_model, x, y)
                reference_log_probs_neg = dpo_log_probs(self.dpo_ref_model, x, z)

        loss, dpo_metrics = dpo_loss(
            policy_log_probs_pos,
            policy_log_probs_neg,
            reference_log_probs_pos,
            reference_log_probs_neg,
            dpo_beta
        )

        loss_for_backward = loss / grad_accum_steps

        n_valid = x.size(0) # Assume 1 valid example as the entire triple.

        if not torch.is_tensor(n_valid):
            n_valid = torch.tensor(n_valid, device=device, dtype=loss.dtype)

        return TaskStepOutput(
            tokens_processed=tokens_processed,
            n_valid=n_valid,
            loss=loss,
            loss_for_backward=loss_for_backward,
            console_logs=[dpo_metrics['str']],
            metrics={
                'Train Loss': loss.detach(),
                **dpo_metrics['wandb']
            }
        )

    @torch.no_grad()
    def validation_step(self, model, batch):
        device = self.ctx.device
        device_type = self.ctx.device_type
        autocast_dtype = self.ctx.autocast_dtype
        use_autocast = self.ctx.use_autocast
        dpo_beta = self.config.dpo_beta

        # x, y, z = prompt, chosen, rejected
        x, y, z = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        z = z.to(device, non_blocking=True)

        with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=use_autocast):
            policy_log_probs_pos = dpo_log_probs(model, x, y)
            policy_log_probs_neg = dpo_log_probs(model, x, z)
            reference_log_probs_pos = dpo_log_probs(self.dpo_ref_model, x, y)
            reference_log_probs_neg = dpo_log_probs(self.dpo_ref_model, x, z)

        loss, dpo_metrics = dpo_loss(
            policy_log_probs_pos,
            policy_log_probs_neg,
            reference_log_probs_pos,
            reference_log_probs_neg,
            dpo_beta
        )
        n_valid = torch.tensor(x.size(0), device=device, dtype=loss.dtype) # Assume 1 valid example as the entire triple.

        return TaskStepOutput(
            n_valid=n_valid,
            loss=loss,
            console_logs=[dpo_metrics['str']],
            metrics=dpo_metrics['wandb']
        )
