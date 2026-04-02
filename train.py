import os
import torch
import torch.distributed as dist
import argparse
import math
import copy
import json
import inspect

from config import (
    config,
    TrainingStage
)
os.environ['HF_HOME'] = config.hf_home
os.environ['HF_DATASETS_CACHE'] = f'{config.hf_home}/datasets'
os.environ['HF_HUB_CACHE'] = f'{config.hf_home}/hub'

from pathlib import Path
from tokenizer import init_tokenizer
from dataloaders import init_data_loaders
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from lora import apply_lora
from lr_schedulers import cosine_scheduler
from wandb_utils import WandbWrapper
from contextlib import nullcontext
from types import SimpleNamespace

from torch.optim import AdamW
from torch.distributed import (
    broadcast,
    destroy_process_group
)
from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler
)
from ddp_utils import (
    init_multi_gpu,
    prepare_model_for_ddp,
    prepare_model_for_fsdp,
    get_model
)
from model import (
    Transformer,
    ModelConfig
)
from generate import generate_and_decode
from checkpoints import (
    save_checkpoint,
    load_checkpoint,
    load_model_state,
    load_optimizer_state
)
from model_utils import (
    get_parameters_count,
    clip_grad_norm,
    log_workload_summary
)
from hellaswag_utils import (
    load_hellaswag_file,
    estimate_correct_candidate_selection
)
from logger import logger
from engine import Trainer
from metrics import (
    collect_moe_metrics,
    accumulate_weighted_metrics,
    combine_weighted_metrics
)
from tasks import get_task


if __name__ == "__main__":
    #### SCRIPT OPTIONS
    parser = argparse.ArgumentParser(description='Script options')
    parser.add_argument('--pretrain_checkpoint', type=str, default=None, help='Pretrain checkpoint to load.')
    parser.add_argument('--instruct_checkpoint', type=str, default=None, help='Instruct checkpoint to load.')
    parser.add_argument('--dpo_checkpoint', type=str, default=None, help='DPO checkpoint to load.')
    parser.add_argument('--reset-optimizer', action='store_true', help='Reset the optimizer state when loading a checkpoint.')
    parser.add_argument('--start-step', type=int, default=None, help='Starting step number for training.')

    args = parser.parse_args()

    trainer = Trainer(
        config = config,
        args=args
    )
    trainer.setup()
    trainer.train()

    # ### CONFIGURATION
    # # set training stage
    # training_stage = config.training_stage
    # is_pretraining = config.is_pretraining
    # is_instruct_training = config.is_instruct_training
    # is_dpo_training = config.is_dpo_training
    # # datasets path / save checkpoints path
    # dataloader_root_path = config.dataloader_root_path
    # save_checkpoints_path = config.save_checkpoints_path
    # hellaswag_path = config.hellaswag_path
    # # load path
    # pretrain_checkpoints_path = config.pretrain_load_checkpoints_path
    # instruct_checkpoints_path = config.instruct_load_checkpoints_path
    # dpo_checkpoints_path = config.dpo_load_checkpoints_path
    # # save toggle
    # save_checkpoints = config.save_checkpoints
    # max_number_checkpoints = config.max_number_checkpoints
    # # wandb
    # wandb_enabled = config.wandb_enabled
    # wandb_project_name = config.wandb_project_name
    # # tokenizer model path
    # tokenizer_checkpoint_path = config.tokenizer_checkpoint_path
    # # value to mask the padded tokens
    # ignore_index = config.ignore_index
    # # train config
    # seed = config.seed
    # total_batch_size = config.total_batch_size
    # max_lr = config.max_lr
    # min_lr = config.min_lr
    # warmup_steps = config.warmup_steps
    # weight_decay = config.weight_decay
    # max_steps = config.max_steps
    # adamw_betas = config.adamw_betas
    # early_stopping_patience = config.early_stopping_patience
    # early_stopping_patience_skip_steps = config.early_stopping_patience_skip_steps
    # dpo_beta = config.dpo_beta
    # is_model_distillation = config.is_model_distillation
    # distillation_temperature = config.distillation_temperature
    # # The teacher model is loaded via huggingface API: AutoModelForCausalLM.from_pretrained(teacher_model_checkpoint, ...) so needs to be a valid checkpoint.
    # teacher_model_checkpoint = config.teacher_model_checkpoint
    # lora_enabled = config.lora_enabled
    # lora_rank = config.lora_rank
    # lora_alpha = config.lora_alpha
    # lora_dropout = config.lora_dropout
    # lora_target_modules = config.lora_target_modules
    # use_torch_compile = config.use_torch_compile
    # use_fsdp = config.use_fsdp
    # # precision, autocast and scaler
    # model_dtype = config.model_dtype
    # fsdp_mp = config.fsdp_mp
    # # validation
    # validate_every_x_steps = config.validate_every_x_steps
    # val_steps = config.val_steps
    # hellaswag_every_x_steps = config.hellaswag_every_x_steps
    # hellaswag_number_of_examples = config.hellaswag_number_of_examples
    # generate_every_x_steps = config.generate_every_x_steps
    # max_test_gen_len = config.max_test_gen_len
    # # test prompts
    # # Init the tokenizer
    # # Extra metadata to store when saving a checkpoint
    # #### INIT DISTRIBUTED DATA PARALLEL (DDP)
    # # SETUP LOG HELPER
    # #### INIT WANDB wrapper
    # #### LOAD CHECKPOINT
    # # defaults
    # #### INIT DATA LOADERS
    # #### HellaSwag data
    # #### INIT MODEL AND TRAINING SETUP
    # #### BATCH SIZE ASSERTIONS
    # # COUNT PARAMS
    # # Model distillation setup

    

    # # DPO (Direct Preference Optimization) reference model setup
    # dpo_ref_model = None
    # if is_dpo_training:
    #     logger.info(f'Preparing DPO reference model...', True)
    #     dpo_ref_model = copy.deepcopy(model).eval()
    #     for p in dpo_ref_model.parameters():
    #         p.requires_grad = False
    #     logger.info(f'Finished preparing DPO reference model', True)

    # #### COMPILE
    # if use_torch_compile:
    #     model.compile()
    #     if is_dpo_training and dpo_ref_model is not None:
    #         dpo_ref_model.compile()

    # #### PREPARE DDP / FSDP
    # # for FSDP no need to move as that would actually cost more VRAM, instead let FSDP initialization alocate the shard to the device id (ddp_local_rank).
    # if use_fsdp and dist.is_initialized():
    #     logger.info('\nFSDP')
    #     logger.info('----------------------------------------')
    #     logger.info('Wrapping the model in preparation for FSDP')
    #     model = prepare_model_for_fsdp(model, ddp_local_rank, fsdp_mp)
    #     if is_dpo_training and dpo_ref_model is not None:
    #         dpo_ref_model = prepare_model_for_fsdp(dpo_ref_model, ddp_local_rank, fsdp_mp)
    # else:
    #     # move to gpu
    #     model.to(device=device, dtype=model_dtype)
    #     if is_dpo_training and dpo_ref_model is not None:
    #         dpo_ref_model.to(device, dtype=model_dtype)

    #     if dist.is_initialized():
    #         logger.info('\nDDP')
    #         logger.info('----------------------------------------')
    #         logger.info('Wrapping the model in preparation for DDP')
    #         model = prepare_model_for_ddp(model, ddp_local_rank)
    #         if is_dpo_training and dpo_ref_model is not None:
    #             dpo_ref_model = prepare_model_for_ddp(dpo_ref_model, ddp_local_rank)

    # #### PREPARE OPTIMIZER OPTIMAL PARAM GROUPS
    # param_groups = get_model(model).build_optimizer_param_groups(weight_decay=weight_decay)

    # logger.info(f'\nOptimizer param group configuration:')
    # logger.info('----------------------------------------')
    # logger.info(f'num decayed parameter tensors: {len(param_groups.decay_params)}, with {param_groups.num_decay_params:,} parameters')
    # logger.info(f'num non-decayed parameter tensors: {len(param_groups.nodecay_params)}, with {param_groups.num_nodecay_params:,} parameters')
    # if param_groups.lora_params:
    #     logger.info(f'num lora parameter tensors: {len(param_groups.lora_params)}, with {param_groups.num_lora_params:,} parameters')
    # logger.info(f'trainable parameters: {param_groups.total_trainable_params:,}')

    # #### INIT OPTIMIZER
    # optimizer = AdamW(
    #     params=param_groups.optimizer_groups,
    #     lr=max_lr,
    #     betas=adamw_betas,
    #     eps=1e-8,
    #     fused=use_fused
    # )
    # if loaded_optimizer_state is not None:
    #     assert isinstance(loaded_optimizer_state, dict)
    #     load_optimizer_state(optimizer, model, loaded_optimizer_state)

    #     # This is to ensure the optimiser state respect the device for all params
    #     for group in optimizer.param_groups:
    #         for p in group['params']:
    #             state = optimizer.state[p]
    #             for k, v in state.items():
    #                 if torch.is_tensor(v):
    #                     state[k] = v.to(device=p.device, dtype=p.dtype)

    #     logger.info('optimizer state loaded and ready')


    # #### TRAINING LOOP
    # total_tokens = train_loader.calculate_max_tokens()
    # complete_max_steps = math.ceil(total_tokens / total_batch_size)

    # # Workload summary
    # if is_master_process:
    #     log_workload_summary(SimpleNamespace(
    #         checkpoint=checkpoint,
    #         load_checkpoints_path=load_checkpoints_path,
    #         optimizers=[('adamw', optimizer)],
    #         start_step=start_step,
    #         model_params_counts=model_params_counts,
    #         ddp_world_size=ddp_world_size,
    #         grad_accum_steps=grad_accum_steps,
    #         total_tokens=total_tokens,
    #         complete_max_steps=complete_max_steps,
    #         test_generation_prompts=test_generation_prompts,
    #         model_config=model_config.to_dict(),
    #         **config.model_dump(),
    #     ))

    # if ddp:
    #     dist.barrier()

    # logger.info(f'\nDevice: {ddp_local_rank} is ready.', True)

    # tqdm_label = f'Training ({training_stage.value})'

    # # max_steps not set
    # if max_steps == -1:
    #     max_steps = complete_max_steps

    # epochs_no_improve = 0
    # abort_if_no_improve = torch.tensor([0], device=device)
    # early_stopping_patience_skip_steps += start_step

    # task = get_task(config.training_stage)
    # task.setup(
    #     config=config,
    #     ctx=TrainerContext(
    #         device=device,
    #         device_type=device_type,
    #         use_autocast=use_autocast,
    #         autocast_dtype=autocast_dtype,
    #         grad_accum_steps=grad_accum_steps,
    #         scaler=scaler
    #     ),
    #     dpo_ref_model=dpo_ref_model,
    #     teacher_model=teacher_model
    # )

    # def trace_handler(prof):
    #     if config.torch_profiler_tensorboard_enabled:
    #         tensor_board = tensorboard_trace_handler(
    #             dir_name=config.torch_profiler_tensorboard_log_path,
    #             worker_name='rank0' if is_master_process else None,
    #         )
    #         tensor_board(prof)
    #     else:
    #         logger.info(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10), True) # TODO can be configured.

    # torch_profiler_enabled = config.torch_profiler_enabled and is_master_process
    # if torch_profiler_enabled:
    #     logger.info('\nWARN: Torch profiler is enabled!\n')
    # torch_profiler_context = (
    #     profile(
    #         activities=[
    #             ProfilerActivity.CPU,
    #             ProfilerActivity.CUDA
    #         ],
    #         record_shapes=True,
    #         profile_memory=True,
    #         schedule=schedule(
    #             skip_first=config.torch_profiler_schedule_skip_first,
    #             wait=config.torch_profiler_schedule_wait,
    #             warmup=config.torch_profiler_schedule_warmup,
    #             active=config.torch_profiler_schedule_active,
    #             repeat=config.torch_profiler_schedule_repeat
    #         ),
    #         on_trace_ready=trace_handler
    #     ) if torch_profiler_enabled else nullcontext()
    # )

    # with torch_profiler_context as prof:
    #     pbar = tqdm(
    #         range(start_step, max_steps),
    #         initial=start_step,
    #         total=max_steps,
    #         desc=tqdm_label,
    #         disable=not is_master_process,
    #         dynamic_ncols=True,
    #     )
    #     for step in pbar:
    #         if abort_if_no_improve.item() == 1:
    #             logger.info(f'Rank {ddp_rank} received stop signal.', True)
    #             break

    #         last_step = (step == max_steps - 1)

    #         if should_run(step, validate_every_x_steps, last_step):
    #             model.eval()
    #             get_model(model).enable_moe_stats()
    #             get_model(model).reset_moe_stats()

    #             val_loss_sum = torch.tensor(0.0, device=device)
    #             val_tok_sum = torch.tensor(0.0, device=device)
    #             console_logs = []
    #             val_metric_sums = {}
    #             val_metric_weights = {}
    #             with torch.no_grad():
    #                 for _ in tqdm(range(val_steps), 'Validating', disable=not is_master_process, leave=False):
    #                     val_output = task.validation_step(model, val_loader.next_batch())
    #                     loss = val_output.loss
    #                     n_valid = val_output.n_valid

    #                     console_logs.extend(val_output.console_logs)
    #                     accumulate_weighted_metrics(
    #                         weight=n_valid,
    #                         metrics=val_output.metrics,
    #                         metrics_sum_acc=val_metric_sums,
    #                         metrics_weights_acc=val_metric_weights,
    #                         device=device
    #                     )

    #                     val_loss_sum += loss * n_valid
    #                     val_tok_sum += n_valid

    #             if ddp:
    #                 dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
    #                 dist.all_reduce(val_tok_sum, op=dist.ReduceOp.SUM)

    #             val_ce = (val_loss_sum / val_tok_sum).item()

    #             aggregated_val_metrics = combine_weighted_metrics(
    #                 metrics_sum_acc=val_metric_sums,
    #                 metrics_weights_acc=val_metric_weights,
    #                 ddp=ddp
    #             )

    #             # MOE METRICS
    #             moe_metrics = None
    #             if config.is_moe and config.moe_compute_stats:
    #                 moe_metrics = collect_moe_metrics(get_model(model), ddp, is_master_process)

    #             get_model(model).disable_moe_stats()

    #             if is_master_process:
    #                 logger.info(f'\nValidation loss: {val_ce:.4f}', pbar=pbar)
    #                 wandb_metrics = {'Validation Loss': val_ce}

    #                 for log in console_logs:
    #                     logger.info(log, pbar=pbar)
    #                 wandb_metrics.update(aggregated_val_metrics)

    #                 if moe_metrics:
    #                     wandb_metrics.update(moe_metrics)
    #                 wandb.log(wandb_metrics)

    #             if val_ce < best_val_loss:
    #                 best_val_loss = val_ce
    #                 epochs_no_improve = 0

    #                 if save_checkpoints is True:
    #                     save_checkpoint(
    #                         save_checkpoints_path,
    #                         model,
    #                         model_config,
    #                         step,
    #                         val_ce,
    #                         optimizer,
    #                         train_loader,
    #                         val_loader,
    #                         extra_checkpoint_metadata,
    #                         max_number_checkpoints,
    #                         is_master_process,
    #                         pbar
    #                     )
    #             else:
    #                 if step > early_stopping_patience_skip_steps:
    #                     epochs_no_improve += 1
    #                     logger.info(f'Validation loss did not improve. Best: {best_val_loss}, Latest: {val_ce} - Attempts left: {early_stopping_patience - epochs_no_improve}', pbar=pbar)
    #                 else:
    #                     logger.info(f'Validation loss did not improve. Best: {best_val_loss}, Latest: {val_ce} - (Skip phase...) steps left to skip: {early_stopping_patience_skip_steps - step}', pbar=pbar)

    #                 logger.info('Skipping save checkpoint...', pbar=pbar)

    #             if epochs_no_improve == early_stopping_patience:
    #                 logger.info(f'The validation loss did not improve for: {early_stopping_patience} - Aborting training...', pbar=pbar)
    #                 abort_if_no_improve[0] = 1

    #             if ddp:
    #                 broadcast(abort_if_no_improve, src=0)

    #         if is_pretraining and should_run(step, hellaswag_every_x_steps, last_step):
    #             model.eval()
    #             num_correct_norm = 0
    #             num_total = 0
    #             for example in tqdm(HELLASWAG_DATA, 'HellaSwag validation', unit=' examples', disable=not is_master_process, leave=False):
    #                 tokens, mask, label, valid = example['tokens'], example['mask'], example['label'], example['valid']
    #                 tokens = tokens.to(device)
    #                 mask = mask.to(device)

    #                 # get the logits
    #                 with torch.no_grad():
    #                     with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=use_autocast):
    #                         logits = model(tokens)['logits']

    #                 if valid: # Some examples might be dummy in FSDP
    #                     predicted_correct = estimate_correct_candidate_selection(tokens, mask, logits)
    #                     num_total += 1
    #                     num_correct_norm += int(predicted_correct == label)

    #             # Aggregates counts across all GPU processes (in DDP) to compute global accuracy.
    #             if ddp:
    #                 num_total = torch.tensor(num_total, dtype=torch.long, device=device)
    #                 num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
    #                 dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
    #                 dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
    #                 num_total = num_total.item()
    #                 num_correct_norm = num_correct_norm.item()
    #             acc_norm = num_correct_norm / num_total

    #             if is_master_process:
    #                 logger.info(f'HellaSwag accuracy: {num_correct_norm} / {num_total} = {acc_norm:.4f}', pbar=pbar)
    #                 wandb.log({'HellaSwag accuracy': acc_norm})

    #         if should_run(step, generate_every_x_steps, last_step):
    #             model.eval()
    #             logger.info('-----------------------------------------------', pbar=pbar)
    #             for text in generate_and_decode(
    #                 model=get_model(model),
    #                 texts=test_generation_prompts,
    #                 max_gen_len=max_test_gen_len,
    #                 full_seq=True,
    #                 device=device,
    #                 is_instruct=is_instruct_training,
    #                 temperature=0.0,
    #                 top_p=1.0,
    #                 use_kv_cache=True
    #             ):
    #                 logger.info(text, pbar=pbar)
    #             logger.info('-----------------------------------------------', pbar=pbar)

    #         torch.cuda.reset_peak_memory_stats()

    #         model.train()
    #         optimizer.zero_grad(set_to_none=True)
    #         train_local_token_sum = torch.tensor(0.0, device=device)
    #         console_logs = []
    #         train_metric_sums = {}
    #         train_metric_weights = {}
    #         t0 = torch.cuda.Event(enable_timing=True)
    #         t1 = torch.cuda.Event(enable_timing=True)
    #         t0.record()
    #         for micro_step in range(grad_accum_steps):
    #             train_output = task.train_micro_step(model, train_loader.next_batch())
    #             train_local_token_sum += train_output.tokens_processed
    #             loss_scaled = train_output.loss_for_backward

    #             console_logs.extend(train_output.console_logs)
    #             accumulate_weighted_metrics(
    #                 weight=train_output.n_valid,
    #                 metrics=train_output.metrics,
    #                 metrics_sum_acc=train_metric_sums,
    #                 metrics_weights_acc=train_metric_weights,
    #                 device=device
    #             )

    #             if ddp and not use_fsdp: # require_backward_grad_sync is not used with FSDP
    #                 model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

    #             if scaler:
    #                 scaler.scale(loss_scaled).backward()
    #             else:
    #                 loss_scaled.backward()

    #         train_token_sum = train_local_token_sum
    #         if ddp:
    #             dist.all_reduce(train_token_sum, op=dist.ReduceOp.SUM)
    #         train_token_sum = train_token_sum.item()

    #         aggregated_metrics = combine_weighted_metrics(
    #             metrics_sum_acc=train_metric_sums,
    #             metrics_weights_acc=train_metric_weights,
    #             ddp=ddp
    #         )

    #         if scaler:
    #             scaler.unscale_(optimizer) # due to fp16, optimizer gradients are inflated so need to unscale before clipping.
    #         norm = clip_grad_norm(model, 1.0)

    #         lr = cosine_scheduler(step, min_lr, max_lr, warmup_steps, max_steps)
    #         for param_group in optimizer.param_groups:
    #             param_group['lr'] = lr

    #         if scaler:
    #             # The dynamic range in fp16 is low and this handles NaNs/infs which might occur more.
    #             scaler.step(optimizer)
    #             scaler.update()
    #         else:
    #             optimizer.step()

    #         t1.record()
    #         t1.synchronize()
    #         dt = t0.elapsed_time(t1) / 1000.0
    #         tokens_per_sec = train_token_sum / dt

    #         peak_allocated_mb = torch.cuda.max_memory_allocated(ddp_local_rank) / 1024**2
    #         peak_reserved_mb = torch.cuda.max_memory_reserved(ddp_local_rank) / 1024**2
    #         current_allocated_mb = torch.cuda.memory_allocated(ddp_local_rank) / 1024**2
    #         current_reserved_mb = torch.cuda.memory_reserved(ddp_local_rank) / 1024**2

    #         if is_master_process:
    #             train_loss = aggregated_metrics['Train Loss']
    #             logger.info(f'{step:4d} | train: {train_loss:.4f} | val (last/best): {val_ce:.4f} / {best_val_loss:.4f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}s | tok/s: {int(tokens_per_sec)} | mem MiB: {current_allocated_mb:.0f} / {current_reserved_mb:.0f} (peak) {peak_allocated_mb:.0f}', pbar=pbar)

    #             wandb_metrics = dict(aggregated_metrics)
    #             wandb_metrics.update({
    #                 'Learning rate': lr,
    #                 'Norm': norm,
    #                 'Step time (seconds)': dt,
    #                 'Tokens (per sec)': tokens_per_sec,
    #                 'Peak Alloc MiB': peak_allocated_mb,
    #                 'Peak Reserved MiB': peak_reserved_mb,
    #                 'Alloc MiB': current_allocated_mb,
    #                 'Reserved MiB': current_reserved_mb
    #             })

    #             for log in console_logs:
    #                 logger.info(log, pbar=pbar)

    #             wandb.log(wandb_metrics)

    #         if torch_profiler_enabled:
    #             prof.step()

    # wandb.finish()

    # if ddp:
    #     dist.barrier()
    #     destroy_process_group()
