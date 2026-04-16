import argparse

from config import config
from engine import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script options')
    parser.add_argument('--pretrain_checkpoint', type=str, default=None, help='Pretrain checkpoint to load.')
    parser.add_argument('--instruct_checkpoint', type=str, default=None, help='Instruct checkpoint to load.')
    parser.add_argument('--dpo_checkpoint', type=str, default=None, help='DPO checkpoint to load.')
    parser.add_argument('--reset-optimizers', action='store_true', help='Reset the optimizers state when loading a checkpoint.')
    parser.add_argument('--start-step', type=int, default=None, help='Starting step number for training.')

    args = parser.parse_args()

    trainer = Trainer(
        config = config,
        args=args
    )
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
    # #### COMPILE
    # #### PREPARE DDP / FSDP
    # #### PREPARE OPTIMIZER OPTIMAL PARAM GROUPS
    # #### INIT OPTIMIZER
    # #### TRAINING LOOP
    # COMPUTE TOKENS
    # COMPUTE MAX STEPS
    # DEVICE READY CHECK
    # # Workload summary










# if should_run(step, validate_every_x_steps, last_step):
#     model.eval()
#     get_model(model).enable_moe_stats()
#     get_model(model).reset_moe_stats()

#     val_loss_sum = torch.tensor(0.0, device=device)
#     val_tok_sum = torch.tensor(0.0, device=device)
#     console_logs = []
#     val_metric_sums = {}
#     val_metric_weights = {}
#     with torch.no_grad():
#         for _ in tqdm(range(val_steps), 'Validating', disable=not is_master_process, leave=False):
#             val_output = task.validation_step(model, val_loader.next_batch())
#             loss = val_output.loss
#             n_valid = val_output.n_valid

#             console_logs.extend(val_output.console_logs)
#             accumulate_weighted_metrics(
#                 weight=n_valid,
#                 metrics=val_output.metrics,
#                 metrics_sum_acc=val_metric_sums,
#                 metrics_weights_acc=val_metric_weights,
#                 device=device
#             )

#             val_loss_sum += loss * n_valid
#             val_tok_sum += n_valid

#     if ddp:
#         dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
#         dist.all_reduce(val_tok_sum, op=dist.ReduceOp.SUM)

#     val_ce = (val_loss_sum / val_tok_sum).item()

#     aggregated_val_metrics = combine_weighted_metrics(
#         metrics_sum_acc=val_metric_sums,
#         metrics_weights_acc=val_metric_weights,
#         ddp=ddp
#     )

#     # MOE METRICS
#     moe_metrics = None
#     if config.is_moe and config.moe_compute_stats:
#         moe_metrics = collect_moe_metrics(get_model(model), ddp, is_master_process)

#     get_model(model).disable_moe_stats()

#     if is_master_process:
#         logger.info(f'\nValidation loss: {val_ce:.4f}', pbar=pbar)
#         wandb_metrics = {'Validation Loss': val_ce}

#         for log in console_logs:
#             logger.info(log, pbar=pbar)
#         wandb_metrics.update(aggregated_val_metrics)

#         if moe_metrics:
#             wandb_metrics.update(moe_metrics)
#         wandb.log(wandb_metrics)

#     if val_ce < best_val_loss:
#         best_val_loss = val_ce
#         epochs_no_improve = 0

#         if save_checkpoints is True:
#             save_checkpoint(
#                 save_checkpoints_path,
#                 model,
#                 model_config,
#                 step,
#                 val_ce,
#                 optimizer,
#                 train_loader,
#                 val_loader,
#                 extra_checkpoint_metadata,
#                 max_number_checkpoints,
#                 is_master_process,
#                 pbar
#             )
#     else:
#         if step > early_stopping_patience_skip_steps:
#             epochs_no_improve += 1
#             logger.info(f'Validation loss did not improve. Best: {best_val_loss}, Latest: {val_ce} - Attempts left: {early_stopping_patience - epochs_no_improve}', pbar=pbar)
#         else:
#             logger.info(f'Validation loss did not improve. Best: {best_val_loss}, Latest: {val_ce} - (Skip phase...) steps left to skip: {early_stopping_patience_skip_steps - step}', pbar=pbar)

#         logger.info('Skipping save checkpoint...', pbar=pbar)

#     if epochs_no_improve == early_stopping_patience:
#         logger.info(f'The validation loss did not improve for: {early_stopping_patience} - Aborting training...', pbar=pbar)
#         abort_if_no_improve[0] = 1

#     if ddp:
#         broadcast(abort_if_no_improve, src=0)

# if is_pretraining and should_run(step, hellaswag_every_x_steps, last_step):
#     model.eval()
#     num_correct_norm = 0
#     num_total = 0
#     for example in tqdm(HELLASWAG_DATA, 'HellaSwag validation', unit=' examples', disable=not is_master_process, leave=False):
#         tokens, mask, label, valid = example['tokens'], example['mask'], example['label'], example['valid']
#         tokens = tokens.to(device)
#         mask = mask.to(device)

#         # get the logits
#         with torch.no_grad():
#             with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=use_autocast):
#                 logits = model(tokens)['logits']

#         if valid: # Some examples might be dummy in FSDP
#             predicted_correct = estimate_correct_candidate_selection(tokens, mask, logits)
#             num_total += 1
#             num_correct_norm += int(predicted_correct == label)

#     # Aggregates counts across all GPU processes (in DDP) to compute global accuracy.
#     if ddp:
#         num_total = torch.tensor(num_total, dtype=torch.long, device=device)
#         num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
#         dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
#         dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
#         num_total = num_total.item()
#         num_correct_norm = num_correct_norm.item()
#     acc_norm = num_correct_norm / num_total

#     if is_master_process:
#         logger.info(f'HellaSwag accuracy: {num_correct_norm} / {num_total} = {acc_norm:.4f}', pbar=pbar)
#         wandb.log({'HellaSwag accuracy': acc_norm})

# if should_run(step, generate_every_x_steps, last_step):
#     model.eval()
#     logger.info('-----------------------------------------------', pbar=pbar)
#     for text in generate_and_decode(
#         model=get_model(model),
#         texts=test_generation_prompts,
#         max_gen_len=max_test_gen_len,
#         full_seq=True,
#         device=device,
#         is_instruct=is_instruct_training,
#         temperature=0.0,
#         top_p=1.0,
#         use_kv_cache=True
#     ):
#         logger.info(text, pbar=pbar)
#     logger.info('-----------------------------------------------', pbar=pbar)

