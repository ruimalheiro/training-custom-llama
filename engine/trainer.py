import json
import math
import torch
import torch.distributed as dist

from torch.distributed import (
    destroy_process_group,
    broadcast
)
from torch.distributed.tensor import DTensor
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.nn.utils import clip_grad_norm_
from pathlib import Path
from logger import logger
from engine.context import (
    DistributedContext,
    DeviceContext,
    PrecisionContext,
    TrainerContext
)
from engine.core import TrainerState
from engine.optim import (
    classify_trainable_parameters,
    build_optimizer_plan,
    build_optimizers,
    move_optimizer_state_to_param_device
)
from engine.logging import (
    prepare_workload_summary
)
from engine.torch_profiler import (
    init_torch_profiler_context
)
from ddp_utils import (
    init_multi_gpu,
    prepare_model_for_ddp,
    prepare_model_for_fsdp,
    get_model
)
from config import (
    DeviceType,
    TrainingStage,
    TrainingPrecision
)
from hellaswag_utils import load_hellaswag_file
from tokenizer import init_tokenizer
from model import (
    ModelConfig, 
    Transformer
)
from dataloaders import init_data_loaders
from checkpoints import (
    save_checkpoint,
    load_checkpoint,
    load_model_state,
    load_optimizer_state
)
from lora import (
    apply_lora,
    freeze_non_lora_parameters
)
from tasks import get_task
from wandb_utils import WandbWrapper
from lr_schedulers import cosine_scheduler
from tqdm.auto import tqdm


class Trainer:
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.distributed_ctx = None
        self.device_ctx = None
        self.precision_ctx = None
        self.trainer_ctx = None
        self.trainer_state = TrainerState()
        self.callbacks = []
        self.test_generation_prompts = None
        self.hellaswag_data = None
        self.tokenizer = None
        self.model_config = None
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.checkpoint_data = None
        self.task = None
        self.task_assets = None
        self.optimizer_plan = None
        self.optimizers = None
        self.workload_summary = None
        self.wandb = None
        self.torch_profiler_context = None

        self.validate_config()

    def validate_config(self):
        config = self.config

        # device type
        if self.config.device_type != DeviceType.CUDA:
            raise ValueError('Only cuda is supported at the moment.')

        # model config
        if config.dim % config.n_heads != 0:
            raise ValueError(f'"dim" ({config.dim}) must be divisible by "n_heads" ({config.n_heads})')
        if config.n_kv_heads > config.n_heads:
            raise ValueError(f'"n_kv_heads" ({config.n_kv_heads}) must be less or equal to "n_heads" ({config.n_heads})')
        if config.n_heads % config.n_kv_heads != 0:
            raise ValueError(f'"n_heads" ({config.n_heads}) must be divisible by n_kv_heads" ({config.n_kv_heads})')

    def setup(self):
        self.setup_global_torch_optimizations()
        self.build_contexts()
        self.set_logger_master()
        self.load_assets()
        self.build_components()
        self.resolve_checkpoint()
        self.apply_lora()
        self.build_task()
        self.prepare_model_for_distributed_context()
        self.move_task_assets_to_device()
        self.build_optimizer_plan()
        self.build_optimizers()
        self.resolve_optimizers_checkpoint()
        self.prepare_runtime()
        self.prepare_workload_summary_json()
        self.log_workload_summary()
        self.check_all_devices_ready()
        self.setup_wandb()
        self.setup_torch_profiler()

    def setup_global_torch_optimizations(self):
        torch.backends.cuda.matmul.fp32_precision = 'tf32'
        torch.backends.cudnn.conv.fp32_precision = 'tf32'

    def build_contexts(self):
        device = self.build_distributed_context()
        self.build_device_context(device)
        self.build_precision_context()
        self.build_trainer_context()

    def load_assets(self):
        self.load_test_generation_prompts()
        self.load_hellaswag_eval_data()

    def build_components(self):
        self.build_tokenizer()
        self.build_model()
        self.build_data_loaders()

    def resolve_checkpoint(self):
        checkpoint_req = self.resolve_checkpoint_request()
        if checkpoint_req:
            self.load_checkpoint_data(checkpoint_req)
            self.apply_lora_for_checkpoint()
            self.restore_model_state()
            self.restore_data_loaders_state()

    def build_task(self):
        self.task = get_task(self.config.training_stage)
        self.task.setup(config=self.config, ctx=self.trainer_ctx)
        self.task_assets = self.task.build_assets(tokenizer=self.tokenizer, model=self.model)

    def build_distributed_context(self):
        ddp, ddp_rank, ddp_local_rank, ddp_world_size, is_master_process, device = init_multi_gpu(self.config.seed, self.config.device_type.value)
        self.distributed_ctx = DistributedContext(
            ddp=ddp,
            ddp_rank=ddp_rank,
            ddp_local_rank=ddp_local_rank,
            ddp_world_size=ddp_world_size,
            use_fsdp=self.config.use_fsdp,
            is_master_process=is_master_process
        )
        return device

    def build_device_context(self, device):
        self.device_ctx = DeviceContext(
            device_type=self.config.device_type.value,
            device=device
        )

    def set_logger_master(self):
        logger.set_master(self.distributed_ctx.is_master_process)

    def build_precision_context(self):
        if self.config.training_precision == TrainingPrecision.BF16:
            self.precision_ctx = PrecisionContext(
                use_autocast=True,
                scaler=None,
                model_dtype=torch.float32,
                autocast_dtype=torch.bfloat16,
                fsdp_mp=MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32
                ) if self.config.use_fsdp else None
            )
        elif self.config.training_precision == TrainingPrecision.FP16:
            self.precision_ctx = PrecisionContext(
                use_autocast=True,
                scaler=torch.amp.GradScaler(self.device_ctx.device_type), # need gradscaler when fp16
                model_dtype=torch.float32,
                autocast_dtype=torch.float16,
                fsdp_mp=MixedPrecisionPolicy(
                    param_dtype=torch.float16,
                    reduce_dtype=torch.float32
                ) if self.config.use_fsdp else None
            )
        elif self.config.training_precision == TrainingPrecision.FP32:
            self.precision_ctx = PrecisionContext(
                use_autocast=False,
                scaler=None,
                model_dtype=torch.float32,
                autocast_dtype=torch.float32,
                fsdp_mp=None
            )
        else:
            raise ValueError('Invalid training precision')

    def compute_grad_accum_steps(self, ddp_world_size):
        #### BATCH SIZE CHECKS
        # NOTE: total_batch_size is the total batch size in tokens. The model max_batch_size is the number of sequences per device during forward pass (micro batches).
        # The total batch size must be a multiple of (max_batch_size * max_seq_len * ddp_world_size). This is needed for the gradient accumulation steps to be calculated correctly.
        if self.config.total_batch_size % (self.config.max_batch_size * self.config.max_seq_len * ddp_world_size) != 0:
            raise ValueError('total_batch_size must be divisible by (max_batch_size * max_seq_len * ddp_world_size)')

        # Gradient accumulation steps
        grad_accum_steps = self.config.total_batch_size // (self.config.max_batch_size * self.config.max_seq_len * ddp_world_size)

        # Final check to validate previous calculations.
        if self.config.total_batch_size != (self.config.max_batch_size * self.config.max_seq_len * ddp_world_size * grad_accum_steps):
            raise ValueError('total batch size MUST EQUAL (max_batch_size * max_seq_len * ddp_world_size * grad_accum_steps)')

        return grad_accum_steps

    def build_trainer_context(self):
        self.trainer_ctx = TrainerContext(
            distributed=self.distributed_ctx,
            device=self.device_ctx,
            precision=self.precision_ctx,
            grad_accum_steps=self.compute_grad_accum_steps(self.distributed_ctx.ddp_world_size)
        )

    def load_test_generation_prompts(self):
        if self.config.generate_every_x_steps <= 0:
            return
        test_prompts_data=json.loads(Path(self.config.test_prompts_path).read_text())
        stage = self.config.training_stage.value
        if stage not in test_prompts_data:
            raise ValueError(f'Missing test prompts for training stage: {stage}')
        self.test_generation_prompts = test_prompts_data[stage]

    def load_hellaswag_eval_data(self):
        if self.config.hellaswag_every_x_steps <= 0 or self.config.training_stage != TrainingStage.PRETRAIN:
            return
        self.hellaswag_data = load_hellaswag_file(
            self.config.hellaswag_path,
            self.distributed_ctx.ddp,
            self.distributed_ctx.is_master_process,
            size=self.config.hellaswag_number_of_examples
        )
    
    def build_tokenizer(self):
        self.tokenizer = init_tokenizer(self.config.tokenizer_checkpoint_path, self.config.huggingface_tokenizer)

    def build_model(self):
        config = self.config
        tokenizer = self.tokenizer
        self.model_config = ModelConfig(
            dim=config.dim,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            multiple_of=config.multiple_of,
            ffn_dim_multiplier=config.ffn_dim_multiplier,
            norm_eps=config.norm_eps,
            rope_theta=config.rope_theta,
            max_batch_size=config.max_batch_size,
            max_seq_len=config.max_seq_len,
            # tokenizer aux config
            tokenizer=tokenizer,
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_id,
            stop_tokens=tokenizer.stop_tokens,
            ignore_index=config.ignore_index,
            # moe
            is_moe=config.is_moe,
            moe_num_experts=config.moe_num_experts,
            moe_expert_dim=config.moe_expert_dim,
            moe_top_k=config.moe_top_k,
            moe_load_balancing_coef=config.moe_load_balancing_coef,
            moe_z_loss_coef=config.moe_z_loss_coef,
            moe_compute_stats=config.moe_compute_stats
        )
        self.model = Transformer(self.model_config)

    def build_data_loaders(self):
        self.train_loader, self.val_loader = init_data_loaders(
            batch_size=self.config.max_batch_size,
            sequence_length=self.config.max_seq_len,
            is_master_process=self.distributed_ctx.is_master_process,
            process_rank=self.distributed_ctx.ddp_rank,
            num_processes=self.distributed_ctx.ddp_world_size,
            data_root=self.config.dataloader_root_path,
            pad_id=self.tokenizer.pad_id,
            training_stage=self.config.training_stage.value
        )

    def resolve_checkpoint_request(self):
        args = self.args

        selected = [
            bool(args.pretrain_checkpoint),
            bool(args.instruct_checkpoint),
            bool(args.dpo_checkpoint),
        ]
        selected_sum = sum(selected)
        if selected_sum == 0:
            return None
        elif selected_sum > 1:
            raise ValueError('Only one checkpoint argument can be provided.')

        path = None
        name = None
        if args.pretrain_checkpoint:
            path = self.config.pretrain_load_checkpoints_path
            name = args.pretrain_checkpoint
        elif args.instruct_checkpoint:
            path = self.config.instruct_load_checkpoints_path
            name = args.instruct_checkpoint
        elif args.dpo_checkpoint:
            path = self.config.dpo_load_checkpoints_path
            name = args.dpo_checkpoint
        return (path, name)

    def load_checkpoint_data(self, checkpoint_request):
        args = self.args
        path, name = checkpoint_request

        checkpoint_data = load_checkpoint(
            path,
            name,
            reset_optimizers=args.reset_optimizers,
            is_master_process=self.distributed_ctx.is_master_process
        )

        training_stage_changed = (
            checkpoint_data.metadata.get('training_stage', None) != self.config.training_stage.value
        )

        if training_stage_changed:
            logger.info('** WARNING: Training stage has changed **')
            if checkpoint_data.resume_step:
                logger.info('ignoring stored resume step...')
                checkpoint_data.resume_step=0
            if checkpoint_data.train_loader_state is not None and checkpoint_data.val_loader_state is not None:
                logger.info('ignoring stored metadata for dataset...')
                checkpoint_data.train_loader_state = None
                checkpoint_data.val_loader_state = None
            if checkpoint_data.optimizers_state is not None:
                logger.info('ignoring stored state of optimizer(s)...')
                checkpoint_data.optimizers_state = None
            logger.info('\n')
        
        self.checkpoint_data = checkpoint_data

    def apply_lora_modification(self):
        config = self.config
        apply_lora(
            self.model,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            device=self.device_ctx.device,
            is_master_process=self.distributed_ctx.is_master_process
        )
        # by default we freeze the other parameters
        freeze_non_lora_parameters(self.model)

    def apply_lora_for_checkpoint(self):
        if self.checkpoint_data.is_lora_checkpoint:
            if not self.config.lora_enabled:
                raise ValueError('"lora_enabled" must be set to True when loading checkpoint that includes LoRA')
            self.apply_lora_modification()

    def apply_lora(self):
        if (
            self.config.lora_enabled and
            (
                (not self.checkpoint_data) or
                (self.checkpoint_data and not self.checkpoint_data.is_lora_checkpoint)
            )
        ):
            self.apply_lora_modification()

    def restore_model_state(self):
        load_model_state(self.model, self.checkpoint_data.model_state)
        logger.info('\nModel loading')
        logger.info('----------------------------------------')
        logger.info('Model checkpoint loaded and ready')

    def restore_data_loaders_state(self):
        checkpoint_data = self.checkpoint_data
        if checkpoint_data.train_loader_state is not None and checkpoint_data.val_loader_state is not None:
            self.train_loader.load_state_dict(checkpoint_data.train_loader_state)
            self.val_loader.load_state_dict(checkpoint_data.val_loader_state)

    def compile_model(self):
        if self.config.use_torch_compile:
            self.model.compile()

    def prepare_model_for_distributed_context(self):
        device = self.device_ctx.device
        ddp_local_rank = self.distributed_ctx.ddp_local_rank
        fsdp_mp = self.precision_ctx.fsdp_mp
        model_dtype = self.precision_ctx.model_dtype

        if self.config.use_fsdp:
            if not dist.is_initialized():
                raise ValueError('dist must be initialized if "USE_FSDP" flag is set.')
            if self.config.use_torch_compile:
                raise ValueError('Currently not supporting torch compile for FSDP. Please set "USE_TORCH_COMPILE" flag to False.')
            logger.info('\nFSDP')
            logger.info('----------------------------------------')
            logger.info('Wrapping the model in preparation for FSDP')
            # for FSDP no need to move explicitly to device here as that would actually cost more VRAM, instead let FSDP initialization alocate the shard to the device id (ddp_local_rank).
            self.model = prepare_model_for_fsdp(self.model, ddp_local_rank, fsdp_mp)
        else:
            # move to gpu
            self.model.to(device=device, dtype=model_dtype)
            if dist.is_initialized():
                logger.info('\nDDP')
                logger.info('----------------------------------------')
                logger.info('Wrapping the model in preparation for DDP')
                self.model = prepare_model_for_ddp(self.model, ddp_local_rank)
            self.compile_model()

    def move_task_assets_to_device(self):
        self.task_assets = self.task.move_assets_to_device(self.task_assets)

    def build_optimizer_plan(self):
        parameter_buckets = classify_trainable_parameters(get_model(self.model))
        self.optimizer_plan = build_optimizer_plan(self.config, parameter_buckets)

    def build_optimizers(self):
        self.optimizers = build_optimizers(self.config, self.optimizer_plan)

    def resolve_optimizers_checkpoint(self):
        if not self.checkpoint_data or not self.checkpoint_data.optimizers_state:
            return
        if self.optimizers.adamw and self.checkpoint_data.optimizers_state['adamw']:
            load_optimizer_state(self.optimizers.adamw, get_model(self.model), self.checkpoint_data.optimizers_state['adamw'])
            move_optimizer_state_to_param_device(self.optimizers.adamw)
            logger.info('AdamW optimizer state loaded and ready')
        if self.optimizers.muon and self.checkpoint_data.optimizers_state['muon']:
            load_optimizer_state(self.optimizers.muon, get_model(self.model), self.checkpoint_data.optimizers_state['muon'])
            move_optimizer_state_to_param_device(self.optimizers.muon)
            logger.info('Muon optimizer state loaded and ready')

    def prepare_runtime(self):
        self.trainer_state.max_steps = self.config.max_steps
        if self.trainer_state.max_steps <= 0:
            self.trainer_state.max_steps = math.ceil(self.train_loader.calculate_max_tokens() / self.config.total_batch_size)
        if self.checkpoint_data:
            self.trainer_state.start_step = self.checkpoint_data.resume_step if self.args.start_step is None else self.args.start_step
            self.trainer_state.current_step = self.checkpoint_data.resume_step if self.args.start_step is None else self.args.start_step
            if not self.args.reset_optimizers:
                self.trainer_state.last_val_loss = self.checkpoint_data.last_val_loss
                self.trainer_state.best_val_loss = self.checkpoint_data.best_val_loss
        else:
            self.trainer_state.start_step = 0 if self.args.start_step is None else self.args.start_step
            self.trainer_state.current_step = 0 if self.args.start_step is None else self.args.start_step

    def prepare_workload_summary_json(self):
        if self.distributed_ctx.is_master_process:
            self.workload_summary = prepare_workload_summary(
                config=self.config,
                model_config=self.model_config,
                checkpoint_data=self.checkpoint_data,
                trainer_ctx=self.trainer_ctx,
                optimizer_plan=self.optimizer_plan,
                trainer_state=self.trainer_state,
                model_params_count=get_model(self.model).get_total_parameters_count(),
                model_trainable_params_count=get_model(self.model).get_trainable_parameters_count(),
                total_tokens=self.train_loader.calculate_max_tokens()
            )

    def log_workload_summary(self):
        if self.distributed_ctx.is_master_process:
            logger.info(f'\n{self.config.training_stage.value.upper()} WORKLOAD SUMMARY:')
            logger.info('--------------------------------------------------------')
            logger.info(self.workload_summary, is_json=True)
            logger.info('--------------------------------------------------------')

    def setup_wandb(self):
        self.wandb = WandbWrapper(
            enabled=self.config.wandb_enabled,
            is_master_process=self.distributed_ctx.is_master_process
        )
        self.wandb.init(
            self.config.wandb_project_name,
            job_name=self.config.wandb_run_name,
            config=self.workload_summary
        )

    def setup_torch_profiler(self):
        self.torch_profiler_context = init_torch_profiler_context(
            self.config,
            self.distributed_ctx
        )

    def check_all_devices_ready(self):
        if self.distributed_ctx.ddp and dist.is_initialized():
            dist.barrier()
        logger.info(f'\nDevice: {self.distributed_ctx.ddp_local_rank} is ready.', True)

    def should_run(self, step, every, last_step, run_last_step=True):
        if every == -1:
            return run_last_step and last_step
        return (step > 0 and step % every == 0) or (run_last_step and last_step)

    def clip_grad_norm(self, model, max_norm):
        norm = clip_grad_norm_(model.parameters(), max_norm)
        if isinstance(norm, DTensor):
            return norm.to_local()
        return norm

    def run_train(self):
        pass

    def run_validation(self):
        pass

    def run_save_checkpoint(self, pbar=None):
        if (
            not self.config.save_checkpoints or
            self.config.save_best_only and self.trainer_state.num_val_runs_no_improve > 0
        ):
            return
        save_checkpoint(
            self.config.save_checkpoints_path,
            get_model(self.model),
            self.model_config,
            self.config,
            self.trainer_state.current_step,
            self.trainer_state.last_val_loss,
            self.trainer_state.best_val_loss,
            self.optimizers,
            self.train_loader,
            self.val_loader,
            {
                'training_stage': self.config.training_stage.value,
                'lora_enabled': self.config.lora_enabled
            },
            self.config.max_number_checkpoints,
            self.distributed_ctx.is_master_process,
            pbar
        )

    def run_hellaswag_eval(self):
        pass

    def run_generation(self):
        pass

    def process_step(self, pbar):
        step = self.trainer_state.current_step
        is_last_step = self.trainer_state.is_last_step

        self.run_train()
        if self.should_run(step, self.config.validate_every_x_steps, is_last_step):
            self.run_validation()
        if self.should_run(step, self.config.save_every_x_steps, is_last_step):
            self.run_save_checkpoint(pbar)
        if self.should_run(step, self.config.hellaswag_every_x_steps, is_last_step):
            self.run_hellaswag_eval()
        if self.should_run(step, self.config.generate_every_x_steps, is_last_step):
            self.run_generation()

    def start_training_loop(self):
        is_master_process = self.trainer_ctx.distributed.is_master_process
        ddp_rank = self.trainer_ctx.distributed.ddp_rank
        device = self.trainer_ctx.device.device
        start_step = self.trainer_state.start_step
        current_step = self.trainer_state.current_step
        max_steps = self.trainer_state.max_steps

        tqdm_label = f'Training ({self.config.training_stage.value})'
        abort_signal = torch.tensor([0], device=device)
        early_stopping_patience_skip_steps = self.config.early_stopping_patience_skip_steps + self.trainer_state.start_step

        with self.torch_profiler_context as torch_profiler_ctx:
            pbar = tqdm(
                range(start_step, max_steps),
                initial=current_step,
                total=max_steps,
                desc=tqdm_label,
                disable=not is_master_process,
                dynamic_ncols=True
            )
            for step in pbar:
                if abort_signal.item() == 1:
                    logger.info(f'Rank {ddp_rank} received stop signal.', True)
                    break
                self.trainer_state.current_step = step
                self.trainer_state.is_last_step = (step == max_steps - 1)
                self.process_step(pbar)
                torch_profiler_ctx.step()
                abort_signal[0] = 1 if self.trainer_state.should_stop else 0
                if self.distributed_ctx.ddp:
                    broadcast(abort_signal, src=0)

    def cleanup(self):
        if self.wandb:
            self.wandb.finish()

        if self.distributed_ctx and self.distributed_ctx.ddp and dist.is_initialized():
            dist.barrier()
            destroy_process_group()

    def train(self):
        try:
            self.setup()
            self.start_training_loop()
        finally:
            self.cleanup()
