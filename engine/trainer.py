import json
import math
import torch
import torch.distributed as dist

from torch.distributed import destroy_process_group
from torch.distributed.fsdp import MixedPrecisionPolicy
from pathlib import Path
from logger import logger
from engine.context import (
    DistributedContext,
    DeviceContext,
    PrecisionContext,
    TrainerContext
)
from engine.core import TrainerState
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
    load_checkpoint,
    load_model_state,
    load_optimizer_state
)
from lora import (
    apply_lora,
    freeze_non_lora_parameters
)
from tasks import get_task
from engine.optim import (
    classify_trainable_parameters,
    build_optimizer_plan,
    build_optimizers,
    move_optimizer_state_to_param_device_and_dtype
)


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
        self.resolve_optimizer_checkpoints()
        self.compute_max_steps()

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
            reset_optimizer=args.reset_optimizer,
            force_start_step=args.start_step,
            is_master_process=self.distributed_ctx.is_master_process
        )

        if not args.reset_optimizer:
            self.trainer_state.best_val_loss = checkpoint_data.best_val_loss

        if checkpoint_data.metadata.get('training_stage', None) != self.config.training_stage.value:
            logger.info('** WARNING: Training stage has changed **')
            if checkpoint_data.start_step and not args.start_step:
                logger.info('ignoring stored start step...')
                checkpoint_data.start_step=0
            if checkpoint_data.train_loader_state is not None and checkpoint_data.val_loader_state is not None:
                logger.info('ignoring stored metadata for dataset...')
                checkpoint_data.train_loader_state = None
                checkpoint_data.val_loader_state = None
            if checkpoint_data.optimizer_state is not None:
                logger.info('ignoring stored state of optimizer...')
                checkpoint_data.optimizer_state = None
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

    def resolve_optimizer_checkpoints(self):
        if not self.checkpoint_data:
            return
        # current checkpoints logic assumes only adamW state. Will modify later to support Muon combined.
        if self.optimizers.adamw and self.checkpoint_data.optimizer_state is not None:
            load_optimizer_state(self.optimizers.adamw, self.model, self.checkpoint_data.optimizer_state)
            move_optimizer_state_to_param_device_and_dtype(self.optimizers.adamw)
            logger.info('AdamW optimizer state loaded and ready')

    def compute_max_steps(self):
        if self.trainer_state.max_steps <= 0:
            total_tokens = self.train_loader.calculate_max_tokens()
            self.trainer_state.max_steps = math.ceil(total_tokens / self.config.total_batch_size)

    def train(self):
        self.cleanup()

    def cleanup(self):
        if self.distributed_ctx.ddp:
            dist.barrier()
            destroy_process_group()
