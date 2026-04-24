import os

from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings
from enum import Enum
from typing import Tuple, Annotated


class DeviceType(str, Enum):
    CUDA = 'cuda'

class TrainingStage(str, Enum):
    PRETRAIN = 'pretrain'
    INSTRUCT = 'instruct'
    DPO = 'dpo'

class TrainingPrecision(str, Enum):
    BF16 = 'bf16'
    FP16 = 'fp16'
    FP32 = 'fp32'

class TrainConfig(BaseSettings):
    # third party envs
    wandb_api_key: Annotated[str | None, Field(alias='WANDB_API_KEY', exclude=True)] = None
    hf_token: Annotated[str | None, Field(alias='HF_TOKEN', exclude=True)] = None
    hf_home: str = Field(default='./cache', alias='HF_HOME')

    # datasets
    pretrain_dataset_mix_file: str = Field(alias='HF_PRETRAIN_DATASET_MIX_FILE')
    pretrain_dataset_target_path: str = Field(alias='HF_PRETRAIN_DATASET_TARGET_PATH')

    instruct_dataset_mix_file: Annotated[str | None, Field(alias='HF_INSTRUCT_DATASET_MIX_FILE')] = None
    instruct_dataset_target_path: str = Field(alias='HF_INSTRUCT_DATASET_TARGET_PATH')

    dpo_dataset_mix_file: Annotated[str | None, Field(alias='HF_DPO_DATASET_MIX_FILE')] = None
    dpo_dataset_target_path: str = Field(alias='HF_DPO_DATASET_TARGET_PATH')

    hf_include_source_id: bool = Field(default=False, alias='HF_INCLUDE_SOURCE_ID')

    # processes and batch sizes
    number_of_cpu_processes: int = Field(default=0, alias='NUMBER_OF_CPU_PROCESSES')
    mp_pool_chunk_size: int = Field(default=64, alias='MP_POOL_CHUNK_SIZE')
    hf_map_batch_size: int = Field(default=1000, alias='HF_MAP_BATCH_SIZE')
    hf_map_writer_batch_size: int = Field(default=1000, alias='HF_MAP_WRITER_BATCH_SIZE')

    # torch profiler
    torch_profiler_enabled: bool = Field(default=False, alias='TORCH_PROFILER_ENABLED')
    torch_profiler_schedule_skip_first: int = Field(default=0, alias='TORCH_PROFILER_SCHEDULE_SKIP_FIRST')
    torch_profiler_schedule_wait: int = Field(default=1, alias='TORCH_PROFILER_SCHEDULE_WAIT')
    torch_profiler_schedule_warmup: int = Field(default=1, alias='TORCH_PROFILER_SCHEDULE_WARMUP')
    torch_profiler_schedule_active: int = Field(default=1, alias='TORCH_PROFILER_SCHEDULE_ACTIVE')
    torch_profiler_schedule_repeat: int = Field(default=0, alias='TORCH_PROFILER_SCHEDULE_REPEAT')

    # paths for dataloaders
    pretrain_dataloader_root_path: str = Field(alias='PRETRAIN_DATALOADER_ROOT_PATH')
    instruct_dataloader_root_path: str = Field(alias='INSTRUCT_DATALOADER_ROOT_PATH')
    dpo_dataloader_root_path: str = Field(alias='DPO_DATALOADER_ROOT_PATH')

    # paths for eval datasets
    hellaswag_path: str = Field(alias='HELLASWAG_PATH')
    winogrande_path: str = Field(alias='WINOGRANDE_PATH')

    # system prompt
    system_prompt: str = Field(default='You are a helpful AI assistant', alias='SYSTEM_PROMPT')

    # save / load path
    pretrain_save_checkpoints_path: str = Field(alias='PRETRAIN_SAVE_CHECKPOINTS_PATH')
    pretrain_load_checkpoints_path: str = Field(alias='PRETRAIN_LOAD_CHECKPOINTS_PATH')
    instruct_save_checkpoints_path: str = Field(alias='INSTRUCT_SAVE_CHECKPOINTS_PATH')
    instruct_load_checkpoints_path: str = Field(alias='INSTRUCT_LOAD_CHECKPOINTS_PATH')
    dpo_save_checkpoints_path: str = Field(alias='DPO_SAVE_CHECKPOINTS_PATH')
    dpo_load_checkpoints_path: str = Field(alias='DPO_LOAD_CHECKPOINTS_PATH')

    save_checkpoints: bool = Field(default=False, alias='SAVE_CHECKPOINTS')
    save_best_only: bool = Field(default=False, alias='SAVE_BEST_ONLY')
    save_every_x_steps: int = Field(alias='SAVE_EVERY_X_STEPS')
    max_number_checkpoints: int = Field(default=2, alias='MAX_NUMBER_CHECKPOINTS')

    # wandb
    wandb_enabled: bool = Field(default=False, alias='WANDB_ENABLED')
    wandb_project_name: str = Field(alias='WANDB_PROJECT_NAME')
    wandb_run_name: str = Field(default=None, alias='WANDB_RUN_NAME')

    # tokenizer model path
    tokenizer_checkpoint_path: str = Field(alias='TOKENIZER_CHECKPOINT_PATH')
    huggingface_tokenizer: bool = Field(default=True, alias='HUGGINGFACE_TOKENIZER')

    # value to mask the padded tokens in the loss calculation
    ignore_index: int = Field(default=-100, alias='IGNORE_INDEX')

    # train config
    seed: int = Field(default=42, alias='SEED')
    device_type: DeviceType = Field(default=DeviceType.CUDA, alias='DEVICE_TYPE')
    training_precision: TrainingPrecision = Field(default=TrainingPrecision.BF16, alias='TRAINING_PRECISION')
    training_stage: TrainingStage = Field(default=TrainingStage.PRETRAIN, alias='TRAINING_STAGE')
    total_batch_size: int = Field(alias='TOTAL_BATCH_SIZE')
    max_steps: int = Field(default=-1, alias='MAX_STEPS') # If not set, it is aprox calculated

    adamw_min_lr: float = Field(alias='ADAMW_MIN_LR')
    adamw_max_lr: float = Field(alias='ADAMW_MAX_LR')
    adamw_weight_decay: float = Field(alias='ADAMW_WEIGHT_DECAY')
    adamw_betas: Tuple[float, float] = Field(default=(0.9, 0.95), alias='ADAMW_BETAS')
    adamw_use_fused: Annotated[bool | None, Field(alias='ADAMW_USE_FUSED')] = None
    adamw_warmup_steps: int = Field(alias='ADAMW_WARMUP_STEPS')

    use_muon: bool = Field(default=False, alias='USE_MUON')
    muon_min_lr: float = Field(alias='MUON_MIN_LR')
    muon_max_lr: float = Field(alias='MUON_MAX_LR')
    muon_weight_decay: float = Field(alias='MUON_WEIGHT_DECAY')
    muon_momentum: float = Field(alias='MUON_MOMENTUM', default=0.95)
    muon_warmup_steps: int = Field(alias='MUON_WARMUP_STEPS')

    early_stopping_patience: int = Field(alias='EARLY_STOPPING_PATIENCE')
    early_stopping_patience_skip_steps: int = Field(alias='EARLY_STOPPING_PATIENCE_SKIP_STEPS')
    dpo_beta: float = Field(default=0.1, alias='DPO_BETA')
    is_model_distillation: bool = Field(alias='IS_MODEL_DISTILLATION')
    distillation_temperature: float = Field(alias='DISTILLATION_TEMPERATURE')
    # The teacher model is loader via huggingface API: AutoModelForCausalLM.from_pretrained(teacher_model_checkpoint, ...) so needs to ve a valid checkpoint.
    teacher_model_checkpoint: str = Field(alias='TEACHER_MODEL_CHECKPOINT')
    lora_enabled: bool = Field(default=False, alias='LORA_ENABLED')
    lora_rank: int = Field(default=16, alias='LORA_RANK')
    lora_alpha: int = Field(default=8, alias='LORA_ALPHA')
    lora_dropout: float = Field(default=0.05, alias='LORA_DROPOUT')
    lora_target_modules: list[str] = Field(alias='LORA_TARGET_MODULES')
    use_torch_compile: bool = Field(default=False, alias='USE_TORCH_COMPILE') # Will add more options later.
    use_fsdp: bool = Field(default=False, alias='USE_FSDP')

    # validation
    validate_every_x_steps: int = Field(alias='VALIDATE_EVERY_X_STEPS')
    validation_steps: int = Field(alias='VALIDATION_STEPS')

    # evals
    hellaswag_every_x_steps: int = Field(alias='HELLASWAG_EVERY_X_STEPS')
    hellaswag_number_of_examples: int = Field(alias='HELLASWAG_NUMBER_OF_EXAMPLES')
    winogrande_every_x_steps: int = Field(alias='WINOGRANDE_EVERY_X_STEPS')
    winogrande_number_of_examples: int = Field(alias='WINOGRANDE_NUMBER_OF_EXAMPLES')

    # generation
    test_prompts_path: str = Field(alias='TEST_PROMPTS_FILE')
    generate_every_x_steps: int = Field(alias='GENERATE_EVERY_X_STEPS')
    max_test_gen_len: int = Field(alias='MAX_TEST_GEN_LEN')

    # model architecture config
    dim: int = Field(default=768, alias='DIM')
    n_layers: int = Field(default=16, alias='N_LAYERS')
    n_heads: int = Field(default=16, alias='N_HEADS')
    n_kv_heads: int = Field(default=8, alias='N_KV_HEADS')
    multiple_of: int = Field(default=1024, alias='MULTIPLE_OF')
    ffn_dim_multiplier: float = Field(default=1.0, alias='FFN_DIM_MULTIPLIER')
    norm_eps: float = Field(default=1e-05, alias='NORM_EPS')
    is_rope_cis: bool = Field(default=False, alias='IS_ROPE_CIS')
    rope_theta: float = Field(default=500000.0, alias='ROPE_THETA')
    max_batch_size: int = Field(default=4, alias='MAX_BATCH_SIZE')
    max_seq_len: int = Field(default=1024, alias='MAX_SEQ_LEN')
    is_moe: bool = Field(default=False, alias='IS_MOE')
    moe_num_experts: int = Field(alias='MOE_NUM_EXPERTS')
    moe_expert_dim: int = Field(alias='MOE_EXPERT_DIM')
    moe_top_k: int = Field(alias='MOE_TOP_K')
    moe_load_balancing_coef: float = Field(alias='MOE_LOAD_BALANCING_COEF')
    moe_z_loss_coef: float = Field(alias='MOE_Z_LOSS_COEF')
    moe_compute_stats: bool = Field(alias='MOE_COMPUTE_STATS')

    #### DERIVED PROPERTIES ####
    is_pretraining: bool = Field(default=False, repr=False)
    is_instruct_training: bool = Field(default=False, repr=False)
    is_dpo_training: bool = Field(default=False, repr=False)

    dataloader_root_path: str = Field(default='', repr=False)
    save_checkpoints_path: str = Field(default='', repr=False)

    def model_post_init(self, __context: any) -> None:
        self.is_pretraining = self.training_stage == TrainingStage.PRETRAIN
        self.is_instruct_training = self.training_stage == TrainingStage.INSTRUCT
        self.is_dpo_training = self.training_stage == TrainingStage.DPO

        if self.is_pretraining:
            self.dataloader_root_path = self.pretrain_dataloader_root_path
            self.save_checkpoints_path = self.pretrain_save_checkpoints_path
        elif self.is_instruct_training:
            self.dataloader_root_path = self.instruct_dataloader_root_path
            self.save_checkpoints_path = self.instruct_save_checkpoints_path
        elif self.is_dpo_training:
            self.dataloader_root_path = self.dpo_dataloader_root_path
            self.save_checkpoints_path = self.dpo_save_checkpoints_path
        else:
            raise ValueError(f'Invalid training stage: {self.training_stage}')

        self.configure_hf_environment()

    def configure_hf_environment(self) -> None:
        # Sets default paths for hf
        os.environ['HF_HOME'] = self.hf_home
        os.environ['HF_DATASETS_CACHE'] = f'{self.hf_home}/datasets'
        os.environ['HF_HUB_CACHE'] = f'{self.hf_home}/hub'

    def to_summary_dict(self, include_model_config: bool = True) -> dict:
        data = self.model_dump(exclude={'wandb_api_key', 'hf_token'})
        summary = {
            'third_part': {
                'hf_home': data['hf_home']
            },
            'runtime': {
                'number_of_cpu_processes': data['number_of_cpu_processes'],
                'mp_pool_chunk_size': data['mp_pool_chunk_size'],
                'hf_map_batch_size': data['hf_map_batch_size'],
                'hf_map_writer_batch_size': data['hf_map_writer_batch_size'],
                'use_torch_compile': data['use_torch_compile'],
                'use_fsdp': data['use_fsdp']
            },
            'torch_profiler': {
                'torch_profiler_enabled': data['torch_profiler_enabled'],
                'torch_profiler_schedule_skip_first': data['torch_profiler_schedule_skip_first'],
                'torch_profiler_schedule_wait': data['torch_profiler_schedule_wait'],
                'torch_profiler_schedule_warmup': data['torch_profiler_schedule_warmup'],
                'torch_profiler_schedule_active': data['torch_profiler_schedule_active'],
                'torch_profiler_schedule_repeat': data['torch_profiler_schedule_repeat']
            },
            'datasets': {
                'pretrain_dataset_mix_file': data['pretrain_dataset_mix_file'],
                'pretrain_dataset_target_path': data['pretrain_dataset_target_path'],
                'instruct_dataset_mix_file': data['instruct_dataset_mix_file'],
                'instruct_dataset_target_path': data['instruct_dataset_target_path'],
                'dpo_dataset_mix_file': data['dpo_dataset_mix_file'],
                'dpo_dataset_target_path': data['dpo_dataset_target_path'],
                'hf_include_source_id': data['hf_include_source_id']
            },
            'eval_data': {
                'hellaswag_path': data['hellaswag_path'],
                'winogrande_path': data['winogrande_path']
            },
            'generation_data': {
                'test_prompts_path': data['test_prompts_path']
            },
            'dataloaders': {
                'pretrain_dataloader_root_path': data['pretrain_dataloader_root_path'],
                'instruct_dataloader_root_path': data['instruct_dataloader_root_path'],
                'dpo_dataloader_root_path': data['dpo_dataloader_root_path'],
                'dataloader_root_path': data['dataloader_root_path']
            },
            'checkpoints': {
                'pretrain_save_checkpoints_path': data['pretrain_save_checkpoints_path'],
                'pretrain_load_checkpoints_path': data['pretrain_load_checkpoints_path'],
                'instruct_save_checkpoints_path': data['instruct_save_checkpoints_path'],
                'instruct_load_checkpoints_path': data['instruct_load_checkpoints_path'],
                'dpo_save_checkpoints_path': data['dpo_save_checkpoints_path'],
                'dpo_load_checkpoints_path': data['dpo_load_checkpoints_path'],
                'save_checkpoints': data['save_checkpoints'],
                'save_best_only': data['save_best_only'],
                'save_every_x_steps': data['save_every_x_steps'],
                'max_number_checkpoints': data['max_number_checkpoints']
            },
            'wandb': {
                'wandb_enabled': data['wandb_enabled'],
                'wandb_project_name': data['wandb_project_name'],
                'wandb_run_name': data['wandb_run_name']
            },
            'system_prompt': data['system_prompt'],
            'tokenizer': {
                'huggingface_tokenizer': data['huggingface_tokenizer'],
                'tokenizer_checkpoint_path': data['tokenizer_checkpoint_path']
            },
            'padding': {
                'ignore_index': data['ignore_index']
            },
            'distillation_config': {
                'is_model_distillation': data['is_model_distillation'],
                'distillation_temperature': data['distillation_temperature'],
                'teacher_model_checkpoint': data['teacher_model_checkpoint']
            },
            'dpo_config': {
                'dpo_beta': data['dpo_beta']
            },
            'lora': {
                'lora_enabled': data['lora_enabled'],
                'lora_rank': data['lora_rank'],
                'lora_alpha': data['lora_alpha'],
                'lora_dropout': data['lora_dropout'],
                'lora_target_modules': data['lora_target_modules']
            },
            'optimizers_config': {
                'adamw': {
                    'adamw_min_lr': data['adamw_min_lr'],
                    'adamw_max_lr': data['adamw_max_lr'],
                    'adamw_weight_decay': data['adamw_weight_decay'],
                    'adamw_betas': data['adamw_betas'],
                    'adamw_use_fused': data['adamw_use_fused'],
                    'adamw_warmup_steps': data['adamw_warmup_steps']
                },
                'muon': {
                    'use_muon': data['use_muon'],
                    'muon_min_lr': data['muon_min_lr'],
                    'muon_max_lr': data['muon_max_lr'],
                    'muon_weight_decay': data['muon_weight_decay'],
                    'muon_momentum': data['muon_momentum'],
                    'muon_warmup_steps': data['muon_warmup_steps']
                }
            },
            'training_config': {
                'seed': data['seed'],
                'training_stage': data['training_stage'].value,
                'device_type': data['device_type'].value,
                'training_precision': data['training_precision'].value,
                'total_batch_size': data['total_batch_size'],
                'max_steps': data['max_steps'],
                'early_stopping_patience': data['early_stopping_patience'],
                'early_stopping_patience_skip_steps': data['early_stopping_patience_skip_steps']
            },
            'validation_config': {
                'validate_every_x_steps': data['validate_every_x_steps'],
                'validation_steps': data['validation_steps']
            },
            'eval_config': {
                'hellaswag_every_x_steps': data['hellaswag_every_x_steps'],
                'hellaswag_number_of_examples': data['hellaswag_number_of_examples'],
                'winogrande_every_x_steps': data['winogrande_every_x_steps'],
                'winogrande_number_of_examples': data['winogrande_number_of_examples']
            },
            'generation_config': {
                'generate_every_x_steps': data['generate_every_x_steps'],
                'max_test_gen_len': data['max_test_gen_len']
            },
            'derived': {
                'is_pretraining': data['is_pretraining'],
                'is_instruct_training': data['is_instruct_training'],
                'is_dpo_training': data['is_dpo_training'],
                'dataloader_root_path': data['dataloader_root_path'],
                'save_checkpoints_path': data['save_checkpoints_path']
            }
        }
        if include_model_config is True:
            summary['model_config'] = {
                'dim': data['dim'],
                'n_layers': data['n_layers'],
                'n_heads': data['n_heads'],
                'n_kv_heads': data['n_kv_heads'],
                'multiple_of': data['multiple_of'],
                'ffn_dim_multiplier': data['ffn_dim_multiplier'],
                'norm_eps': data['norm_eps'],
                'is_rope_cis': data['is_rope_cis'],
                'rope_theta': data['rope_theta'],
                'max_batch_size': data['max_batch_size'],
                'max_seq_len': data['max_seq_len'],
                'is_moe': data['is_moe'],
                'moe_num_experts': data['moe_num_experts'],
                'moe_expert_dim': data['moe_expert_dim'],
                'moe_top_k': data['moe_top_k'],
                'moe_load_balancing_coef': data['moe_load_balancing_coef'],
                'moe_z_loss_coef': data['moe_z_loss_coef'],
                'moe_compute_stats': data['moe_compute_stats']
            }
        return summary

    model_config = ConfigDict(
        env_file = '.env',
        extra = 'ignore'
    )

config = TrainConfig()
