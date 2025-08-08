from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings
from enum import Enum

class TrainingStage(str, Enum):
    PRETRAIN = 'pretrain'
    INSTRUCT = 'instruct'
    DPO = 'dpo'

class TrainConfig(BaseSettings):
    # datasets_path
    pretrain_dataset: str = Field(alias='HF_PRETRAIN_DATASET')
    pretrain_dataset_name: str = Field(default='default', alias='HF_PRETRAIN_DATASET_NAME')
    pretrain_dataset_split: str = Field(default='train', alias='HF_PRETRAIN_DATASET_SPLIT')
    pretrain_dataset_target_path: str = Field(alias='HF_PRETRAIN_DATASET_TARGET_PATH')
    pretrain_dataset_shard_prefix: str = Field(alias='HF_PRETRAIN_DATASET_SHARD_PREFIX')

    instruct_dataset_mix_file: str | None = Field(default=None, alias="HF_INSTRUCT_DATASET_MIX_FILE")
    instruct_dataset_target_path: str = Field(alias='HF_INSTRUCT_DATASET_TARGET_PATH')

    dpo_dataset: str = Field(alias='HF_DPO_DATASET')
    dpo_dataset_name: str = Field(default='default', alias='HF_DPO_DATASET_NAME')
    dpo_dataset_split: str = Field(default='train', alias='HF_DPO_DATASET_SPLIT')
    dpo_dataset_target_path: str = Field(alias='HF_DPO_DATASET_TARGET_PATH')
    dpo_dataset_shard_prefix: str = Field(alias='HF_DPO_DATASET_SHARD_PREFIX')

    number_of_cpu_processes: int = Field(default=0, alias='NUMBER_OF_CPU_PROCESSES')

    pretrain_dataloader_root_path: str = Field(alias='PRETRAIN_DATALOADER_ROOT_PATH')
    instruct_dataloader_root_path: str = Field(alias='INSTRUCT_DATALOADER_ROOT_PATH')
    dpo_dataloader_root_path: str = Field(alias='DPO_DATALOADER_ROOT_PATH')
    hellaswag_path: str = Field(alias='HELLASWAG_PATH')

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

    # wnb
    wnb_enabled: bool = Field(default=False, alias='WNB_ENABLED')
    wnb_project_name: str = Field(alias='WNB_PROJECT_NAME')

    # tokenizer model path
    tokenizer_checkpoint_path: str = Field(alias='TOKENIZER_CHECKPOINT_PATH')
    huggingface_tokenizer: bool = Field(default=False, alias='HUGGINGFACE_TOKENIZER')

    # train config
    training_stage: TrainingStage = Field(default=TrainingStage.PRETRAIN, alias='TRAINING_STAGE')
    total_batch_size: int = Field(alias='TOTAL_BATCH_SIZE')
    max_lr: float = Field(alias='MAX_LR')
    min_lr: float = Field(alias='MIN_LR')
    warmup_steps: int = Field(alias='WARMUP_STEPS')
    weight_decay: float = Field(alias='WEIGHT_DECAY')
    max_steps: int = Field(default=-1, alias='MAX_STEPS') # If not set, it is aprox calculated
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

    # validation
    validate_every_x_steps: int = Field(alias='VALIDATE_EVERY_X_STEPS')
    val_steps: int = Field(alias='VAL_STEPS')
    hellaswag_every_x_steps: int = Field(alias='HELLASWAG_EVERY_X_STEPS')
    hellaswag_number_of_examples: int = Field(alias='HELLASWAG_NUMBER_OF_EXAMPLES')
    generate_every_x_steps: int = Field(alias='GENERATE_EVERY_X_STEPS')
    max_test_gen_len: int = Field(alias='MAX_TEST_GEN_LEN')

    # test prompts
    test_prompts_path: str = Field(alias='TEST_PROMPTS_FILE')

    # model architecture config
    dim: int = Field(default=768, alias='DIM')
    n_layers: int = Field(default=16, alias='N_LAYERS')
    n_heads: int = Field(default=16, alias='N_HEADS')
    n_kv_heads: int = Field(default=8, alias='N_KV_HEADS')
    multiple_of: int = Field(default=1024, alias='MULTIPLE_OF')
    ffn_dim_multiplier: float = Field(default=1.0, alias='FFN_DIM_MULTIPLIER')
    norm_eps: float = Field(default=1e-05, alias='NORM_EPS')
    rope_theta: float = Field(default=500000.0, alias='ROPE_THETA')
    max_batch_size: int = Field(default=4, alias='MAX_BATCH_SIZE')
    max_seq_len: int = Field(default=1024, alias='MAX_SEQ_LEN')

    model_config = ConfigDict(
        env_file = '.env',
        extra = 'ignore'
    )

config = TrainConfig()
