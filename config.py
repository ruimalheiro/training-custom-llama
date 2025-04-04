from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings

class TrainConfig(BaseSettings):
    # datasets_path
    pretrain_dataset: str = Field(alias='HF_PRETRAIN_DATASET')
    pretrain_dataset_name: str = Field(default='default', alias='HF_PRETRAIN_DATASET_NAME')
    pretrain_dataset_split: str = Field(default='train', alias='HF_PRETRAIN_DATASET_SPLIT')
    pretrain_dataset_target_path: str = Field(alias='HF_PRETRAIN_DATASET_TARGET_PATH')
    pretrain_dataset_shard_prefix: str = Field(alias='HF_PRETRAIN_DATASET_SHARD_PREFIX')

    instruct_dataset: str = Field(alias='HF_INSTRUCT_DATASET')
    instruct_dataset_name: str = Field(default='default', alias='HF_INSTRUCT_DATASET_NAME')
    instruct_dataset_split: str = Field(default='train', alias='HF_INSTRUCT_DATASET_SPLIT')
    instruct_dataset_target_path: str = Field(alias='HF_INSTRUCT_DATASET_TARGET_PATH')
    instruct_dataset_shard_prefix: str = Field(alias='HF_INSTRUCT_DATASET_SHARD_PREFIX')

    number_of_cpu_processes: int = Field(default=0, alias='NUMBER_OF_CPU_PROCESSES')

    dataloader_root_path: str = Field(alias='DATALOADER_ROOT_PATH')
    hellaswag_path: str = Field(alias='HELLASWAG_PATH')

    # save / load path
    load_checkpoints_path: str = Field(alias='LOAD_CHECKPOINTS_PATH')
    save_checkpoints_path: str = Field(alias='SAVE_CHECKPOINTS_PATH')
    save_checkpoints: bool = Field(default=False, alias='SAVE_CHECKPOINTS')

    # wnb
    wnb_enabled: bool = Field(default=False, alias='WNB_ENABLED')
    wnb_project_name: str = Field(alias='WNB_PROJECT_NAME')

    # tokenizer model path
    tokenizer_checkpoint_path: str = Field(alias='TOKENIZER_CHECKPOINT_PATH')

    # train config
    total_batch_size: int = Field(alias='TOTAL_BATCH_SIZE')
    max_lr: float = Field(alias='MAX_LR')
    min_lr: float = Field(alias='MIN_LR')
    warmup_steps: int = Field(alias='WARMUP_STEPS')
    weight_decay: float = Field(alias='WEIGHT_DECAY')
    max_steps: int = Field(default=-1, alias='MAX_STEPS') # If not set, it is aprox calculated
    early_stopping_patience: int = Field(alias='EARLY_STOPPING_PATIENCE')
    is_instruct_training: bool = Field(default=False, alias='IS_INSTRUCT_TRAINING')
    is_model_distillation: bool = Field(alias='IS_MODEL_DISTILLATION')
    distillation_temperature: float = Field(alias='DISTILLATION_TEMPERATURE')
    # The teacher model is loader via huggingface API: AutoModelForCausalLM.from_pretrained(teacher_model_checkpoint, ...) so needs to ve a valid checkpoint.
    teacher_model_checkpoint: str = Field(alias='TEACHER_MODEL_CHECKPOINT')

    # validation
    validate_every_x_steps: int = Field(alias='VALIDATE_EVERY_X_STEPS')
    val_steps: int = Field(alias='VAL_STEPS')
    hellaswag_every_x_steps: int = Field(alias='HELLASWAG_EVERY_X_STEPS')
    hellagswag_number_of_examples: int = Field(alias='HELLASWAG_NUMBER_OF_EXAMPLES')
    generate_every_x_steps: int = Field(alias='GENERATE_EVERY_X_STEPS')
    max_test_gen_len: int = Field(alias='MAX_TEST_GEN_LEN')

    # test prompts
    test_pretrain_generation_prompts: list[str] = Field(alias='TEST_PRETRAIN_GENERATION_PROMPTS')
    test_instruct_generation_prompts: list[str] = Field(alias='TEST_INSTRUCT_GENERATION_PROMPTS')

    # model architecture config
    dim: int = Field(default=768, alias='DIM')
    n_layers: int = Field(default=16, alias='N_LAYERS')
    n_heads: int = Field(default=16, alias='N_HEADS')
    n_kv_heads: int = Field(default=8, alias='N_KV_HEADS')
    vocab_size: int = Field(default=128256, alias='VOCAB_SIZE')
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
