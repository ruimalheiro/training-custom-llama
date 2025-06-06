import os
import numpy as np

from datasets import load_dataset
from tokenizer import init_tokenizer
from data_preparation_utils import prepare_dataset
from config import config


tokenizer = init_tokenizer(config.tokenizer_checkpoint_path, config.huggingface_tokenizer)

NUMBER_OF_PROCESSES = max(1, os.cpu_count() // 2)
if config.number_of_cpu_processes != 0:
    NUMBER_OF_PROCESSES = max(1, min(config.number_of_cpu_processes, os.cpu_count()))
print(f'Number of CPU processes: {NUMBER_OF_PROCESSES}\n')

dataset = load_dataset(
    config.pretrain_dataset,
    name=config.pretrain_dataset_name,
    split=config.pretrain_dataset_split,
    cache_dir='./cache',
    num_proc=NUMBER_OF_PROCESSES
)

def tokenize(doc):
    tokens = [tokenizer.eos_id]
    tokens.extend(tokenizer.encode(doc['text']))
    tokens_np = np.array(tokens)
    return tokens_np

prepare_dataset(
    dataset=dataset,
    tokenize_function=tokenize,
    target_folder=config.pretrain_dataset_target_path,
    shard_file_prefix=config.pretrain_dataset_shard_prefix,
    number_of_processes=NUMBER_OF_PROCESSES,
    chunksize=16
)

