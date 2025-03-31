import os
import numpy as np

from datasets import load_dataset
from tokenizer import Tokenizer
from data_preparation_utils import prepare_dataset
from config import config


NUMBER_OF_PROCESSES = max(1, os.cpu_count() // 2)
if config.number_of_cpu_processes != 0:
    NUMBER_OF_PROCESSES = max(1, config.number_of_cpu_processes)
print(f'Number of CPU processes: {NUMBER_OF_PROCESSES}\n')

dataset = load_dataset(
    config.pretrain_dataset,
    name=config.pretrain_dataset_name,
    split=config.pretrain_dataset_split,
    cache_dir='./cache',
    num_proc=NUMBER_OF_PROCESSES
)

tokenizer = Tokenizer('./tokenizer.model')

def tokenize(doc):
    eot = tokenizer.special_tokens['<|end_of_text|>']
    tokens = [eot]
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

