import os
import numpy as np

from datasets import load_dataset
from tokenizer import Tokenizer
from data_preparation_utils import prepare_dataset
from config import config


NUMBER_OF_PROCESSES = max(1, os.cpu_count() // 2)
if config.number_of_cpu_processes != 0:
    NUMBER_OF_PROCESSES = max(1, min(config.number_of_cpu_processes, os.cpu_count()))
print(f'Number of CPU processes: {NUMBER_OF_PROCESSES}\n')

dataset = load_dataset(
    config.instruct_dataset,
    name=config.instruct_dataset_name,
    split=config.instruct_dataset_split,
    cache_dir='./cache',
    num_proc=NUMBER_OF_PROCESSES
)

tokenizer = Tokenizer('./tokenizer.model')

def tokenize(doc):
    assert doc['conversation'][0]['role'] == 'user'

    bot = tokenizer.special_tokens['<|begin_of_text|>']
    sh = tokenizer.special_tokens['<|start_header_id|>']
    eh = tokenizer.special_tokens['<|end_header_id|>']
    eot = tokenizer.special_tokens['<|eot_id|>']

    tokens = [bot]
    tokens.extend([sh])
    tokens.extend(tokenizer.encode('system'))
    tokens.extend([eh])
    tokens.extend(tokenizer.encode('\n' + 'You are a helpful AI assistant'))
    tokens.extend([eot])

    for interaction in doc['conversation']:
        role = interaction['role']
        content = interaction['content']

        tokens.extend([sh])
        tokens.extend(tokenizer.encode(role))
        tokens.extend([eh])
        tokens.extend(tokenizer.encode('\n' + content))
        tokens.extend([eot])

    tokens_np = np.array(tokens)
    return tokens_np
        
prepare_dataset(
    dataset=dataset,
    tokenize_function=tokenize,
    target_folder=config.instruct_dataset_target_path,
    shard_file_prefix=config.instruct_dataset_shard_prefix,
    number_of_processes=NUMBER_OF_PROCESSES,
    chunksize=16
)
