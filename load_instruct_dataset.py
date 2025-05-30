import os
import numpy as np

from datasets import load_dataset
from tokenizer import init_tokenizer
from config import config


tokenizer = init_tokenizer(config.tokenizer_checkpoint_path, config.huggingface_tokenizer)

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

def tokenize(doc):
    # Many datasets follow the same content (messages / role) structure. The only difference is the root key property.
    selector_key = config.instruct_dataset_selector_key

    assert doc[selector_key][0]['role'] == 'user'

    bot = tokenizer.bos_id
    sh = tokenizer.sh_id
    eh = tokenizer.eh_id
    eot = tokenizer.eot_id

    tokens, labels = [], []
    def push(tok_ids, is_assistant):
        tokens.extend(tok_ids)
        if is_assistant:
            labels.extend(tok_ids)
        else:
            labels.extend([-100] * len(tok_ids))

    push([bot], False)
    push([sh], False)
    push(tokenizer.encode('system'), False)
    push([eh], False)
    push(tokenizer.encode('\n' + 'You are a helpful AI assistant'), False)
    push([eot], False)

    for interaction in doc[selector_key]:
        role = interaction['role']
        content = interaction['content']

        push([sh], False)
        push(tokenizer.encode(role), False)
        push([eh], False)
        push(tokenizer.encode('\n'), False)
        push(tokenizer.encode(content), role == 'assistant')
        push([eot], False)

    input_ids = np.array(tokens, dtype=np.uint32)
    labels = np.array(labels[1:] + [-100], dtype=np.int32)

    return { 'input_ids': input_ids, 'labels': labels }

dataset = dataset.map(
    tokenize,
    num_proc=NUMBER_OF_PROCESSES,
    remove_columns=dataset.column_names
)

train_val = dataset.train_test_split(test_size=0.01, seed=42)
train_val['train'].save_to_disk(os.path.join(config.instruct_dataset_target_path, 'train'))
train_val['test'] .save_to_disk(os.path.join(config.instruct_dataset_target_path, 'val'))
