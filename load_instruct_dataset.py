import os
import numpy as np
import json

from datasets import load_dataset, interleave_datasets
from tokenizer import init_tokenizer
from config import config


########## SUPPORTED DATASETS ##########
SUPPORTED_HF_DATASETS = [
    'HuggingFaceH4/ultrachat_200k'
]
########################################

############# ADAPTERS #################
def adapt_ultrachat_200k(doc):
    messages = doc['messages']
    conversation = []
    for message in messages:
        conversation.append({'role': message['role'], 'content': message['content']})
    return conversation

ADAPTERS_MAP = {
    'HuggingFaceH4/ultrachat_200k': adapt_ultrachat_200k
}
########################################

tokenizer = init_tokenizer(config.tokenizer_checkpoint_path, config.huggingface_tokenizer)

NUMBER_OF_PROCESSES = max(1, os.cpu_count() // 2)
if config.number_of_cpu_processes != 0:
    NUMBER_OF_PROCESSES = max(1, min(config.number_of_cpu_processes, os.cpu_count()))
print(f'Number of CPU processes: {NUMBER_OF_PROCESSES}\n')

datasets_mix = json.load(open(config.instruct_dataset_mix_file))

assert 'datasets' in datasets_mix
assert 'seed' in datasets_mix

datasets, seed = datasets_mix['datasets'], datasets_mix['seed']

# Validate candidates
valid_datasets = []
for dataset in datasets:
    assert 'id' in dataset
    assert dataset['id'] in SUPPORTED_HF_DATASETS, dataset['id']
    assert 'split' in dataset
    assert 'weight' in dataset
    assert 0.0 <= float(dataset['weight']) <= 1.0

    # Get the ones with weight > 0, assume 100% default and include. Only ignore if 0.0 is set
    weight = dataset.get('weight', 1.0)
    if weight > 0:
        valid_datasets.append(dataset)

assert valid_datasets, 'No datasets with weight > 0'

def ensure_user_first(conversation):
    if not conversation:
        return conversation
    if conversation[0]['role'] != 'user':
        # add default empty user conversation if it is missing (If it starts with assistant)
        return [{'role': 'user', 'content': ''}] + conversation
    return conversation

def has_assistant_content(conversation):
    return any(message['role'] == 'assistant' and message['content'].strip() for message in conversation)

def tokenize(doc):
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
    push(tokenizer.encode('\n' + config.system_prompt), False)
    push([eot], False)

    for interaction in doc['conversation']:
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

probabilities = [float(ds['weight']) for ds in valid_datasets]

# normalize probabilities
total_p = sum(probabilities)
assert total_p > 0
probabilities = [p / total_p for p in probabilities]

print('Mixture probabilities:', {ds['id']: round(p, 3) for ds, p in zip(valid_datasets, probabilities)}, '\n')

prepared_datasets = []
for dataset in valid_datasets:
    ds_id = dataset['id']
    name = dataset.get('name', None)
    split = dataset['split']
    transforms = dataset.get('transforms', {})
    max_turns = transforms.get('max_turns', 8)

    ds = load_dataset(
        ds_id,
        name=name,
        split=split,
        cache_dir='./cache',
        num_proc=NUMBER_OF_PROCESSES
    )
    ds = ds.shuffle(seed=seed)

    adapter = ADAPTERS_MAP.get(ds_id)

    def normalize(doc):
        conversation = adapter(doc)
        conversation = ensure_user_first(conversation)

        if not has_assistant_content(conversation):
            return {'conversation': [], 'source': ds_id}

        conversation = conversation[:max_turns]
        return {'conversation': conversation, 'source': ds_id}

    ds = ds.map(normalize, num_proc=NUMBER_OF_PROCESSES)
    ds = ds.filter(lambda x: len(x['conversation']) > 0, num_proc=NUMBER_OF_PROCESSES)

    columns_to_remove = [c for c in ds.column_names if c not in ['source']]
    tokenized_ds = ds.map(tokenize, num_proc=NUMBER_OF_PROCESSES, remove_columns=columns_to_remove)

    prepared_datasets.append(tokenized_ds)

mixed_datasets = interleave_datasets(prepared_datasets, probabilities=probabilities, seed=seed)

train_val = mixed_datasets.train_test_split(test_size=0.01, seed=seed)
train_val['train'].save_to_disk(os.path.join(config.instruct_dataset_target_path, 'train'))
train_val['test'] .save_to_disk(os.path.join(config.instruct_dataset_target_path, 'val'))
