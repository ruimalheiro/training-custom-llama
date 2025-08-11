import os
import numpy as np
import json

from datasets import load_dataset, interleave_datasets
from tokenizer import init_tokenizer
from data_preparation_utils import prepare_dataset
from config import config


#### SUPPORTED DATASETS
SUPPORTED_HF_DATASETS = [
    'HuggingFaceFW/fineweb-edu'
]

#### ADAPTERS
def adapt_fineweb_edu(doc, transforms):
    text = doc['text']
    return text

ADAPTERS_MAP = {
    'HuggingFaceFW/fineweb-edu': adapt_fineweb_edu
}

#### PREPARATION
tokenizer = init_tokenizer(config.tokenizer_checkpoint_path, config.huggingface_tokenizer)

NUMBER_OF_PROCESSES = max(1, os.cpu_count() // 2)
if config.number_of_cpu_processes != 0:
    NUMBER_OF_PROCESSES = max(1, min(config.number_of_cpu_processes, os.cpu_count()))
print(f'Number of CPU processes: {NUMBER_OF_PROCESSES}\n')

datasets_mix = json.load(open(config.pretrain_dataset_mix_file))

assert 'datasets' in datasets_mix
assert 'seed' in datasets_mix
assert 'shard_size' in datasets_mix

datasets, seed, shard_size = datasets_mix['datasets'], datasets_mix['seed'], datasets_mix['shard_size']

assert isinstance(seed, int)
assert isinstance(shard_size, int)

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

probabilities = [float(ds['weight']) for ds in valid_datasets]

# normalize probabilities
total_p = sum(probabilities)
assert total_p > 0
probabilities = [p / total_p for p in probabilities]

print('Mixture probabilities:', {ds['id']: round(p, 3) for ds, p in zip(valid_datasets, probabilities)}, '\n')

def tokenize(doc):
    tokens = [tokenizer.eos_id]
    tokens.extend(tokenizer.encode(doc['text']))
    tokens_np = np.asarray(tokens, dtype=np.uint32)
    return { 'input_ids': tokens_np }

prepared_datasets = []
for dataset in valid_datasets:
    ds_id = dataset['id']
    name = dataset.get('name', None)
    split = dataset['split']
    transforms = dataset.get('transforms', {})

    shuffle = transforms.get('shuffle', False)
    max_datapoints = transforms.get('max_datapoints', None)

    ds = load_dataset(
        ds_id,
        name=name,
        split=split,
        cache_dir='./cache',
        num_proc=NUMBER_OF_PROCESSES
    )

    if shuffle:
        ds = ds.shuffle(seed=seed)

    if max_datapoints:
        max_datapoints = int(max_datapoints)
        ds = ds.select(range(max_datapoints))

    adapter = ADAPTERS_MAP.get(ds_id)

    def normalize(doc):
        text = adapter(doc, transforms)
        return {'text': text, 'source': ds_id}

    ds = ds.map(normalize, num_proc=NUMBER_OF_PROCESSES)
    ds = ds.filter(lambda x: len(x['text']) > 0, num_proc=NUMBER_OF_PROCESSES)

    columns_to_remove = [c for c in ds.column_names if c not in ['source']]
    tokenized_ds = ds.map(tokenize, num_proc=NUMBER_OF_PROCESSES, remove_columns=columns_to_remove)

    prepared_datasets.append(tokenized_ds)

mixed_datasets = interleave_datasets(prepared_datasets, probabilities=probabilities, seed=seed)

print('Summary:')
for d, ds in zip(valid_datasets, prepared_datasets):
    print(f'- Total for: {d["id"]} : {len(ds)}')

print('- Mix total len:', len(mixed_datasets))

splits = mixed_datasets.train_test_split(test_size=0.01, seed=seed)

print('- Train len:', len(splits['train']), ' Val len:', len(splits['test']), '\n')

train_ds = splits['train'].with_format('numpy', columns=['input_ids'])
val_ds   = splits['test'] .with_format('numpy', columns=['input_ids'])

prepare_dataset(
    dataset=train_ds,
    target_folder=os.path.join(config.pretrain_dataset_target_path, 'train'),
    shard_file_prefix='data',
    shard_size=shard_size
)

prepare_dataset(
    dataset=val_ds,
    target_folder=os.path.join(config.pretrain_dataset_target_path, 'val'),
    shard_file_prefix='data',
    shard_size=shard_size
)

print('\nData preparation completed.')
