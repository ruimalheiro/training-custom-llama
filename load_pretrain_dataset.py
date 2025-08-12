import os
import numpy as np
import json

from tokenizer import init_tokenizer
from config import config
from datasets import (
    load_dataset,
    interleave_datasets
)
from data_preparation_utils import (
    prepare_dataset,
    get_max_number_of_cpu_processes,
    assert_common_structure_and_extract
)


NUMBER_OF_PROCESSES = get_max_number_of_cpu_processes()

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

#### VERIFY MIX FILE STRUCTURE
datasets_mix = json.load(open(config.pretrain_dataset_mix_file))

assert 'shard_size' in datasets_mix
shard_size = datasets_mix['shard_size']
assert isinstance(shard_size, int)

seed, valid_datasets, probabilities = assert_common_structure_and_extract(datasets_mix, SUPPORTED_HF_DATASETS)

#### PREPARATION
tokenizer = init_tokenizer(config.tokenizer_checkpoint_path, config.huggingface_tokenizer)

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
        num_proc=NUMBER_OF_PROCESSES,
        token=config.hf_token
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
