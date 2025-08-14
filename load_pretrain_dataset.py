import os
import numpy as np
import json

from config import config
os.environ['HF_HOME'] = config.hf_home
os.environ['HF_DATASETS_CACHE'] = f'{config.hf_home}/datasets'
os.environ['HF_HUB_CACHE'] = f'{config.hf_home}/hub'

from tokenizer import init_tokenizer
from datasets import (
    load_dataset,
    interleave_datasets,
    Features,
    Sequence,
    Value
)
from data_preparation_utils import (
    prepare_dataset,
    get_max_number_of_cpu_processes,
    assert_common_structure_and_extract
)


NUMBER_OF_PROCESSES = get_max_number_of_cpu_processes()

#### SUPPORTED DATASETS
SUPPORTED_HF_DATASETS = [
    'HuggingFaceFW/fineweb-edu',
    'HuggingFaceTB/smollm-corpus'
]

#### ADAPTERS
def adapt_fineweb_edu(batch, transforms):
    text = batch['text']
    return {'text': batch['text']}

def adapt_smollm_corpus_fineweb_edu_dedup(batch, transforms):
    text = batch['text']
    return {'text': batch['text']}

ADAPTERS_MAP = {
    'HuggingFaceFW/fineweb-edu_sample-10BT': adapt_fineweb_edu,
    'HuggingFaceFW/fineweb-edu_sample-100BT': adapt_fineweb_edu,
    'HuggingFaceTB/smollm-corpus_fineweb-edu-dedup': adapt_smollm_corpus_fineweb_edu_dedup
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
    input_ids = tokenizer.encode(doc['text'])
    tokens_np = np.empty(len(input_ids) + 1, dtype=np.uint32)
    tokens_np[0] = tokenizer.eos_id
    tokens_np[1:] = input_ids
    return tokens_np

prepared_datasets = []
for dataset in valid_datasets:
    ds_id = dataset['id']
    name = dataset.get('name', None)

    adapter_id = f'{ds_id}_{name}' if name and name != 'default' else ds_id

    split = dataset['split']
    transforms = dataset.get('transforms', {})

    shuffle = transforms.get('shuffle', False)
    max_datapoints = transforms.get('max_datapoints', None)

    if max_datapoints:
        max_datapoints = int(max_datapoints)
        assert isinstance(max_datapoints, int)
        assert max_datapoints > 0
        split = f'{split}[:{max_datapoints}]'

    ds = load_dataset(
        ds_id,
        name=name,
        split=split,
        num_proc=NUMBER_OF_PROCESSES,
        token=config.hf_token
    )

    columns_to_remove = ds.column_names

    if shuffle:
        ds = ds.shuffle(seed=seed)

    adapter = ADAPTERS_MAP.get(adapter_id)
    if adapter is None:
        raise ValueError(f'No adapter for {adapter_id}')

    def normalize_and_tokenize(batch):
        batch = adapter(batch, transforms)
        texts = batch['text']
        if config.hf_include_source_id:
            return {'text': texts, 'source': [ds_id] * len(texts)}
        return {'text': texts}

    ds = ds.map(
        normalize_and_tokenize,
        batched=True,
        batch_size=config.hf_map_batch_size,
        num_proc=NUMBER_OF_PROCESSES,
        writer_batch_size=config.hf_map_writer_batch_size,
        remove_columns=columns_to_remove
    )

    prepared_datasets.append(ds)

if len(prepared_datasets) > 1:
    print('Interleaving datasets... This operation can take a few minutes...')
    prepared_dataset = interleave_datasets(prepared_datasets, probabilities=probabilities, seed=seed)
    print('Interleaving datasets complete')
    print('- Mix total len:', len(prepared_dataset), '\n')
else:
    prepared_dataset = prepared_datasets[0]

print('Summary:')
for d, ds in zip(valid_datasets, prepared_datasets):
    print(f'- Total for: {d["id"]} : {len(ds)}')


splits = prepared_dataset.train_test_split(test_size=0.01, seed=seed)

print('- Train len:', len(splits['train']), ' Val len:', len(splits['test']), '\n')

train_ds = splits['train'].with_format('python', columns=['text'])
val_ds   = splits['test'] .with_format('python', columns=['text'])

# Tokenization is being done inside the prepare_dataset. This is because tokenizing with datasets map seems to require a full rewrite of
# the arrow dataset which can take a lot of time. At this point we just need to tokenize and write the shard (not recreate the arrow dataset).
print('Preparing train dataset...')
prepare_dataset(
    dataset=train_ds,
    tokenize_function=tokenize,
    target_folder=os.path.join(config.pretrain_dataset_target_path, 'train'),
    shard_file_prefix='data',
    shard_size=shard_size,
    number_of_processes=NUMBER_OF_PROCESSES,
    chunksize=config.mp_pool_chunk_size
)

print('Preparing val dataset...')
prepare_dataset(
    dataset=val_ds,
    tokenize_function=tokenize,
    target_folder=os.path.join(config.pretrain_dataset_target_path, 'val'),
    shard_file_prefix='data',
    shard_size=shard_size,
    number_of_processes=NUMBER_OF_PROCESSES,
    chunksize=config.mp_pool_chunk_size
)

print('\nData preparation completed.')
