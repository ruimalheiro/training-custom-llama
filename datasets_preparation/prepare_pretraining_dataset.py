import os
import numpy as np
import time
import copy

from config import config
from tokenizer import init_tokenizer
from datasets import (
    load_dataset,
    interleave_datasets
)
from datasets_preparation.data_preparation_utils import (
    stable_hash,
    prepare_dataset,
    get_max_number_of_cpu_processes,
    assert_common_structure_and_extract
)


NUMBER_OF_PROCESSES = get_max_number_of_cpu_processes()

#### SUPPORTED DATASETS
SUPPORTED_HF_DATASETS = {
    'HuggingFaceFW/fineweb-edu': {
        'sample-10BT': {
            'id': 'HuggingFaceFW/fineweb-edu',
            'split': 'train',
            'adapter': 'HuggingFaceFW/fineweb-edu_sample-10BT'
        },
        'sample-100BT': {
            'id': 'HuggingFaceFW/fineweb-edu',
            'split': 'train',
            'adapter': 'HuggingFaceFW/fineweb-edu_sample-100BT'
        }
    },
    'HuggingFaceTB/smollm-corpus': {
        'fineweb-edu-dedup': {
            'id': 'HuggingFaceTB/smollm-corpus',
            'split': 'train',
            'adapter': 'HuggingFaceTB/smollm-corpus_fineweb-edu-dedup'
        }
    }
}

DEFAULT_MIX = {
    'seed': 42,
    'shard_size': 100_000_000,
    'datasets': {
        'HuggingFaceFW/fineweb-edu': {
            'sample-10BT': {
                'weight': 1.0,
                'transforms': {}
            }
        }
    }
}

#### ADAPTERS
def adapt_fineweb_edu(batch, transforms):
    return {'text': batch['text']}

def adapt_smollm_corpus_fineweb_edu_dedup(batch, transforms):
    return {'text': batch['text']}

ADAPTERS_MAP = {
    'HuggingFaceFW/fineweb-edu_sample-10BT': adapt_fineweb_edu,
    'HuggingFaceFW/fineweb-edu_sample-100BT': adapt_fineweb_edu,
    'HuggingFaceTB/smollm-corpus_fineweb-edu-dedup': adapt_smollm_corpus_fineweb_edu_dedup
}

def download_and_prepare_data(
    *,
    seed,
    valid_datasets,
    probabilities
):
    prepared_datasets = []
    for dataset in valid_datasets:
        ds_id = dataset['id']
        name = dataset.get('name', None)

        dataset_config = SUPPORTED_HF_DATASETS[ds_id][name]
        split = dataset_config['split']
        adapter_id = dataset_config['adapter']

        transforms = dataset.get('transforms', {})

        max_datapoints = transforms.get('max_datapoints', None)

        ds = load_dataset(
            ds_id,
            name=name,
            split=split,
            streaming=True,
            token=config.hf_token
        )

        columns_to_remove = ds.column_names

        if max_datapoints:
            max_datapoints = int(max_datapoints)
            assert isinstance(max_datapoints, int)
            assert max_datapoints > 0
            ds = ds.take(max_datapoints)

        adapter = ADAPTERS_MAP.get(adapter_id)
        if adapter is None:
            raise ValueError(f'No adapter for {adapter_id}')

        def normalize(
            batch,
            *,
            adapter=adapter,
            transforms=transforms,
            ds_id=ds_id,
        ):
            batch = adapter(batch, transforms)
            texts = batch['text']

            if config.hf_include_source_id:
                return {
                    'text': texts,
                    'source': [ds_id] * len(texts),
                }

            return {'text': texts}

        ds = ds.map(
            normalize,
            batched=True,
            batch_size=config.hf_map_batch_size,
            remove_columns=columns_to_remove
        )

        prepared_datasets.append(ds)

    if len(prepared_datasets) > 1:
        print('Preparing Interleaving iterator... This operation can take a few minutes...')
        prepared_dataset = interleave_datasets(
            prepared_datasets,
            probabilities=probabilities,
            seed=seed
        )
        time.sleep(2) # Workaround for occasional streaming/interleave iterator shutdown issue.
        print('Interleaving datasets complete')
    else:
        prepared_dataset = prepared_datasets[0]

    # Split into train / val iterators. (no train_test_split with stream) The idea here is based in the law of large numbers (assuming we have enough datapoints).
    # We want to draw datapoints and approach the proportions (probs) for train / val so. The hash is to make the assignment deterministic.
    HASH_BYTES = 8
    HASH_SPACE = 1 << (HASH_BYTES * 8) # 64 bit

    VAL_FRACTION = 0.01
    SEPARATION_THRESHOLD = int(VAL_FRACTION * HASH_SPACE)

    def is_train(ex):
        return stable_hash(ex['text'], seed=seed, hash_bytes=HASH_BYTES) >= SEPARATION_THRESHOLD

    def is_val(ex):
        return not is_train(ex)

    train_ds = prepared_dataset.filter(is_train)
    val_ds = prepared_dataset.filter(is_val)

    return train_ds, val_ds

tokenizer = None
def tokenize(doc):
    global tokenizer
    if tokenizer is None:
        tokenizer = init_tokenizer(config.tokenizer_checkpoint_path, config.huggingface_tokenizer)
    input_ids = tokenizer.encode(doc['text'])
    tokens_np = np.empty(len(input_ids) + 1, dtype=np.uint32)
    tokens_np[0] = tokenizer.eos_id
    tokens_np[1:] = input_ids
    return tokens_np

def shard_and_tokenize(
    *,
    shard_size,
    train_ds,
    val_ds
):
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

def prepare_pretraining_dataset(
    *,
    datasets_mix
):
    datasets_mix = copy.deepcopy(datasets_mix) if datasets_mix else copy.deepcopy(DEFAULT_MIX)

    assert 'shard_size' in datasets_mix
    shard_size = datasets_mix['shard_size']
    assert isinstance(shard_size, int)

    #### VERIFY MIX FILE STRUCTURE
    seed, valid_datasets, probabilities = assert_common_structure_and_extract(datasets_mix, SUPPORTED_HF_DATASETS)

    train_ds, val_ds = download_and_prepare_data(
        seed=seed,
        valid_datasets=valid_datasets,
        probabilities=probabilities
    )

    shard_and_tokenize(
        shard_size=shard_size,
        train_ds=train_ds,
        val_ds=val_ds
    )

    print('\nData preparation completed.')
