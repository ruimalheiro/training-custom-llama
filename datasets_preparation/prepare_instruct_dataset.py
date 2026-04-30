import os
import numpy as np
import re
import random
import copy
import time

from config import config
from tokenizer import init_tokenizer
from datasets import (
    load_dataset,
    interleave_datasets
)
from datasets_preparation.data_preparation_utils import (
    stable_hash,
    get_max_number_of_cpu_processes,
    assert_common_structure_and_extract
)
from datasets_preparation.default_mixes import DEFAULT_INSTRUCT_MIX


#### ADAPTERS
def adapt_ultrachat_200k(doc, transforms, seed):
    messages = doc['messages']
    conversation = []
    for message in messages:
        conversation.append({'role': message['role'], 'content': message['content']})
    return conversation

def adapt_lmsys_chat_1m(doc, transforms, seed):
    messages = doc['conversation']
    replace = False
    name_mapping = None

    if transforms.get('placeholders') and transforms['placeholders'].get('replace', False):
        # The replacements in the config file are a suggestion of more neutral names.
        replace = True
        assert 'random_options' in transforms['placeholders']
        assert len(transforms['placeholders']['random_options']) > 0, 'List of replacements("random_options") cannot be empty'
        replacements = transforms['placeholders']['random_options']

        full_conversation = ' '.join(message['content'] for message in messages)
        name_ids = sorted(set(re.compile(r'\bNAME_(\d+)\b').findall(full_conversation)))

        # Local random generator because of the multi processing
        local_rng = random.Random(seed ^ stable_hash(full_conversation))

        # Selected names. + 1 here because when len(name_ids) < len(replacements)
        name_pool = (replacements * ((len(name_ids) // len(replacements)) + 1))[:len(name_ids)]

        local_rng.shuffle(name_pool)

        # mapping 1 (NAME_1) -> Sam
        name_mapping = {f'NAME_{name_id}': name for name_id, name in zip(name_ids, name_pool)}

    conversation = []
    for message in messages:
        content = message['content']
        if replace:
            for name_id, name in name_mapping.items():
                content = content.replace(name_id, name)
        conversation.append({'role': message['role'], 'content': content})
    return conversation

#### SUPPORTED DATASETS
SUPPORTED_HF_DATASETS = {
    'HuggingFaceH4/ultrachat_200k': {
        'default': {
            'id': 'HuggingFaceH4/ultrachat_200k',
            'split': 'train_sft',
            'adapter': adapt_ultrachat_200k
        }
    },
    'lmsys/lmsys-chat-1m': {
        'default': {
            'id': 'lmsys/lmsys-chat-1m',
            'split': 'train',
            'adapter': adapt_lmsys_chat_1m
        }
    }
}

def ensure_user_first(conversation):
    if not conversation:
        return conversation
    if conversation[0]['role'] != 'user':
        # add default empty user conversation if it is missing (If it starts with assistant)
        return [{'role': 'user', 'content': ''}] + conversation
    return conversation

def has_assistant_content(conversation):
    return any(message['role'] == 'assistant' and message['content'].strip() for message in conversation)

tokenizer = None
SYS = None
SYS_PROMPT = None
NL = None

def tokenize(doc):
    global tokenizer, SYS, SYS_PROMPT, NL
    if tokenizer is None:
        tokenizer = init_tokenizer(config.tokenizer_checkpoint_path, config.huggingface_tokenizer)
        SYS = tokenizer.encode('system')
        SYS_PROMPT = tokenizer.encode('\n' + config.system_prompt)
        NL = tokenizer.encode('\n')

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
            labels.extend([config.ignore_index] * len(tok_ids))

    push([bot], False)
    push([sh], False)
    push(SYS, False)
    push([eh], False)
    push(SYS_PROMPT, False)
    push([eot], False)

    for interaction in doc['conversation']:
        role = interaction['role']
        content = interaction['content']

        push([sh], False)
        push(tokenizer.encode(role), False)
        push([eh], False)
        push(NL, False)
        push(tokenizer.encode(content), role == 'assistant')
        push([eot], role == 'assistant')

    input_ids = np.array(tokens, dtype=np.uint32)
    labels = np.array(labels[1:] + [config.ignore_index], dtype=np.int32)

    return { 'input_ids': input_ids, 'labels': labels }

def download_and_prepare_data(
    *,
    seed,
    valid_datasets,
    probabilities,
    number_of_processes
):
    prepared_datasets = []
    for dataset in valid_datasets:
        ds_id = dataset['id']
        name = dataset.get('name', None)

        dataset_config = SUPPORTED_HF_DATASETS[ds_id][name]
        split = dataset_config['split']
        adapter = dataset_config['adapter']

        transforms = dataset.get('transforms', {})

        max_datapoints = transforms.get('max_datapoints', None)
        max_turns = transforms.get('max_turns', 8)

        hf_name = None if name == 'default' else name

        ds = load_dataset(
            ds_id,
            name=hf_name,
            split=split,
            num_proc=number_of_processes,
            token=config.hf_token
        )

        if max_datapoints:
            max_datapoints = int(max_datapoints)
            ds = ds.select(range(max_datapoints))

        def normalize(doc):
            conversation = adapter(doc, transforms, seed)
            conversation = ensure_user_first(conversation)
            conversation = conversation[:max_turns]

            result = {'conversation': []}
            if config.hf_include_source_id:
                result['source'] = ds_id

            if not has_assistant_content(conversation):
                return result

            result['conversation'] = conversation
            return result

        ds = ds.map(normalize, num_proc=number_of_processes)
        ds = ds.filter(lambda x: len(x['conversation']) > 0, num_proc=number_of_processes)

        columns_to_remove = [c for c in ds.column_names if c not in ['source']]
        tokenized_ds = ds.map(tokenize, num_proc=number_of_processes, remove_columns=columns_to_remove)

        prepared_datasets.append(tokenized_ds)

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

    print('Summary:')
    for d, ds in zip(valid_datasets, prepared_datasets):
        print(f'- Total for: {d["id"]} : {len(ds)}')

    print('- Mix total len:', len(prepared_dataset))

    splits = prepared_dataset.train_test_split(test_size=0.01, seed=seed)

    print('- Train len:', len(splits['train']), ' Val len:', len(splits['test']), '\n')

    splits['train'].save_to_disk(os.path.join(config.instruct_dataset_target_path, 'train'))
    splits['test'] .save_to_disk(os.path.join(config.instruct_dataset_target_path, 'val'))

def prepare_instruct_dataset(
    *,
    datasets_mix
):
    number_of_processes = get_max_number_of_cpu_processes()

    datasets_mix = copy.deepcopy(datasets_mix) if datasets_mix else copy.deepcopy(DEFAULT_INSTRUCT_MIX)

    #### VERIFY MIX FILE STRUCTURE
    seed, valid_datasets, probabilities = assert_common_structure_and_extract(datasets_mix, SUPPORTED_HF_DATASETS)

    download_and_prepare_data(
        seed=seed,
        valid_datasets=valid_datasets,
        probabilities=probabilities,
        number_of_processes=number_of_processes
    )

    print('\nData preparation completed.')
