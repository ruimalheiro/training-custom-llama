import os
import numpy as np
import re
import json

from tokenizer import init_tokenizer
from config import config
from datasets import (
    load_dataset,
    interleave_datasets
)
from data_preparation_utils import (
    get_max_number_of_cpu_processes,
    assert_common_structure_and_extract
)


NUMBER_OF_PROCESSES = get_max_number_of_cpu_processes()

#### SUPPORTED DATASETS
SUPPORTED_HF_DATASETS = [
    'Anthropic/hh-rlhf'
]

#### ADAPTERS
def adapt_anthropic_hh_rlhf(doc, transforms):
    def extract_interactions(text):
        conversation = []
        assistant_statements = []

        role_re = re.compile(
            r'(?:^|\n\n)(Human|Assistant): (.*?)(?=\n\n(?:Human|Assistant): |\Z)',
            re.DOTALL,
        )

        for role, content in role_re.findall(text):
            role = 'user' if role.lower() == 'human' else 'assistant'

            conversation.append({
                'role': role,
                'content': content
            })

            if role == 'assistant':
                assistant_statements.append(content)

        if not len(assistant_statements):
            assistant_statements.append('')

        return conversation, assistant_statements

    chosen_conversation, chosen_assistant = extract_interactions(doc['chosen'])
    _, rejected_assistant = extract_interactions(doc['rejected'])

    prompt = chosen_conversation[:-1]

    return {'prompt': prompt, 'chosen': chosen_assistant[-1], 'rejected': rejected_assistant[-1] }

ADAPTERS_MAP = {
    'Anthropic/hh-rlhf': adapt_anthropic_hh_rlhf
}

#### VERIFY MIX FILE STRUCTURE
datasets_mix = json.load(open(config.dpo_dataset_mix_file))

seed, valid_datasets, probabilities = assert_common_structure_and_extract(datasets_mix, SUPPORTED_HF_DATASETS)

#### PREPARATION
def ensure_user_first(conversation):
    if not conversation:
        return conversation
    if conversation[0]['role'] != 'user':
        # add default empty user conversation if it is missing (If it starts with assistant)
        return [{'role': 'user', 'content': ''}] + conversation
    return conversation

tokenizer = init_tokenizer(config.tokenizer_checkpoint_path, config.huggingface_tokenizer)
SYS = tokenizer.encode('system')
SYS_PROMPT = tokenizer.encode('\n' + config.system_prompt)
NL = tokenizer.encode('\n')
ASSIST = tokenizer.encode('assistant')

def tokenize(doc):
    prompt_sequence = doc['prompt']
    chosen_assistant = doc['chosen']
    rejected_assistant = doc['rejected']

    bot = tokenizer.bos_id
    sh = tokenizer.sh_id
    eh = tokenizer.eh_id
    eot = tokenizer.eot_id

    tokens = []
    tokens.extend([bot])
    tokens.extend([sh])
    tokens.extend(SYS)
    tokens.extend([eh])
    tokens.extend(SYS_PROMPT)
    tokens.extend([eot])

    for interaction in prompt_sequence:
        role = interaction['role']
        content = interaction['content']

        tokens.extend([sh])
        tokens.extend(tokenizer.encode(role))
        tokens.extend([eh])
        tokens.extend(NL)
        tokens.extend(tokenizer.encode(content))
        tokens.extend([eot])

    prompt_input_ids = np.array(tokens, dtype=np.uint32)

    def build_answer_sequence(text):
        tokens = []
        tokens.extend([sh])
        tokens.extend(ASSIST)
        tokens.extend([eh])
        tokens.extend(NL)
        tokens.extend(tokenizer.encode(text))
        tokens.extend([eot])
        return np.array(tokens, dtype=np.uint32)

    chosen_input_ids = build_answer_sequence(chosen_assistant)
    rejected_input_ids = build_answer_sequence(rejected_assistant)

    return { 
        'prompt_input_ids': prompt_input_ids,
        'chosen_input_ids': chosen_input_ids,
        'rejected_input_ids': rejected_input_ids 
    }

prepared_datasets = []
for dataset in valid_datasets:
    ds_id = dataset['id']
    name = dataset.get('name', None)

    adapter_id = f'{ds_id}_{name}' if name and name != 'default' else ds_id

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

    adapter = ADAPTERS_MAP.get(adapter_id)

    def normalize(doc):
        data = adapter(doc, transforms)
        data['prompt'] = ensure_user_first(data['prompt'])
        data.update({'source': ds_id})

        return data

    ds = ds.map(normalize, num_proc=NUMBER_OF_PROCESSES)
    ds = ds.filter(lambda x: len(x['prompt']) > 0, num_proc=NUMBER_OF_PROCESSES)

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

splits['train'].save_to_disk(os.path.join(config.dpo_dataset_target_path, 'train'))
splits['test'] .save_to_disk(os.path.join(config.dpo_dataset_target_path, 'val'))

print('\nData preparation completed.')
