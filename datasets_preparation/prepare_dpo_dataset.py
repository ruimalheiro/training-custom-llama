import os
import numpy as np
import re
import copy
import time

from config import config
from tokenizer import init_tokenizer
from datasets import (
    load_dataset,
    interleave_datasets
)
from datasets_preparation.data_preparation_utils import (
    get_max_number_of_cpu_processes,
    assert_common_structure_and_extract
)
from datasets_preparation.default_mixes import DEFAULT_DPO_MIX


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

#### SUPPORTED DATASETS
SUPPORTED_HF_DATASETS = {
    'Anthropic/hh-rlhf': {
        'default': {
            'id': 'Anthropic/hh-rlhf',
            'split': 'train',
            'adapter': adapt_anthropic_hh_rlhf
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

tokenizer = None
SYS = None
SYS_PROMPT = None
NL = None
ASSIST = None

def tokenize(doc):
    global tokenizer, SYS, SYS_PROMPT, NL, ASSIST
    if tokenizer is None:
        tokenizer = init_tokenizer(config.tokenizer_checkpoint_path, config.huggingface_tokenizer)
        SYS = tokenizer.encode('system')
        SYS_PROMPT = tokenizer.encode('\n' + config.system_prompt)
        NL = tokenizer.encode('\n')
        ASSIST = tokenizer.encode('assistant')

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
            data = adapter(doc, transforms)
            data['prompt'] = ensure_user_first(data['prompt'])
            if config.hf_include_source_id:
                data.update({'source': ds_id})

            return data

        ds = ds.map(normalize, num_proc=number_of_processes)
        ds = ds.filter(lambda x: len(x['prompt']) > 0, num_proc=number_of_processes)

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

    splits['train'].save_to_disk(os.path.join(config.dpo_dataset_target_path, 'train'))
    splits['test'] .save_to_disk(os.path.join(config.dpo_dataset_target_path, 'val'))


def prepare_dpo_dataset(
    *,
    datasets_mix
):
    number_of_processes = get_max_number_of_cpu_processes()

    datasets_mix = copy.deepcopy(datasets_mix) if datasets_mix else copy.deepcopy(DEFAULT_DPO_MIX)

    #### VERIFY MIX FILE STRUCTURE
    seed, valid_datasets, probabilities = assert_common_structure_and_extract(datasets_mix, SUPPORTED_HF_DATASETS)

    download_and_prepare_data(
        seed=seed,
        valid_datasets=valid_datasets,
        probabilities=probabilities,
        number_of_processes=number_of_processes
    )

    print('\nData preparation completed.')
