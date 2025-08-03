import os
import numpy as np
import re

from datasets import load_dataset
from tokenizer import init_tokenizer
from config import config


tokenizer = init_tokenizer(config.tokenizer_checkpoint_path, config.huggingface_tokenizer)

NUMBER_OF_PROCESSES = max(1, os.cpu_count() // 2)
if config.number_of_cpu_processes != 0:
    NUMBER_OF_PROCESSES = max(1, min(config.number_of_cpu_processes, os.cpu_count()))
print(f'Number of CPU processes: {NUMBER_OF_PROCESSES}\n')

""" NOTE: This is prepared to work with Anthropic specific RLHF dataset format but could be adapted for different structures. The dataset in mind for this one was:
    https://huggingface.co/datasets/Anthropic/hh-rlhf 
"""

dataset = load_dataset(
    config.dpo_dataset,
    name=config.dpo_dataset_name,
    split=config.dpo_dataset_split,
    cache_dir='./cache',
    num_proc=NUMBER_OF_PROCESSES
)

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

def tokenize(doc):
    chosen_conversation, chosen_assistant = extract_interactions(doc['chosen'])
    _, rejected_assistant = extract_interactions(doc['rejected'])

    prompt_sequence = chosen_conversation[:-1]

    assert prompt_sequence[0]['role'] == 'user'

    bot = tokenizer.bos_id
    sh = tokenizer.sh_id
    eh = tokenizer.eh_id
    eot = tokenizer.eot_id

    tokens = []
    tokens.extend([bot])
    tokens.extend([sh])
    tokens.extend(tokenizer.encode('system'))
    tokens.extend([eh])
    tokens.extend(tokenizer.encode('\n' + 'You are a helpful AI assistant'))
    tokens.extend([eot])

    for interaction in prompt_sequence:
        role = interaction['role']
        content = interaction['content']

        tokens.extend([sh])
        tokens.extend(tokenizer.encode(role))
        tokens.extend([eh])
        tokens.extend(tokenizer.encode('\n'))
        tokens.extend(tokenizer.encode(content))
        tokens.extend([eot])

    prompt_input_ids = np.array(tokens, dtype=np.uint32)

    def build_answer_sequence(text):
        tokens = []
        tokens.extend([sh])
        tokens.extend(tokenizer.encode('assistant'))
        tokens.extend([eh])
        tokens.extend(tokenizer.encode('\n'))
        tokens.extend(tokenizer.encode(text))
        tokens.extend([eot])
        return np.array(tokens, dtype=np.uint32)

    chosen_input_ids = build_answer_sequence(chosen_assistant[-1])
    rejected_input_ids = build_answer_sequence(rejected_assistant[-1])

    return { 
        'prompt_input_ids': prompt_input_ids,
        'chosen_input_ids': chosen_input_ids,
        'rejected_input_ids': rejected_input_ids 
    }

dataset = dataset.map(
    tokenize,
    num_proc=NUMBER_OF_PROCESSES,
    remove_columns=dataset.column_names
)

train_val = dataset.train_test_split(test_size=0.01, seed=42)
train_val['train'].save_to_disk(os.path.join(config.dpo_dataset_target_path, 'train'))
train_val['test'] .save_to_disk(os.path.join(config.dpo_dataset_target_path, 'val'))
