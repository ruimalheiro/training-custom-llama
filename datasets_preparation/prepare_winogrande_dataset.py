import os
import json
import torch

from pathlib import Path
from tqdm import tqdm
from config import config
from tokenizer import init_tokenizer
from datasets_preparation.data_preparation_utils import get_max_number_of_cpu_processes
from datasets import load_dataset


def prepare_winogrande_dataset():
    number_of_processes = get_max_number_of_cpu_processes()

    current_dir = Path(__file__).resolve().parent.parent

    data_cache_dir = current_dir / config.winogrande_path
    data_cache_dir.mkdir(parents=True, exist_ok=True)
    data_filename = data_cache_dir / 'winogrande_val.jsonl'

    tokenizer = init_tokenizer(config.tokenizer_checkpoint_path, config.huggingface_tokenizer)

    def prepare_example(example):
        """
        Sample example:
        {
            "sentence": "Only the bag got melted and not the wood when they were inside the flame. The _ is soft.",
            "option1": "wood",
            "option2": "bag",
            "answer": "2",
        }

        """
        sentence = example['sentence']
        option1 = example['option1'].strip()
        option2 = example['option2'].strip()
        label_index = int(example['answer']) - 1

        if sentence.count('_') != 1:
            raise ValueError(f'Expected exactly one blank in sentence, got: {sentence}')
        prefix, suffix = sentence.split('_', 1)
        prefix_tokens = tokenizer.encode(prefix)

        tokens_rows = []
        mask_rows = []
        for option in [option1, option2]:
            candidate_text = prefix + option + suffix
            candidate_tokens = tokenizer.encode(candidate_text)

            mask_row = torch.cat([
                torch.zeros(len(prefix_tokens), dtype=torch.long),
                torch.ones(len(candidate_tokens) - len(prefix_tokens), dtype=torch.long)
            ])

            tokens_rows.append(candidate_tokens)
            mask_rows.append(mask_row)

        max_len = max(len(row) for row in tokens_rows)

        tokens = torch.zeros((2, max_len), dtype=torch.long)
        mask = torch.zeros((2, max_len), dtype=torch.long)
        for i, (tokens_row, mask_row) in enumerate(zip(tokens_rows, mask_rows)):
            tokens[i, :len(tokens_row)] = torch.tensor(tokens_row, dtype=torch.long)
            mask[i, :len(mask_row)] = mask_row

        processed_example = {
            'tokens': tokens.tolist(),
            'mask': mask.tolist(),
            'label_index': label_index
        }

        return processed_example

    if not data_filename.exists():
        ds = load_dataset(
            'allenai/winogrande',
            name='winogrande_debiased',
            split='validation',
            num_proc=number_of_processes,
            token=config.hf_token
        )

        with open(data_filename, 'w', encoding='utf-8') as file:
            for example in tqdm(ds, desc='Preparing WinoGrande eval dataset'):
                processed_example = prepare_example(example)
                json.dump(processed_example, file, ensure_ascii=False)
                file.write('\n')
        print(f'WinoGrande preprocessing completed and stored at: {data_filename}')
    else:
        print(f'WinoGrande preprocessed file already exists: {data_filename}')
