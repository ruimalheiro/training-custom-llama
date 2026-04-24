import os
import json
import torch

from pathlib import Path
from tqdm import tqdm
from config import config
from tokenizer import init_tokenizer
from data_preparation_utils import get_max_number_of_cpu_processes
from datasets import load_dataset


NUMBER_OF_PROCESSES = get_max_number_of_cpu_processes()

CURRENT_DIR = Path(__file__).resolve().parent

DATA_CACHE_DIR = CURRENT_DIR / config.hellaswag_path
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATA_FILENAME = DATA_CACHE_DIR / 'hellaswag_val.jsonl'

tokenizer = init_tokenizer(config.tokenizer_checkpoint_path, config.huggingface_tokenizer)

def prepare_example(example):
    """
    Sample example from hellaswag (without some of the metadata):
        {
          "ctx": "A man is sitting on a roof. he",
          "label": 3,
          "endings": [
            "is using wrap to wrap a pair of skis.",
            "is ripping level tiles off.",
            "is holding a rubik's cube.",
            "starts pulling up roofing on a roof."
          ]
        }
    """
    context = example['ctx']
    label_index = example['label'] # Index for the correct completion
    endings = example['endings'] # Candidates - always 4

    context_tokens = tokenizer.encode(context)

    tokens_rows = []
    mask_rows = []
    for ending in endings:
        ending_tokens = tokenizer.encode(ending)
        tokens_rows.append(context_tokens + ending_tokens)

        mask_row = torch.cat([
            torch.zeros(len(context_tokens), dtype=torch.long),
            torch.ones(len(ending_tokens), dtype=torch.long)
        ])
        mask_rows.append(mask_row)

    # rows can have different lengths so pick max for row length.
    max_len = max(len(row) for row in tokens_rows)

    # (4 candidates * max length)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tokens_row, mask_row) in enumerate(zip(tokens_rows, mask_rows)):
        tokens[i, :len(tokens_row)] = torch.tensor(tokens_row, dtype=torch.long)
        mask[i, :len(mask_row)] = mask_row

    processed_example = {
        'tokens': tokens.tolist(),
        'mask': mask.tolist(),
        'label_index': label_index
    }

    return processed_example

if not DATA_FILENAME.exists():
    ds = load_dataset(
        'Rowan/hellaswag',
        split='validation',
        num_proc=NUMBER_OF_PROCESSES,
        token=config.hf_token
    )

    with open(DATA_FILENAME, 'w', encoding='utf-8') as file:
        for example in tqdm(ds, desc='Preparing HellaSwag eval dataset'):
            processed_example = prepare_example(example)
            json.dump(processed_example, file, ensure_ascii=False)
            file.write('\n')
    print(f'HellaSwag preprocessing completed and stored at: {DATA_FILENAME}')
else:
    print(f'HellaSwag preprocessed file already exists: {DATA_FILENAME}')
