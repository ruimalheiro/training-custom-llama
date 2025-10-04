import os
import requests
import json
import torch

from tqdm import tqdm
from config import config
from tokenizer import init_tokenizer


HELLASWAG_VAL_URL = 'https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl'

DATA_CACHE_DIR = os.path.join(config.hellaswag_path)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

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
    label = example['label'] # Index for the correct completion
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
        mask[i, :len(mask_row)] = mask_row.clone().detach()

    return tokens, mask, label

#### DOWNLOAD_DATA
TEMP_FILE_PATH = os.path.join('./cache', 'hellaswag_val.jsonl')

if not os.path.exists(TEMP_FILE_PATH):
    print(f'Downloading hellaswag to {TEMP_FILE_PATH}...')

    response = requests.get(HELLASWAG_VAL_URL, stream=True)
    content_length = int(response.headers.get('content-length', 0))

    pbar = tqdm(total=content_length, desc=f'Downloading...')

    with open(TEMP_FILE_PATH, 'wb') as file:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

    pbar.close()
    print('\nDownload complete.')
else:
    print(f'temp file already exists: {TEMP_FILE_PATH}...')

#### PREPARATION
DATA_FILENAME = os.path.join(DATA_CACHE_DIR, f'hellaswag_val.jsonl')

if not os.path.exists(DATA_FILENAME):
    with open(TEMP_FILE_PATH, 'r') as file:
        lines = file.readlines()
    with open(DATA_FILENAME, 'w', encoding='utf-8') as file:
        for line in tqdm(lines, 'Preparing examples...'):
            example = json.loads(line)
            tokens, mask, label = prepare_example(example)
            processed_example = {
                'tokens': tokens.tolist(),
                'mask': mask.tolist(),
                'label': label
            }
            json.dump(processed_example, file, ensure_ascii=False)
            file.write('\n')
    print(f'Hellaswag preprocessing completed and stored at: {DATA_FILENAME}')
else:
    print(f'Hellaswag preprocessed file already exists: {DATA_FILENAME}')
