import os
import requests

from tqdm import tqdm
from config import config


HELLASWAG_VAL_URL = 'https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl'

DATA_CACHE_DIR = os.path.join(config.hellaswag_path)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

DATA_FILENAME = os.path.join(DATA_CACHE_DIR, f'hellaswag_val.jsonl')

if not os.path.exists(DATA_FILENAME):
    print(f'Downloading {HELLASWAG_VAL_URL} to {DATA_FILENAME}...')

    response = requests.get(HELLASWAG_VAL_URL, stream=True)
    content_length = int(response.headers.get('content-length', 0))

    pbar = tqdm(total=content_length, desc=f'Downloading... {DATA_FILENAME}')

    with open(DATA_FILENAME, 'wb') as file:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

    pbar.close()
    print('\nDownload complete.')
