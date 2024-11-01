import os
import requests

from tqdm import tqdm


DATA_CACHE_DIR = os.path.join(os.getcwd(), 'hellaswag')
DATA_CACHE_DIR

def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

hellaswags = {
    'train': 'https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl',
    'val': 'https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl',
    'test': 'https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl',
}

def download(split):
    """Downloads HellaSwag DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f'hellaswag_{split}.jsonl')
    if not os.path.exists(data_filename):
        print(f'Downloading {data_url} to {data_filename}...')
        download_file(data_url, data_filename)

download('val')
