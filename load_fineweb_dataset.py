import os
import multiprocessing as mp
import numpy as np
import tiktoken

from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict
from tiktoken.load import load_tiktoken_bpe
from pathlib import Path
from tokenizer import Tokenizer


DATA_CACHE_DIR = os.path.join(os.getcwd(), 'edu_fineweb10B')
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

dataset = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train', cache_dir='./cache')

tokenizer = Tokenizer('./tokenizer.model')

PAD_TOKEN_ID = tokenizer.pad_id
STOP_TOKENS = tokenizer.stop_tokens
EOT_TOKEN_ID = tokenizer.special_tokens['<|end_of_text|>']

print(f'Tokenizer vocabulary size: {tokenizer.number_of_words}')

NUMBER_OF_PROCESSES = max(1, os.cpu_count() - 1)
CHUNKSIZE = 16

print(f'\nPreparing dataset:\n')
print(f'Number of CPU processes: {NUMBER_OF_PROCESSES}\n')


def tokenize(doc):
    tokens = [EOT_TOKEN_ID]
    tokens.extend(tokenizer.encode(doc['text']))
    tokens_np = np.array(tokens)
    return tokens_np

def get_progress_bar(shard_size):
    return tqdm(total=shard_size, unit='tokens', desc=f'Shard {shard_index}')

def get_filename(shard_index):
    split = 'val' if shard_index == 0 else 'train'
    return os.path.join(DATA_CACHE_DIR, f'edufineweb_{split}_{shard_index:06d}')

with mp.Pool(NUMBER_OF_PROCESSES) as pool:
    shard_size = int(1e8)
    shard_index = 0
    token_count = 0
    progress_bar = None

    all_tokens_np = np.empty((shard_size,), dtype=np.uint32)

    for tokens in pool.imap(tokenize, dataset, chunksize=CHUNKSIZE):

        if token_count + len(tokens) < shard_size:
            new_token_count = token_count + len(tokens)

            all_tokens_np[token_count:new_token_count] = tokens
            token_count = new_token_count

            if progress_bar is None:
                progress_bar = get_progress_bar(shard_size)
            progress_bar.update(len(tokens))
        else:
            remaining = shard_size - token_count
            progress_bar.update(remaining)

            all_tokens_np[token_count:token_count + remaining] = tokens[:remaining]

            # Store maxed out shard
            np.save(get_filename(shard_index), all_tokens_np)

            shard_index += 1

            progress_bar.close()
            progress_bar = None

            # reset and populate with remaining. Update new token_count
            all_tokens_np[0:len(tokens) - remaining] = tokens[remaining:]
            token_count = len(tokens) - remaining

    if token_count != 0:
        np.save(get_filename(shard_index), all_tokens_np[:token_count])


print('\nData preparation completed.')
