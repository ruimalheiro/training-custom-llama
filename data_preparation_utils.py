import os
import multiprocessing as mp
import numpy as np

from tqdm import tqdm


def get_progress_bar(shard_index, shard_size):
    return tqdm(total=shard_size, unit='tokens', desc=f'Shard {shard_index}')

def get_filename(shard_index, data_cache_dir, shard_file_prefix):
    split = 'val' if shard_index == 0 else 'train'
    return os.path.join(data_cache_dir, f'{shard_file_prefix}_{split}_{shard_index:06d}')

def prepare_dataset(
    *,
    dataset,
    tokenize_function,
    target_folder,
    shard_file_prefix,
    number_of_processes=None,
    chunksize=16
):
    data_cache_dir = os.path.join(os.getcwd(), target_folder)
    os.makedirs(data_cache_dir, exist_ok=True)

    if number_of_processes is None:
        number_of_processes = max(1, os.cpu_count() // 2)
        print(f'Number of CPU processes: {number_of_processes}\n')

    print(f'\nPreparing dataset:\n')

    with mp.Pool(number_of_processes) as pool:
        shard_size = int(1e8)
        shard_index = 0
        token_count = 0
        progress_bar = None

        all_tokens_np = np.empty((shard_size,), dtype=np.uint32)

        for tokens in pool.imap(tokenize_function, dataset, chunksize=chunksize):

            if token_count + len(tokens) < shard_size:
                new_token_count = token_count + len(tokens)

                all_tokens_np[token_count:new_token_count] = tokens
                token_count = new_token_count

                if progress_bar is None:
                    progress_bar = get_progress_bar(shard_index, shard_size)
                progress_bar.update(len(tokens))
            else:
                remaining = shard_size - token_count
                progress_bar.update(remaining)

                all_tokens_np[token_count:token_count + remaining] = tokens[:remaining]

                # Store maxed out shard
                np.save(get_filename(shard_index, data_cache_dir, shard_file_prefix), all_tokens_np)

                shard_index += 1

                progress_bar.close()
                progress_bar = None

                # reset and populate with remaining. Update new token_count
                all_tokens_np[0:len(tokens) - remaining] = tokens[remaining:]
                token_count = len(tokens) - remaining

        if token_count != 0:
            np.save(get_filename(shard_index, data_cache_dir, shard_file_prefix), all_tokens_np[:token_count])


    print('\nData preparation completed.')
