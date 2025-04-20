import os
import glob
import re
import multiprocessing as mp
import numpy as np
import time
import sys

from tqdm import tqdm


def get_progress_bar(shard_index, shard_size, initial_tokens=0):
    return tqdm(total=shard_size, initial=initial_tokens, unit='tokens', desc=f'Shard {shard_index}')

def get_filename(shard_index, data_cache_dir, shard_file_prefix):
    split = 'val' if shard_index == 0 else 'train'
    base_filename = f'{shard_file_prefix}_{split}_{shard_index:06d}'
    final_path = os.path.join(data_cache_dir, f'{base_filename}.npy')
    temp_path = os.path.join(data_cache_dir, f'{base_filename}.temp.npy')
    return temp_path, final_path

def save_filename(pool, all_tokens_np, shard_index, data_cache_dir, shard_file_prefix):
    temp_filepath, final_filepath = get_filename(shard_index, data_cache_dir, shard_file_prefix)
    try:
        np.save(temp_filepath, all_tokens_np)
        os.rename(temp_filepath, final_filepath) # Small protection in case of file corruption..
    except Exception as e:
        print(f'\nError saving shard {shard_index} to {final_filepath}: {e}')
        print('Stopping processing. Rerun the script to resume.')
        if os.path.exists(temp_filepath):
            try: os.remove(temp_filepath)
            except OSError: pass
        if pool:
            pool.terminate()
        sys.exit(1)


def find_last_shard_info(data_cache_dir, shard_file_prefix):
    last_shard_index = -1
    total_tokens_saved = 0
    shard_pattern = re.compile(rf'{shard_file_prefix}_(train|val)_(\d+)\.npy')
    shard_sizes = {}

    potential_files = glob.glob(os.path.join(data_cache_dir, f'{shard_file_prefix}_*.npy'))

    for f in potential_files:
        match = shard_pattern.match(os.path.basename(f))
        if match:
            shard_index = int(match.group(2))
            with open(f, 'rb') as fh:
                version = np.lib.format.read_magic(fh)
                shape, fortran_order, dtype = np.lib.format._read_array_header(fh, version)
                shard_size = shape[0]

            shard_sizes[shard_index] = shard_size
            if shard_index > last_shard_index:
                last_shard_index = shard_index

    if last_shard_index >= 0:
        current_tokens_sum = 0
        for i in range(last_shard_index + 1):
            if i in shard_sizes:
                current_tokens_sum += shard_sizes[i]
            else:
                raise ValueError('Shard sequence broken (not continuous)')
        total_tokens_saved = current_tokens_sum

    print(f'Resuming after shard {last_shard_index}. Total tokens in existing shards: {total_tokens_saved}')
    return last_shard_index, total_tokens_saved

def prepare_dataset(
    *,
    dataset,
    tokenize_function,
    target_folder,
    shard_file_prefix,
    shard_size=int(1e8),
    number_of_processes=None,
    chunksize=16
):
    data_cache_dir = os.path.join(os.getcwd(), target_folder)
    os.makedirs(data_cache_dir, exist_ok=True)

    if number_of_processes is None:
        number_of_processes = max(1, os.cpu_count() // 2)
        print(f'Number of CPU processes: {number_of_processes}\n')

    last_shard_index, tokens_to_skip = find_last_shard_info(data_cache_dir, shard_file_prefix)
    start_shard_index = last_shard_index + 1
    processed_token_count_in_loop = 0
    skipping_phase = tokens_to_skip > 0

    print(f'\nPreparing dataset:\n')

    skipping_progress_bar = None
    if skipping_phase:
        print('Starting skipping phase...')
        skipping_progress_bar = tqdm(total=tokens_to_skip, desc='Skipping tokens', unit='tokens', smoothing=0.1)

    with mp.Pool(number_of_processes) as pool:
        shard_index = start_shard_index
        token_count = 0
        progress_bar = None
        all_tokens_np = np.empty((shard_size,), dtype=np.uint32)

        for tokens in pool.imap(tokenize_function, dataset, chunksize=chunksize):
            if tokens.size == 0:
                continue

            tokens_len = len(tokens)

            # Skipping phase...
            if skipping_phase:
                if processed_token_count_in_loop + tokens_len <= tokens_to_skip:
                    processed_token_count_in_loop += tokens_len
                    if skipping_progress_bar:
                        skipping_progress_bar.update(tokens_len)
                    continue
                else:
                    if skipping_progress_bar:
                        remaining_to_skip = tokens_to_skip - processed_token_count_in_loop
                        if remaining_to_skip > 0:
                           skipping_progress_bar.update(remaining_to_skip)
                        skipping_progress_bar.close()
                        skipping_progress_bar = None

                    offset = tokens_to_skip - processed_token_count_in_loop
                    tokens_to_process = tokens[offset:]
                    tokens_len = len(tokens_to_process)
                    print(f'Skipping phase complete. Starting to process from token {offset+1} of the current batch.')
                    skipping_phase = False
            else:
                tokens_to_process = tokens

            # Normal processing...
            if token_count + tokens_len < shard_size:
                all_tokens_np[token_count:token_count + tokens_len] = tokens_to_process
                token_count += tokens_len

                if progress_bar is None:
                    progress_bar = get_progress_bar(shard_index, shard_size, initial_tokens=token_count)
                progress_bar.update(tokens_len)
            else:
                remaining_space = shard_size - token_count
                all_tokens_np[token_count:token_count + remaining_space] = tokens_to_process[:remaining_space]

                if progress_bar is None:
                    progress_bar = get_progress_bar(shard_index, shard_size, initial_tokens=shard_size)
                progress_bar.update(remaining_space)
                progress_bar.close()

                # save file
                save_filename(pool, all_tokens_np, shard_index, data_cache_dir, shard_file_prefix)

                shard_index += 1
                progress_bar = None

                tokens_leftover = tokens_to_process[remaining_space:]
                token_count = len(tokens_leftover)
                all_tokens_np[:token_count] = tokens_leftover

                if token_count > 0:
                     progress_bar = get_progress_bar(shard_index, shard_size, initial_tokens=token_count)

        if token_count > 0:
            if progress_bar:
                progress_bar.close()
            save_filename(pool, all_tokens_np[:token_count], shard_index, data_cache_dir, shard_file_prefix)


    print('\nData preparation completed.')
