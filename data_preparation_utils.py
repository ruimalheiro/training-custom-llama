import os
import glob
import re
import numpy as np
import sys

from tqdm import tqdm
from config import config


def get_max_number_of_cpu_processes():
    NUMBER_OF_PROCESSES = max(1, os.cpu_count() // 2)
    if config.number_of_cpu_processes != 0:
        NUMBER_OF_PROCESSES = max(1, min(config.number_of_cpu_processes, os.cpu_count()))
    print(f'Number of CPU processes: {NUMBER_OF_PROCESSES}\n')
    return NUMBER_OF_PROCESSES

def assert_common_structure_and_extract(datasets_mix, supported_datasets):
    ''' Validates common file structure and extracts seed, valid datasets and probabilities (normalized weight distribution)
    '''
    assert 'datasets' in datasets_mix
    assert 'seed' in datasets_mix

    datasets, seed = datasets_mix['datasets'], datasets_mix['seed']

    assert isinstance(seed, int)

    # Validate candidates
    valid_datasets = []
    for dataset in datasets:
        assert 'id' in dataset
        assert dataset['id'] in supported_datasets, dataset['id']
        assert 'split' in dataset
        assert 'weight' in dataset
        assert 0.0 <= float(dataset['weight']) <= 1.0

        # Get the ones with weight > 0, assume 100% default and include. Only ignore if 0.0 is set
        weight = dataset.get('weight', 1.0)
        if weight > 0:
            valid_datasets.append(dataset)

    assert valid_datasets, 'No datasets with weight > 0'

    probabilities = [float(ds['weight']) for ds in valid_datasets]

    # normalize probabilities
    total_p = sum(probabilities)
    assert total_p > 0
    probabilities = [p / total_p for p in probabilities]

    print('Mixture probabilities:', {ds['id']: round(p, 3) for ds, p in zip(valid_datasets, probabilities)}, '\n')

    return seed, valid_datasets, probabilities

def get_progress_bar(shard_index, shard_size, initial_tokens=0):
    return tqdm(total=shard_size, initial=initial_tokens, unit='tokens', desc=f'Shard {shard_index}')

def get_filename(shard_index, data_cache_dir, shard_file_prefix):
    base_filename = f'{shard_file_prefix}_{shard_index:06d}'
    final_path = os.path.join(data_cache_dir, f'{base_filename}.npy')
    temp_path = os.path.join(data_cache_dir, f'{base_filename}.temp.npy')
    return temp_path, final_path

def save_filename(all_tokens_np, shard_index, data_cache_dir, shard_file_prefix):
    temp_filepath, final_filepath = get_filename(shard_index, data_cache_dir, shard_file_prefix)
    try:
        np.save(temp_filepath, all_tokens_np)
        os.replace(temp_filepath, final_filepath) # Small protection in case of file corruption..
    except Exception as e:
        print(f'\nError saving shard {shard_index} to {final_filepath}: {e}')
        print('Stopping processing. Rerun the script to resume.')
        if os.path.exists(temp_filepath):
            try: os.remove(temp_filepath)
            except OSError: pass
        sys.exit(1)


def find_last_shard_info(data_cache_dir, shard_file_prefix):
    shard_pattern = re.compile(rf'^{re.escape(shard_file_prefix)}_(\d+)\.npy$')
    files = []

    for file_path in glob.glob(os.path.join(data_cache_dir, f'{shard_file_prefix}_*.npy')):
        match = shard_pattern.match(os.path.basename(file_path))
        if match:
            files.append((int(match.group(1)), file_path))

    if not files:
        print('No previous shards found.')
        return -1, 0

    files.sort(key=lambda x: x[0])

    # validate sequence: There should not be missing shards like 1 2 3 6
    indexes = [i for i, _ in files]
    if indexes != list(range(0, indexes[-1] + 1)):
        raise ValueError(f'Shard sequence broken: {indexes}')

    total_tokens_saved = 0
    for _, file_path in files:
        shard = np.load(file_path, mmap_mode='r', allow_pickle=False)
        total_tokens_saved += shard.shape[0]
        del shard

    last_shard_index = files[-1][0]
    print(f'Resuming after shard {last_shard_index}. Total tokens in existing shards: {total_tokens_saved}')
    return last_shard_index, total_tokens_saved

def prepare_dataset(
    *,
    dataset,
    target_folder,
    shard_file_prefix,
    shard_size
):
    print(f'\nPreparing dataset:')

    data_cache_dir = os.path.join(os.getcwd(), target_folder)
    os.makedirs(data_cache_dir, exist_ok=True)

    last_shard_index, tokens_to_skip = find_last_shard_info(data_cache_dir, shard_file_prefix)
    start_shard_index = last_shard_index + 1
    processed_token_count_in_loop = 0
    skipping_phase = tokens_to_skip > 0

    skipping_progress_bar = None
    if skipping_phase:
        print('Starting skipping phase...')
        skipping_progress_bar = tqdm(total=tokens_to_skip, desc='Skipping tokens', unit='tokens', smoothing=0.1)

    shard_index = start_shard_index
    token_count = 0
    progress_bar = None
    all_tokens_np = np.empty((shard_size,), dtype=np.uint32)

    for batch in dataset.iter(batch_size=2048):
        for tokens in batch['input_ids']:
            if tokens.size == 0:
                continue

            tokens_len = tokens.size

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
                    tokens_len = tokens_to_process.size
                    print(f'Skipping phase complete. Starting to process from token {offset+1} of the current batch.')
                    skipping_phase = False
            else:
                tokens_to_process = tokens

            if progress_bar is None:
                progress_bar = get_progress_bar(shard_index, shard_size, initial_tokens=token_count)

            # Normal processing...
            if token_count + tokens_len < shard_size:
                all_tokens_np[token_count:token_count + tokens_len] = tokens_to_process
                token_count += tokens_len
                progress_bar.update(tokens_len)
            else:
                remaining_space = shard_size - token_count
                all_tokens_np[token_count:token_count + remaining_space] = tokens_to_process[:remaining_space]
                progress_bar.update(remaining_space)
                progress_bar.close()

                # save file
                save_filename(all_tokens_np, shard_index, data_cache_dir, shard_file_prefix)

                shard_index += 1
                progress_bar = None

                tokens_leftover = tokens_to_process[remaining_space:]
                token_count = tokens_leftover.size
                all_tokens_np[:token_count] = tokens_leftover

                if token_count > 0:
                    progress_bar = get_progress_bar(shard_index, shard_size, initial_tokens=token_count)

    if token_count > 0:
        if progress_bar:
            progress_bar.close()
        save_filename(all_tokens_np[:token_count], shard_index, data_cache_dir, shard_file_prefix)
