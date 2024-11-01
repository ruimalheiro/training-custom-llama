import os
import numpy as np
import torch
import random


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, batch_size, sequence_length, is_master_process, process_rank, num_processes, data_root, split, use_shuffle=False):
        self.B = batch_size
        self.S = sequence_length
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}
        self.use_shuffle = use_shuffle

        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards

        assert len(self.shards) > 0, f'no shards found in split {split}'

        if is_master_process:
            print(f'found {len(shards)} shards for split {split}')

        self.reset()

    def reset(self):
        self.current_shard = 0
        if self.use_shuffle:
            random.shuffle(self.shards)
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position =  self.B * self.S * self.process_rank

    def next_batch(self):
        B, S = self.B, self.S
        buf = self.tokens[self.current_position : self.current_position+B*S+1]
        x = (buf[:-1]).view(B, S)
        y = (buf[1:]).view(B, S)
        self.current_position += B * S * self.num_processes
        if self.current_position + (B * S * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.S * self.process_rank
        return x, y


def init_data_loaders(batch_size, sequence_length, is_master_process, process_rank, num_processes, data_root, use_shuffle):
    train_loader = DataLoaderLite(
        batch_size=batch_size,
        sequence_length=sequence_length,
        is_master_process=is_master_process,
        process_rank=process_rank,
        num_processes=num_processes,
        data_root=data_root,
        split='train',
        use_shuffle=use_shuffle
    )
    val_loader = DataLoaderLite(
        batch_size=batch_size,
        sequence_length=sequence_length,
        is_master_process=is_master_process,
        process_rank=process_rank,
        num_processes=num_processes,
        data_root=data_root,
        split='val',
        use_shuffle=use_shuffle
    )
    return train_loader, val_loader
