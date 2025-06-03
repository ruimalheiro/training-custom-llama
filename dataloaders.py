import os
import numpy as np
import torch
import random
import datasets

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class PretrainDataLoader:
    def __init__(self, batch_size, sequence_length, is_master_process, process_rank, num_processes, data_root, split, use_shuffle=False):
        self.B = batch_size
        self.S = sequence_length
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.data_root = data_root
        assert split in {'train', 'val'}
        self.split = split
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

    def calculate_max_tokens(self):
        total_tokens = (len(self.shards) - 1) * 1e8
        total_tokens += np.load(self.shards[-1]).size
        return int(total_tokens)

    def reset(self):
        self.current_shard = 0
        if self.use_shuffle:
            random.shuffle(self.shards)
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position =  self.B * self.S * self.process_rank

    def state_dict(self):
        return {
            'shards'          : list(self.shards),
            'current_shard'   : self.current_shard,
            'current_position': self.current_position,
        }

    def load_state_dict(self, state):
        self.shards         = state['shards']
        self.current_shard  = state['current_shard']
        self.tokens         = load_tokens(self.shards[self.current_shard])
        self.current_position = state['current_position']

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

class InstructDataLoader:
    def __init__(self, batch_size, sequence_length, is_master_process, process_rank, num_processes, data_root, split, use_shuffle, pad_id, drop_last):
        self.B = batch_size
        self.S = sequence_length

        dataset = datasets.load_from_disk(os.path.join(data_root, split))
        if is_master_process:
            print(f'found {len(dataset)} examples for {split}')

        self.is_master_process = is_master_process

        assert isinstance(dataset, datasets.Dataset)

        self.sampler = DistributedSampler(
            dataset,
            num_replicas=num_processes,
            rank=process_rank,
            shuffle=use_shuffle
        )

        def collate(examples):
            ids = [torch.tensor(e['input_ids']) for e in examples]
            labels = [torch.tensor(e['labels']) for e in examples]

            ids = pad_sequence(
                ids,
                batch_first=True,
                padding_value=int(pad_id)
            )
            labels = pad_sequence(
                labels,
                batch_first=True,
                padding_value=-100
            )
            if ids.size(1) > sequence_length:
                ids  = ids[:, -sequence_length:]
                labels = labels[:, -sequence_length:]

            return ids, labels

        self._dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=self.sampler,
            collate_fn=collate,
            drop_last=drop_last,
            pin_memory=True
        )
        self._iterator = iter(self._dataloader)

    def next_batch(self):
        try:
            return next(self._iterator)
        except StopIteration:
            self.reset()
            return next(self._iterator)

    def reset(self):
        if hasattr(self.sampler, 'set_epoch'):
            update_epoch = 0
            if hasattr(self.sampler, 'epoch'):
                update_epoch = self.sampler.epoch + 1
            self.sampler.set_epoch(update_epoch)
        self._iterator = iter(self._dataloader)

    def num_examples(self):
        return len(self._dataloader.dataset)

    def calculate_max_tokens(self):
        raise NotImplementedError('Not applicable for dialogue loader')

    def state_dict(self):
        return {'epoch': getattr(self.sampler, 'epoch', 0)}

    def load_state_dict(self, state):
        if 'epoch' not in state:
            if self.is_master_process:
                print('Warning - "epoch" not present, starting fresh dataloader (most likely transition from pretraining to SFT).')
            return
        epoch = state['epoch']
        self.sampler.set_epoch(epoch)
        self.sampler.epoch = epoch
        self._iterator = iter(self._dataloader)

def init_data_loaders(
    batch_size,
    sequence_length,
    is_master_process,
    process_rank,
    num_processes,
    data_root,
    is_instruct_training=False,
    pad_id=None
):
    if not is_instruct_training:
        if is_master_process:
            print('Pretrain Data Loaders:')
            print('----------------------------------------')
        train_loader = PretrainDataLoader(
            batch_size=batch_size,
            sequence_length=sequence_length,
            is_master_process=is_master_process,
            process_rank=process_rank,
            num_processes=num_processes,
            data_root=data_root,
            split='train',
            use_shuffle=True
        )
        val_loader = PretrainDataLoader(
            batch_size=batch_size,
            sequence_length=sequence_length,
            is_master_process=is_master_process,
            process_rank=process_rank,
            num_processes=num_processes,
            data_root=data_root,
            split='val',
            use_shuffle=True
        )
    else:
        assert pad_id is not None

        if is_master_process:
            print('Instruct Finetuning Data Loaders:')
            print('----------------------------------------')

        train_loader = InstructDataLoader(
            batch_size=batch_size,
            sequence_length=sequence_length,
            is_master_process=is_master_process,
            process_rank=process_rank,
            num_processes=num_processes,
            data_root=data_root,
            split='train',
            use_shuffle=True,
            pad_id=pad_id,
            drop_last=True,
        )
        val_loader = InstructDataLoader(
            batch_size=batch_size,
            sequence_length=sequence_length,
            is_master_process=is_master_process,
            process_rank=process_rank,
            num_processes=num_processes,
            data_root=data_root,
            split='val',
            use_shuffle=False,
            pad_id=pad_id,
            drop_last=False,
        )

    return train_loader, val_loader
