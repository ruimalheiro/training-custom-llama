import os
import numpy as np
import torch
import random
import datasets
import torch.distributed as dist
import re

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from config import (
    TrainingStage,
    config
)


def load_tokens(filename):
    npt = np.load(filename, allow_pickle=False)
    if npt.dtype != np.int64:
        npt = npt.astype(np.int64)
    ptt = torch.from_numpy(npt)
    return ptt

class PretrainDataLoader:
    def __init__(self, batch_size, sequence_length, is_master_process, process_rank, num_processes, data_root, split, use_shuffle=False):
        self.B = batch_size
        self.S = sequence_length
        self.is_master_process = is_master_process
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.data_root = data_root
        assert split in {'train', 'val'}
        self.split = split
        self.use_shuffle = use_shuffle

        split_root = os.path.join(data_root, split)
        assert os.path.isdir(split_root), f'missing split dir: {split_root}'

        target_pattern = re.compile(r'^data_(\d+)\.npy$')
        valid_shards = []
        for file_name in os.listdir(split_root):
            match = target_pattern.match(file_name)
            if match:
                valid_shards.append((int(match.group(1)), os.path.join(split_root, file_name)))

        valid_shards.sort(key=lambda x: x[0])
        indexes = [i for i, _ in valid_shards]
        assert indexes == list(range(len(valid_shards))), f'Shard sequence is broken: {indexes}'

        self.shards = [shard_path for _, shard_path in valid_shards]
        assert self.shards, f'no shards found in split {split}'

        if is_master_process:
            print(f'found {len(self.shards)} shards for split {split}')

        self.reset()

    def calculate_max_tokens(self):
        def _calculate():
            total = 0
            for path in self.shards:
                shard = np.load(path, mmap_mode='r', allow_pickle=False)
                total += int(shard.shape[0])
                del shard
            return total

        if self.num_processes <= 1 or not dist.is_available() or not dist.is_initialized():
            return _calculate()

        total_tokens = None
        object_list_to_sync = [total_tokens]
        if self.is_master_process:
            object_list_to_sync[0] = _calculate()
        dist.broadcast_object_list(object_list_to_sync, src=0)
        total_tokens = int(object_list_to_sync[0])
        return total_tokens

    def sync_shuffle_shards(self):
        if not self.use_shuffle:
            return

        if self.num_processes <= 1 or not dist.is_available() or not dist.is_initialized():
            random.shuffle(self.shards)
            return

        # create the indexes and shuffle
        target_indexes = list(range(len(self.shards)))
        if self.is_master_process:
            random.shuffle(target_indexes)

        # synchronize the shuffle
        object_list_to_sync = [target_indexes]
        dist.broadcast_object_list(object_list_to_sync, src=0)
        order = object_list_to_sync[0]
        self.shards = [self.shards[i] for i in order]

    def reset(self):
        self.current_shard = 0
        self.sync_shuffle_shards()
        self.tokens = load_tokens(self.shards[self.current_shard])
        if torch.cuda.is_available():
            self.tokens = self.tokens.pin_memory()
        self.current_position = self.B * self.S * self.process_rank

    def state_dict(self):
        return {
            'shards' : list(self.shards),
            'current_shard' : self.current_shard,
            'current_position' : self.current_position
        }

    def load_state_dict(self, state):
        self.shards = state['shards']
        self.current_shard = state['current_shard']
        self.tokens = load_tokens(self.shards[self.current_shard])
        if torch.cuda.is_available():
            self.tokens = self.tokens.pin_memory()
        self.current_position = state['current_position']

    def next_batch(self):
        B, S = self.B, self.S
        buf = self.tokens[self.current_position : self.current_position+B*S+1]
        x = (buf[:-1]).view(B, S)
        y = (buf[1:]).view(B, S)
        self.current_position += B * S * self.num_processes
        if self.current_position + (B * S * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            if self.current_shard == 0:
                self.sync_shuffle_shards()
            self.tokens = load_tokens(self.shards[self.current_shard])
            if torch.cuda.is_available():
                self.tokens = self.tokens.pin_memory()
            self.current_position = self.B * self.S * self.process_rank
        return x, y

class InstructDataLoader:
    def __init__(self, batch_size, sequence_length, is_master_process, process_rank, num_processes, data_root, split, use_shuffle, pad_id, drop_last):
        self.B = batch_size
        self.S = sequence_length
        self.is_master_process = is_master_process
        self.process_rank = process_rank
        self.num_processes = num_processes

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
                padding_value=config.ignore_index
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
        def _calculate():
            return sum(self._dataloader.dataset.map(
                lambda ex: {'len': len(ex['input_ids'])},
                num_proc=config.number_of_cpu_processes,
                remove_columns=[],
                desc='Calculating number of tokens'
            )['len'])

        if self.num_processes <= 1 or not dist.is_available() or not dist.is_initialized():
            return _calculate()

        total_tokens = None
        object_list_to_sync = [total_tokens]
        if self.is_master_process:
            object_list_to_sync[0] = _calculate()
        dist.broadcast_object_list(object_list_to_sync, src=0)
        total_tokens = int(object_list_to_sync[0])
        return total_tokens

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

class DirectPreferenceOptimizationDataLoader:
    def __init__(self, batch_size, sequence_length, is_master_process, process_rank, num_processes, data_root, split, use_shuffle, pad_id, drop_last):
        self.B = batch_size
        self.S = sequence_length
        self.is_master_process = is_master_process
        self.process_rank = process_rank
        self.num_processes = num_processes

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
            prompt = [torch.tensor(e['prompt_input_ids']) for e in examples]
            chosen = [torch.tensor(e['chosen_input_ids']) for e in examples]
            rejected = [torch.tensor(e['rejected_input_ids']) for e in examples]

            prompt_padded = pad_sequence(
                prompt,
                batch_first=True,
                padding_value=int(pad_id)
            )
            if prompt_padded.size(1) > sequence_length:
                prompt_padded = prompt_padded[:, -sequence_length:]

            chosen_padded = pad_sequence(
                chosen,
                batch_first=True,
                padding_value=int(pad_id)
            )
            if chosen_padded.size(1) > sequence_length:
                chosen_padded = chosen_padded[:, -sequence_length:]

            rejected_padded = pad_sequence(
                rejected,
                batch_first=True,
                padding_value=int(pad_id)
            )
            if rejected_padded.size(1) > sequence_length:
                rejected_padded = rejected_padded[:, -sequence_length:]

            return prompt_padded, chosen_padded, rejected_padded

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
        def _calculate():
            return sum(self._dataloader.dataset.map(
                lambda ex: {'len': len(ex['prompt_input_ids']) + len(ex['chosen_input_ids']) + len(ex['rejected_input_ids'])},
                num_proc=config.number_of_cpu_processes,
                remove_columns=[],
                desc='Calculating number of tokens'
            )['len'])

        if self.num_processes <= 1 or not dist.is_available() or not dist.is_initialized():
            return _calculate()

        total_tokens = None
        object_list_to_sync = [total_tokens]
        if self.is_master_process:
            object_list_to_sync[0] = _calculate()
        dist.broadcast_object_list(object_list_to_sync, src=0)
        total_tokens = int(object_list_to_sync[0])
        return total_tokens

    def state_dict(self):
        return {'epoch': getattr(self.sampler, 'epoch', 0)}

    def load_state_dict(self, state):
        if 'epoch' not in state:
            if self.is_master_process:
                print('Warning - "epoch" not present, starting fresh dataloader (most likely transition from pretraining to DPO).')
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
    training_stage,
    pad_id=None
):
    if training_stage == TrainingStage.PRETRAIN:
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
            use_shuffle=False
        )
    elif training_stage == TrainingStage.INSTRUCT:
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
    elif training_stage == TrainingStage.DPO:
        assert pad_id is not None

        if is_master_process:
            print('Direct Preference Optimization Data Loaders:')
            print('----------------------------------------')

        train_loader = DirectPreferenceOptimizationDataLoader(
            batch_size=batch_size,
            sequence_length=sequence_length,
            is_master_process=is_master_process,
            process_rank=process_rank,
            num_processes=num_processes,
            data_root=data_root,
            split='train',
            use_shuffle=True,
            pad_id=pad_id,
            drop_last=False,
        )
        val_loader = DirectPreferenceOptimizationDataLoader(
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
    else:
        raise ValueError('Invalid training stage for dataloader')

    return train_loader, val_loader
