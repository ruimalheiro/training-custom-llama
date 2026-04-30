import argparse
import json

from config import config
from datasets_preparation import (
    prepare_pretraining_dataset,
    prepare_instruct_dataset
)


def load_custom_dataset_mix(mix_file_path):
    if mix_file_path is None:
        return None 
    with open(mix_file_path, 'r') as file:
        return json.load(file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Datasets Preparation Script Options')
    parser.add_argument('--pretraining', action='store_true', help='Prepare pretraining dataset')
    parser.add_argument('--instruct', action='store_true', help='Prepare instruct dataset')
    parser.add_argument('--mix-file', type=str, default=None, help='Path to custom mix file')

    args = parser.parse_args()

    if args.pretraining:
        prepare_pretraining_dataset(datasets_mix=load_custom_dataset_mix(args.mix_file))
    elif args.instruct:
        prepare_instruct_dataset(datasets_mix=load_custom_dataset_mix(args.mix_file))
