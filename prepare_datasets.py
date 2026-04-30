import argparse
import json

from config import config
from datasets_preparation import (
    prepare_pretraining_dataset,
    prepare_instruct_dataset,
    prepare_dpo_dataset,
    prepare_hellaswag_dataset,
    prepare_winogrande_dataset
)


def load_custom_dataset_mix(mix_file_path):
    if mix_file_path is None:
        return None 
    with open(mix_file_path, 'r') as file:
        return json.load(file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Datasets Preparation Script Options')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--hellaswag', action='store_true', help='Prepare HellaSwag eval dataset')
    group.add_argument('--winogrande', action='store_true', help='Prepare WinoGrande eval dataset')
    group.add_argument('--pretraining', action='store_true', help='Prepare pretraining dataset')
    group.add_argument('--instruct', action='store_true', help='Prepare instruct (SFT) dataset')
    group.add_argument('--dpo', action='store_true', help='Prepare DPO (Direct Preference Optimization) dataset')

    parser.add_argument('--mix-file', type=str, default=None, help='Path to custom mix file')

    args = parser.parse_args()

    if (args.hellaswag or args.winogrande) and args.mix_file:
        parser.error('"--mix-file" is only supported for training datasets.')

    datasets_mix = load_custom_dataset_mix(args.mix_file)

    if args.hellaswag:
        prepare_hellaswag_dataset()
    elif args.winogrande:
        prepare_winogrande_dataset()
    elif args.pretraining:
        prepare_pretraining_dataset(datasets_mix=datasets_mix)
    elif args.instruct:
        prepare_instruct_dataset(datasets_mix=datasets_mix)
    elif args.dpo:
        prepare_dpo_dataset(datasets_mix=datasets_mix)
