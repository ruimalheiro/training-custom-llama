import argparse

from config import config
from datasets_preparation import prepare_pretraining_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Datasets Preparation Script Options')
    parser.add_argument('--pretraining', action='store_true', help='Prepare pretraining dataset')

    # parser.add_argument('--x', type=str, default=None, help='')
    # parser.add_argument('--x', type=str, default=None, help='')
    # parser.add_argument('--x', action='store_true', help='')
    # parser.add_argument('--x', type=int, default=None, help='')

    args = parser.parse_args()

    if args.pretraining:
        prepare_pretraining_data(
            config=config
        )
