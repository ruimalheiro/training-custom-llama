import argparse

from config import config
from engine import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Script Options')
    parser.add_argument('--pretrain_checkpoint', type=str, default=None, help='Pretrain checkpoint to load.')
    parser.add_argument('--instruct_checkpoint', type=str, default=None, help='Instruct checkpoint to load.')
    parser.add_argument('--dpo_checkpoint', type=str, default=None, help='DPO checkpoint to load.')
    parser.add_argument('--reset-optimizers', action='store_true', help='Reset the optimizers state when loading a checkpoint.')
    parser.add_argument('--start-step', type=int, default=None, help='Starting step number for training.')

    args = parser.parse_args()

    trainer = Trainer(
        config = config,
        args=args
    )
    trainer.train()
