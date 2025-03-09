# Training custom llama3 on a multi gpu node

This a small project for educational purpose that combines many learnings and implements a simplified version of llama3 architecture in PyTorch, in combination with the scripts for multi gpu training.

The official project from Meta can be found [here](https://github.com/meta-llama/llama3).


## Instructions
- `load_fineweb_dataset.py` Dowloads and prepares a sample of 10B tokens from Fineweb-edu.
  - Dataset can be found [here](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).
- `load_hellaswag_dataset.py` Downloads the HellaSwag validation split.
  - Dataaset can be found [here](https://github.com/rowanz/hellaswag)
- `hellaswag_utils.py` Contains the main logic to iterate, process and evaluate hellaswag examples.
- `model.py` Implements the custom llama3
- `tokenizer.py` Uses tiktoken to setup the tokenizer for llama3 with some changes for encoding / decoding and the special tokens needed.
- `dataloaders.py` Dataloaders logic to sample and distribute the data correctly.
- `model_utils.py` Contain the main logic to setup the torch DDP (Distributed Data Parallel), a wrapper for Weights and Bias and also other quality of life functions to manage checkpoints, save and load the model.
  - Torch DDP [here](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
  - Weight and Bias [here](https://wandb.ai/site/)
- `tokenizer.model` Required to load the pretrained tokenizer (Aligned with llama3)
- `train.py` Main file to configure the model architecture and initialize the training job.

## Setup
### Part 1:
- Create a python environment, ideally running python 3.10.13. Example with conda: `conda create -n my_env python=3.10.13`;
- Activate the environment and run: `pip install -r requirements.txt`;
- Download and prepare the data:
  - Run `python load_hellaswag_dataset.py`
  - Run `python load_fineweb_dataset.py`
    - NOTE: Fineweb download and preparation can take around an hour (depending on the resources)
- (OPTIONAL) Setup your Weights and Bias API key:
  - Use `WANDB_API_KEY` env

### Part 2:
Now we should be ready to start a pre-training job:
- In the `train.py` file there is a section between:
  - ############################# CONFIGURATION #################################      
  - ...
  - #############################################################################
- Modify the configuration in accordance with the experiment. E.g. model architecture, hyperparameters, wnb enabled/disabled, save enabled/disabled etc. Most of what needs to be changed is in that section.
- Once ready, you can:
  - run the project directly with (if only with 1 GPU):
    - `python train.py`
  - run with torch run (1 or more GPUs in the node):
    - `torchrun --standalone --nproc_per_node <NUMBER_OF_GPUs> train.py`
      - More details on torchrun [here](https://pytorch.org/docs/stable/elastic/run.html)
