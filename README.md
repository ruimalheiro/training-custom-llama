# Training custom Llama 3 in a node with multi gpu

This is a small project for educational purposes that combines many learnings and implements a simplified version of the Llama 3 architecture in PyTorch. It also includes scripts for multi-GPU training with Torch DDP.

Main concepts covered in this project:
- Running distributed training jobs using PyTorch DDP.
- Implementing a version of Llama3 and configuring the tokenizer.

**NOTE**: 
- The project can be adapted for other datasets.
- We use a pretrained tokenizer as training the tokenizer is not the main focus of the project.


### Model
The official project from Meta can be found [here](https://github.com/meta-llama/llama3).

The implementation in this project is a bit different but the core ideas are the same. To verify it is correctly implemented, the original pretrained weights can be loaded.


### Supported features in the trainer and config options
- Single / multi GPU (1 node)
- Pretraining
- Instruct fine-tuning
- Model distillation
  - Note that for the moment the teacher model is loaded as a checkpoint from HF. (You can however adapt it to run any model) 
- Saving / loading checkpoint
- Weights and Bias integration
- Early stopping
- Gradient accumulation
- Start from step X / Reset optimizer
  - Note: At the moment these need to be passed as flags when running the script.
- Hellaswag for extra validation


## Instructions
- `load_pretrain_dataset.py` and `load_instruct_dataset.py` Downloads and prepares the dataset to be used for pretraining / instruct fine tuning. It loads the dataset via `load_dataset` from the `datasets` HF package.
- `data_preparation_utils.py` Contains the main logic to process the dataset into multiple shards that will be used by the dataloader.
- `dataloaders.py` Dataloaders logic to sample and distribute the data correctly.
- `hellaswag_utils.py` Contains the main logic to iterate, process and evaluate hellaswag examples.
- `model.py` Implements the custom llama3
- `tokenizer.py` Uses tiktoken to setup the tokenizer for llama3 with some changes for encoding / decoding and the special tokens needed.
  - `tokenizer.model` Required to load the pretrained tokenizer. 
- `model_utils.py` Contains the main logic to setup the torch DDP (Distributed Data Parallel), a wrapper for Weights and Bias and also other quality of life functions to manage checkpoints, save and load the model.
  - Torch DDP [here](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
  - Weight and Bias [here](https://wandb.ai/site/)
- `train.py` Main file that contains the logic for the training process.

## Setup
### Part 1 - Setup the project, download and prepare the data:
- Create a python environment, ideally running python 3.10.13. Example with conda: `conda create -n my_env python=3.10.13`;
- Activate the environment and run: `pip install -r requirements.txt`;
- Download and prepare the data:
  - For hellaswag run:
  ```
  python load_hellaswag_dataset.py
  ```
  - For the pretraining dataset run: 
  ```
  python load_pretrain_dataset.py
  ```
  - For the instruction dataset run: 
  ```
  python load_instruct_dataset.py
  ```
    - **NOTE**: Both `load_pretrain_dataset` and `load_instruct_dataset` expect the dataset structure to be in a particular format and should be modified as necessary. (E.g.: If using different datasets). For example the instruct dataset expects the chat format.
  
- (OPTIONAL) Setup your Weights and Bias API key:
  - Set `WANDB_API_KEY` environment variable.

### Part 2 - Configuring & Running Training:
The project expects a `.env` file to be created. Check `.env.example` and use it as a template.
The file `config.py` defines all the environment variables required.
- Modify it according to your experiment needs (e.g., model architecture, hyperparameters, Weights & Biases settings, checkpointing, etc.).

#### **Running the Training**
- To train on **1 GPU**, run:
    ```
    python train.py
    ```
- To train on **1 or more GPUs** run:
    ```
    export OMP_NUM_THREADS=1 && torchrun --standalone --nproc_per_node <NUMBER_OF_GPUs> train.py
    ```
    - More details on torchrun [here](https://pytorch.org/docs/stable/elastic/run.html)
- To load a checkpoint and continue training, pass the flag to any of the above commands. E.g.:
    ```
    export OMP_NUM_THREADS=1 && torchrun --standalone --nproc_per_node <NUMBER_OF_GPUs> train.py --checkpoint <CHECKPOINT_FILE_NAME>
    ```
    - NOTE: This will also load the optimizer and the step where it was. You can reset the optimizer with the flag `--reset-optimizer` and set the start step with the flag `--start-step`. E.g.: `--start-step 10`
