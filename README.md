# Training custom Llama 3 in a node with multi gpu

This is a small project for educational purposes that combines many learnings and implements a simplified version of the Llama 3 architecture in PyTorch. It also includes scripts for multi-GPU training with Torch DDP.

Main concepts covered in this project:
- Running distributed training jobs using PyTorch DDP.
- Using datasets such as FineWeb-edu and HellaSwag.
- Implementing a basic version of Llama3 and configuring the tokenizer.

**NOTE**: Could be adapted for other datasets / We are not training the tokenizer here.

### Datasets Used
This project uses the following datasets:
- **FineWeb-edu** (10B token subset): A high-quality filtered web dataset designed for educational purposes.  
- **HellaSwag** (validation split): A commonsense reasoning dataset, used here for evaluating the model’s understanding capabilities.

### Model
The official project from Meta can be found [here](https://github.com/meta-llama/llama3).

The implementation in this project is a bit different but the core ideas are the same. To verify it is correctly implemented, the original pretrained weights can be loaded.


## Instructions
- `load_fineweb_dataset.py` Downloads and prepares a sample of 10B tokens from Fineweb-edu.
  - Dataset can be found [here](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).
- `load_hellaswag_dataset.py` Downloads the HellaSwag validation split.
  - Dataset can be found [here](https://github.com/rowanz/hellaswag)
- `hellaswag_utils.py` Contains the main logic to iterate, process and evaluate hellaswag examples.
- `model.py` Implements the custom llama3
- `tokenizer.py` Uses tiktoken to setup the tokenizer for llama3 with some changes for encoding / decoding and the special tokens needed.
- `dataloaders.py` Dataloaders logic to sample and distribute the data correctly.
- `model_utils.py` Contains the main logic to setup the torch DDP (Distributed Data Parallel), a wrapper for Weights and Bias and also other quality of life functions to manage checkpoints, save and load the model.
  - Torch DDP [here](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
  - Weight and Bias [here](https://wandb.ai/site/)
- `tokenizer.model` Required to load the pretrained tokenizer (Aligned with llama3)
- `train.py` Main file to configure the model architecture and initialize the training job.

## Setup
### Part 1 - Setup the project, download and prepare the data:
- Create a python environment, ideally running python 3.10.13. Example with conda: `conda create -n my_env python=3.10.13`;
- Activate the environment and run: `pip install -r requirements.txt`;
- Download and prepare the data:
  - Run:
  ```
  python load_hellaswag_dataset.py
  ```
  - Run 
  ```
  python load_fineweb_dataset.py
  ```
    - NOTE: Fineweb download and preparation can take around an hour (depending on the resources)
- (OPTIONAL) Setup your Weights and Bias API key:
  - Set `WANDB_API_KEY` environment variable.

### Part 2 - Configuring & Running Training:
Now we should be ready to start a pre-training job:
- In the `train.py` file there is a section between:
  - ############################# CONFIGURATION #################################      
  - ...
  - #############################################################################
- Modify it according to your experiment needs (e.g., model architecture, hyperparameters, Weights & Biases settings, checkpointing, etc.).

#### **Running the Training**
- To train on **1 GPU**, run:
    ```
    python train.py
    ```
- To train on **1 or more GPUs** run:
    ```
    torchrun --standalone --nproc_per_node <NUMBER_OF_GPUs> train.py
    ```
    - More details on torchrun [here](https://pytorch.org/docs/stable/elastic/run.html)
