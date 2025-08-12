# Llama-style transformer and multi-node / multi-GPU training

This is a small project for educational purposes that implements an LLM in pure PyTorch. It also includes scripts for data preparation and multi-node / multi-GPU training with Torch DDP.

This repo demonstrates:
- Llama 3 like baseline architecture in pure PyTorch
- Multi-node training with PyTorch DDP
- Multiple training methods (SFT, distill, DPO)

**NOTE**: 
- The project can be adapted for other datasets.
- We use a pretrained tokenizer as training the tokenizer is not the main focus of the project. The default tokenizer is the tiktoken tokenizer with a configuration similar to Llama 3's tokenizer. A Hugging Face tokenizer can also be loaded (check the env.example) which I recommend for smaller experiments, especially when the vocabulary is small.

## Model
The model implementation is based on the Llama 3 architecture - official project from Meta can be found [here](https://github.com/meta-llama/llama3).

The implementation in this project is a bit different but the core ideas are the same. To verify it is correctly implemented, you can load Meta’s released weights for compatibility testing (not included here).
- For this test all model architecture hyperparameters need to match.

## Supported features in the trainer and config options
- Multi-node / multi-GPU
- Fused AdamW (if available)
- Pre-training
- Instruct fine-tuning (SFT)
- Model distillation
  - Note that for the moment the teacher model is loaded from the *Hugging Face Hub*. (You can however adapt it to run any model) 
- Saving / loading checkpoint
- Weights & Biases (W&B) integration
- Early stopping
- Gradient accumulation
- HellaSwag for extra validation
- LoRA configuration
- Direct Preference Optimization (DPO)

## Instructions
- `config.py` Defines the main config and environment variables that are to be extracted from `.env`.
- `data_preparation_utils.py` Contains logic to process a dataset into multiple shards.
- `dataloaders.py` Dataloaders logic to sample and distribute the data.
- `ddp_utils.py` Contains the main logic to set up the Torch DDP (Distributed Data Parallel).
- `distillation_utils.py` Logic for distillation loss.
- `dpo_utils.py` Logic for DPO loss.
- `hellaswag_utils.py` Contains the main logic to iterate, process and evaluate HellaSwag examples.
- The files `load_*_dataset.py` download and prepare the datasets to be used for the respective training stage. They load the datasets via `load_dataset` from the `datasets` HF package.
  - Each load script have an associated configuration file:
    - `hf_pretrain_datasets_mix.json`
    - `hf_instruct_datasets_mix.json`
    - `hf_dpo_datasets_mix.json`
- `lora.py` LoRA module that handles the model modification. Rank, alpha, dropout and target modules can be configured in the `.env` file.
- `lr_schedulers.py` To store learning rate schedulers, for now just a cosine scheduler.
- `model_utils.py` Contains functionality to manage checkpoints, save and load the model. Also contains a dict print helper.
  - Torch DDP [here](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
  - Weights & Biases [here](https://wandb.ai/site/)
- `model.py` Implements the custom Llama architecture. Also includes some extra functionality for text generation.
- `test_prompts.json` JSON with the list of input prompts to try during training. The expected properties in the JSON (as provided in the file) are "pretrain", "instruct", "dpo".
- `tokenizer.model` Required to load the pretrained tokenizer (unless using HF checkpoint).
- `tokenizer.py` Uses tiktoken to setup the tokenizer with some changes for encoding / decoding and the special tokens needed. If a checkpoint from HF is provided, it can load a specific tokenizer instead of using the one provided (`tokenizer.model`).
- `train.py` Main file that contains the logic for the training process.
- `wnb_utils` A wrapper for Weights & Biases.

## Setup
- Create a python environment. Example with conda: `conda create -n my_env python=3.10.10`;
- Activate the environment and run: `pip install -r requirements.txt`;
- Download and prepare the data (Example pretraining / instruct fine-tuning):
  - For HellaSwag run:
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
    All the target paths can be modified in the .env file. (Check config.py for more details.)
  
- (OPTIONAL) Setup your Weights & Biases API key:
  - Set `WANDB_API_KEY` environment variable if you want to log the progress there.

- **NOTE:** For some scenarios you might need to also pass your Hugging Face API token `HF_TOKEN`. E.g.: If performing knowledge distillation and the teacher model requires access permissions.

## Configuring & Running Training:
The project expects a `.env` file to be created. Check `.env.example` and use it as a template.
The file `config.py` defines all the environment variables required.
- Modify it according to your experiment needs (e.g., model architecture, hyperparameters, Weights & Biases settings, checkpointing, etc.).

**NOTE:** Values in `.env` override **all** defaults in `config.py`.

### Common flags
`train.py` accepts some flags that are useful to load a checkpoint or override some properties:

```bash
  --pretrain_checkpoint <file>   # Resume pre-training run
  --instruct_checkpoint <file>   # Resume SFT run
  --dpo_checkpoint <file>        # Resume DPO run
  --reset-optimizer              # Ignore stored optimizer state
  --start-step <N>               # Override internal step counter
```
  NOTE: The checkpoints paths need to be set in the `.env` file (check `config.py`)

### Running the Training
- To train on **Single GPU**, run:
    ```bash
    python train.py
    ```

- To train on **N GPUs** run:
    ```bash
    export OMP_NUM_THREADS=1

    torchrun \
      --standalone \
      --nproc_per_node <NUMBER_OF_GPUs> \
      train.py
    ```

- To load a checkpoint and continue training, pass the flag to any of the above commands. E.g.:
    ```bash
    export OMP_NUM_THREADS=1

    torchrun \
      --standalone \
      --nproc_per_node <NUMBER_OF_GPUs> \
      train.py --pretrain_checkpoint <CHECKPOINT_FILE_NAME>
    ```
    - NOTE: When loading an instruct checkpoint, use `--instruct_checkpoint` instead. This will also load the optimizer and the step where it was. You can reset the optimizer with the flag `--reset-optimizer` and set the start step with the flag `--start-step`. E.g.: `--start-step 10`

- To train on multiple nodes **1 or more GPUs**, for each node configure:
    ```bash
    export OMP_NUM_THREADS=1
    export PYTHONUNBUFFERED=1
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

    NNODES=<NUMBER_OF_NODES>
    NPERNODE=<NUMBER_OF_GPUs>
    RDZV_EP=<MASTER_NODE_MACHINE_IP>:<MASTER_NODE_MACHINE_PORT>
    RDZV_ID=<SOME_SHARED_JOB_NAME>

    torchrun \
      --nnodes ${NNODES} \
      --nproc-per-node ${NPERNODE} \
      --rdzv-backend c10d \
      --rdzv-endpoint ${RDZV_EP} \
      --rdzv-id ${RDZV_ID} \
      train.py
    ```
    - More details on torchrun [here](https://pytorch.org/docs/stable/elastic/run.html)
    - **NOTE:** The same command needs to be run on all nodes
