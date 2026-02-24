# Llama-style transformer and multi-node / multi-GPU training

This project implements an LLM in pure PyTorch. It also includes scripts for data preparation and multi-node / multi-GPU training.

This project mainly focuses on the following:
- Llama-like baseline architecture in pure PyTorch
- Multi-node / multi-GPU training
- Multiple training methods (SFT, distill, DPO, etc.)

**NOTE**: 
- The project can be adapted for other datasets.
- By default, the project uses a Hugging Face tokenizer.
- It also supports a tiktoken-based tokenizer with a configuration similar to Llama 3's tokenizer, but the local BPE/tokenizer file is **not** included in this repository and must be provided separately.

## Model
The model implementation was originally based on the Llama 3 architecture but later diverged due to changes and experimentation. The official project from Meta can be found [here](https://github.com/meta-llama/llama3).

## Supported features in the trainer and config options
- Multi-node / multi-GPU
  - DDP
  - FSDP2 (~ZeRO3)
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
- Torch profiler
- Mixture of Experts (MoE)
  - Uses the existing FF module for the MLPs
  - Load balancing + z-loss
- KV-Cache (1-token decoding)

## Instructions
- `checkpoints.py` Logic to handle checkpointing.
- `config.py` Defines the main config and environment variables that are to be extracted from `.env`.
- `data_preparation_utils.py` Contains logic to process a dataset into multiple shards.
- `dataloaders.py` Dataloaders logic to sample and distribute the data.
- `ddp_utils.py` Contains the main logic to set up the Torch DDP (Distributed Data Parallel) and FSDP2 (Fully Sharded Data Parallel).
- `distillation_utils.py` Logic for distillation loss.
- `dpo_utils.py` Logic for DPO loss.
- `generate.py` Logic for sampling and text generation.
- `hellaswag_utils.py` Contains the main logic to iterate, process and evaluate HellaSwag examples.
- The files `load_*_dataset.py` download and prepare the datasets to be used for the respective training stage. They load the datasets via `load_dataset` from the `datasets` HF package.
  - Each load script has an associated configuration file:
    - `hf_pretrain_datasets_mix.json`
    - `hf_instruct_datasets_mix.json`
    - `hf_dpo_datasets_mix.json`
- `logger.py` Simple reusable logger.
- `lora.py` LoRA module that handles the model modification. Rank, alpha, dropout and target modules can be configured in the `.env` file.
- `lr_schedulers.py` To store learning rate schedulers, for now just a cosine scheduler.
- `model_utils.py` Contains utils like model parameter count, clip grad and logging the task summary.
  - Torch DDP [here](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
  - Weights & Biases [here](https://wandb.ai/site/)
- `model.py` Implements the custom Llama architecture.
- `test_prompts.json` JSON with the list of input prompts to try during training. The expected properties in the JSON (as provided in the file) are "pretrain", "instruct", "dpo".
- `tokenizer.py` Provides the tokenizer abstraction used by the project and supports two backends:
  - `TikTokenizer`: loads tiktoken BPE weights from a local file path and configures the special tokens used by the project.
  - `HFTokenizer`: loads a tokenizer from Hugging Face via `AutoTokenizer.from_pretrained(...)` and aligns the required special tokens (`bos`, `eos`, headers, `eot`, `pad`).
  - `init_tokenizer(...)` selects the backend based on configuration (`HUGGINGFACE_TOKENIZER`).
- `train.py` Main file that contains the logic for the training process.
- `wandb_utils.py` A wrapper for Weights & Biases.

## Setup
- Create a python environment. Example with conda: `conda create -n my_env python=3.11`;
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
  NOTE: The checkpoint paths need to be set in the `.env` file (check `config.py`)

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
  - Static
    - Ethernet
      ```bash
      export NCCL_IB_DISABLE=1
      export NCCL_SOCKET_NTHREADS=4
      export NCCL_NSOCKS_PERTHREAD=8
      ```
    - InfiniBand
      ```bash
      export NCCL_IB_DISABLE=0
      export NCCL_IB_HCA=$(ls /sys/class/infiniband | paste -sd, -)
      ```
    ```bash
    export OMP_NUM_THREADS=1
    export PYTHONUNBUFFERED=1
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export TORCH_DIST_BIND_ADDR=0.0.0.0 # only needed in master but no impact on the workers
    export NCCL_DEBUG=WARN

    NNODES=<NUMBER_OF_NODES>
    NPERNODE=<NUMBER_OF_GPUs>
    NODE_RANK=<NODE_RANK>
    MASTER_ADDR=<MASTER_NODE_MACHINE_IP>
    MASTER_PORT=<MASTER_NODE_MACHINE_PORT>

    # make sure we can find the correct NIC
    _IFACE=$(ip -o route get "$MASTER_ADDR" | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}')
    [ -n "$_IFACE" ] && [ "$_IFACE" != "lo" ] && export NCCL_SOCKET_IFNAME="$_IFACE"

    torchrun \
      --nnodes ${NNODES} \
      --nproc-per-node ${NPERNODE} \
      --node-rank ${NODE_RANK} \
      --master_addr ${MASTER_ADDR} \
      --master_port ${MASTER_PORT} \
      train.py
    ```  
  - Elastic 
    ```bash
    export OMP_NUM_THREADS=1
    export PYTHONUNBUFFERED=1
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export TORCH_DIST_BIND_ADDR=0.0.0.0 # only needed in master but no impact on the workers
    export NCCL_DEBUG=WARN

    NNODES=<NUMBER_OF_NODES>
    NPERNODE=<NUMBER_OF_GPUs>
    MASTER_ADDR=<MASTER_NODE_MACHINE_IP>
    MASTER_PORT=<MASTER_NODE_MACHINE_PORT>
    RDZV_EP="$MASTER_ADDR:$MASTER_PORT"
    RDZV_ID=<SOME_SHARED_JOB_NAME>

    # make sure we can find the correct NIC
    _IFACE=$(ip -o route get "$MASTER_ADDR" | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}')
    [ -n "$_IFACE" ] && [ "$_IFACE" != "lo" ] && export NCCL_SOCKET_IFNAME="$_IFACE"


    torchrun \
      --nnodes ${NNODES} \
      --nproc-per-node ${NPERNODE} \
      --rdzv-backend c10d \
      --rdzv-endpoint ${RDZV_EP} \
      --rdzv-id ${RDZV_ID} \
      train.py
    ```
  - **NOTE:** The same command needs to be run on all nodes

- More details on torchrun [here](https://pytorch.org/docs/stable/elastic/run.html)
- More details on NCCL [here](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#environment-variables)

## Using torch profiler
Details on the environment variables suggested in `.env.example` can be found [here](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).

If using tensorboard, in a different shell start:
```bash
tensorboard --logdir <LOG_PATH> --bind_all
```

## Run tests
From the root folder:
```bash
pytest
```

## Third-party assets and licenses
Tokenizer files, model weights, and datasets obtained from third parties are **not** included in this repository unless explicitly stated, and may be subject to their own licenses and terms.

## License
This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.

## Citation
Please cite this project if it was useful in your work:
```
@software{rui2024trainingcustomllama,
  author = {Rui Malheiro},
  title = {Llama-style transformer and multi-node / multi-GPU training},
  year = {2024},
  url = {https://github.com/ruimalheiro/training-custom-llama}
}
```
