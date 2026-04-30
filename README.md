# Gradient Garden

Gradient Garden is a research playground for model training, evaluation, and experimentation across architectures, benchmarks, and recipes.

The project began with a Llama-inspired transformer baseline and has evolved into a broader codebase for training, evaluation, and experimentation across modern machine learning models, distributed training workflows, and post-training methods.

## Current focus
- Distributed training
- Multi-stage model training and post-training workflows
- Research and experimentation across architectures, benchmarks, and training recipes

## Current capabilities

### Training and distributed execution
- single-GPU training
- multi-GPU / multi-node training
  - DDP
  - FSDP2 (~ZeRO3-like sharding)
- Gradient accumulation
- Mixed precision
  - BF16
  - FP16
  - FP32
- Early stopping
- Checkpoint save / load / resume
- Torch profiler integration
- Weights & Biases (W&B) integration

### Optimization
- AdamW
  - Fused if available on the device
- Muon
  - Separate max LR, min LR, and warmup settings from AdamW
  - Applied to matrix parameters
  - AdamW is applied to the remaining trainable parameters
- Cosine LR scheduling

### Training workflows
- Pretraining
- Supervised fine-tuning (SFT / instruct)
- Direct Preference Optimization (DPO)
- Optional distillation support
  - Teacher models are currently loaded from the Hugging Face Hub, but this can be adapted to other sources if needed

### Model features
- LoRA
- Mixture of Experts (MoE)
  - Reuses the existing FF module for expert MLPs
  - Includes load balancing and z-loss
- KV cache for autoregressive decoding

### Evaluation
- Shared multiple-choice evaluation path
  - HellaSwag
  - WinoGrande
- Additional benchmarks can be added through the same multiple-choice evaluation flow

## Notes
- The project is currently focused on CUDA-based training workflows.
- The initial model implementation was inspired by the Llama family, but the codebase is not intended to stay tied to Llama specifically.
- The project can be adapted to other datasets and model architectures.
- By default, the project uses a Hugging Face tokenizer.
- It also supports a `tiktoken`-based tokenizer with a configuration similar to the Llama 3 tokenizer, but the local BPE/tokenizer file is **not** included in this repository and must be provided separately.

## Project structure
- `datasets_preparation/` Components used for downloading, preparing, and tokenizing datasets.
- `engine/` Trainer and runtime core components.
- `evals/` Shared evaluation loading and scoring utilities.
  - Multiple choice evals: HellaSwag, WinoGrande.
- `examples/` Templates for the `.env` config and dataset mix.
- `metrics/` Utilities for metric aggregation.
- `tasks/` Groups the training tasks.
- `tests/` Groups tests for different components.
- `checkpoints.py` Logic to handle checkpointing.
- `config.py` Defines the main config and environment variables that are to be extracted from `.env`.
- `dataloaders.py` Dataloader logic for sampling and distributing data.
- `ddp_utils.py` Contains the main logic to set up the PyTorch DDP (Distributed Data Parallel) and FSDP2 (Fully Sharded Data Parallel).
  - PyTorch DDP [here](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- `distillation_utils.py` Logic for distillation loss.
- `dpo_utils.py` Logic for DPO loss.
- `generate.py` Logic for sampling and text generation.
- `kv_cache.py` KV cache implementation.
- `logger.py` Simple reusable logger.
- `lora.py` LoRA module that handles the model modification. Rank, alpha, dropout and target modules can be configured in the `.env` file.
- `lr_schedulers.py` Stores learning rate schedulers. At the moment, it includes a cosine scheduler.
- `model.py` Current main model implementation.
- `prepare_datasets.py` Entry point for data downloading and preparation.
- `test_prompts.json` JSON with the list of input prompts to try during training. The expected keys in the JSON (as provided in the file) are "pretrain", "instruct", "dpo".
- `tokenizer.py` Provides the tokenizer abstraction used by the project and supports two backends:
  - `TikTokenizer`: loads tiktoken BPE weights from a local file path and configures the special tokens used by the project.
  - `HFTokenizer`: loads a tokenizer from Hugging Face via `AutoTokenizer.from_pretrained(...)` and aligns the required special tokens (`bos`, `eos`, headers, `eot`, `pad`).
  - `init_tokenizer(...)` selects the backend based on configuration (`HUGGINGFACE_TOKENIZER`).
- `train.py` Entry point for training runs.
- `wandb_utils.py` A wrapper for Weights & Biases.
  - Weights & Biases [here](https://wandb.ai/site/)

## Setup
- Create a python environment. Example with conda: `conda create -n my_env python=3.11`;
- Activate the environment and run: `pip install -r requirements.txt`;
- Download and prepare the data:
  - Evals:
    - HellaSwag: `python prepare_datasets.py --hellaswag`
    - WinoGrande: `python prepare_datasets.py --winogrande`
  - Training and validation:
    - pretraining: `python prepare_datasets.py --pretraining`
    - instruct: `python prepare_datasets.py --instruct`
    - dpo: `python prepare_datasets.py --dpo`
  - **NOTE**: All the target paths can be modified in the `.env` file. (Check config.py for more details.)
  - The training dataset preparation commands also support a custom mix file by passing `--mix-file <file_path>`. Check `examples/pretraining_data_mix.example.json` for an example. Local custom mix files should use the `.local.json` suffix, for example `pretraining_debug.local.json`, so they are ignored by Git. If no `--mix-file` is provided, the built-in default mix for that stage is used.
    - The default mix can be found in `datasets_preparation/default_mixes.py`
  
- (OPTIONAL) Setup your Weights & Biases API key:
  - Set `WANDB_API_KEY` environment variable if you want to log the progress there.

- **NOTE:** For some scenarios you might need to also pass your Hugging Face API token `HF_TOKEN`. E.g.: If performing knowledge distillation and the teacher model requires access permissions.

## Configuring and training:
The project expects a `.env` file at the repository root. Check `examples/.env.example` and use it as a template.
The file `config.py` defines all the environment variables required.
- Modify it according to your experiment needs (e.g., model architecture, hyperparameters, Weights & Biases settings, checkpointing, etc.).

**NOTE:** Values in `.env` override the corresponding defaults defined in `config.py`.

### Common flags
`train.py` accepts some flags that are useful to load a checkpoint or override some properties:

```bash
  --pretrain_checkpoint <file>   # Resume pre-training run
  --instruct_checkpoint <file>   # Resume SFT run
  --dpo_checkpoint <file>        # Resume DPO run
  --reset-optimizers             # Ignore stored optimizer(s) state
  --start-step <N>               # Override internal step counter
```
**NOTE:** The checkpoint paths need to be set in the `.env` file. See `config.py` for details.

### Running the Training
- To train on **single-GPU**, run:
    ```bash
    python train.py
    ```

- To train on **multi-GPU** run:
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
    - NOTE: When loading an instruct checkpoint, use `--instruct_checkpoint` instead. This will also load the optimizer(s) state and resume from the stored step. You can reset the optimizer(s) with the flag `--reset-optimizers` and set the start step with the flag `--start-step`. E.g.: `--start-step 10`

- To train on multiple nodes with **1 or more GPUs per node**, configure each node as follows:
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

## Using Torch Profiler
Details on the environment variables suggested in `examples/.env.example` can be found [here](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).

## Running tests
From the root folder:
```bash
pytest
```

## Contributions
Feel free to reach out if interested in contributing!

## Third-party assets and licenses
Tokenizer files, model weights, and datasets obtained from third parties are **not** included in this repository unless explicitly stated, and may be subject to their own licenses and terms.

## License
This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.

## Citation
Please cite this project if it was useful in your work:

```bibtex
@software{rui2024gradientgarden,
  author = {Rui Malheiro},
  title = {Gradient Garden},
  year = {2024},
  url = {https://github.com/ruimalheiro/gradient-garden}
}
```