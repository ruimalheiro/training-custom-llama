import torch

from dataclasses import dataclass
from lora import is_lora_parameter_name


@dataclass
class PartitionedParameters:
    decay: list[tuple[str, torch.nn.Parameter]]
    nodecay: list[tuple[str, torch.nn.Parameter]]
    lora: list[tuple[str, torch.nn.Parameter]]
    num_decay_params: int
    num_nodecay_params: int
    num_lora_params: int
    total_trainable_params: int

@dataclass
class OptimizersParamGroups:
    optimizers_param_groups: dict[str, list[dict]]
    partitioned_parameters: PartitionedParameters

def partition_trainable_parameters(model) -> PartitionedParameters:
    lora = []
    decay = []
    nodecay = []
    for name, parameter in model.get_named_trainable_parameters():
        if is_lora_parameter_name(name):
            lora.append((name, parameter))
        elif parameter.dim() >= 2:
            decay.append((name, parameter))
        else:
            nodecay.append((name, parameter))

    num_decay_params = sum(p.numel() for _, p in decay)
    num_nodecay_params = sum(p.numel() for _, p in nodecay)
    num_lora_params = sum(p.numel() for _, p in lora)
    total_trainable_params = num_decay_params + num_nodecay_params + num_lora_params

    return PartitionedParameters(
        decay=decay,
        nodecay=nodecay,
        lora=lora,
        num_decay_params=num_decay_params,
        num_nodecay_params=num_nodecay_params,
        num_lora_params=num_lora_params,
        total_trainable_params=total_trainable_params
    )

def build_optimizers_param_groups(partitioned_params, weight_decay, lora_weight_decay=0.0) -> OptimizersParamGroups:
    decay = partitioned_params.decay
    nodecay = partitioned_params.nodecay
    lora = partitioned_params.lora

    adamw_groups = []

    if decay:
        adamw_groups.append({'params': [p for _, p in decay], 'weight_decay': weight_decay})

    if nodecay:
        adamw_groups.append({'params': [p for _, p in nodecay], 'weight_decay': 0.0})

    if lora:
        adamw_groups.append({'params': [p for _, p in lora], 'weight_decay': lora_weight_decay})

    return OptimizersParamGroups(
        optimizers_param_groups = {
            'adamw': adamw_groups
        },
        partitioned_parameters=partitioned_params
    )
