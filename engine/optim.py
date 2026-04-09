import torch

from torch.optim import AdamW, Muon
from dataclasses import dataclass, field
from lora import is_lora_parameter_name
from typing import Literal
from logger import logger


@dataclass
class NamedParam:
    name: str
    param: torch.nn.Parameter

@dataclass
class ParameterBuckets:
    matrix: list[NamedParam] = field(default_factory=list)
    scalar: list[NamedParam] = field(default_factory=list)
    embedding: list[NamedParam] = field(default_factory=list)
    output: list[NamedParam] = field(default_factory=list)
    adapter: list[NamedParam] = field(default_factory=list)

    num_matrix_params: int = 0
    num_scalar_params: int = 0
    num_embedding_params: int = 0
    num_output_params: int = 0
    num_adapter_params: int = 0
    total_trainable_params: int = 0

@dataclass
class ParameterGroupPlan:
    optimizer_kind: Literal['adamw', 'muon']
    group_name: str
    params: list[torch.nn.Parameter]
    param_names: list[str]
    weight_decay: float
    lr_scale: float = 1.0

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.params)

@dataclass
class AdamWPlan:
    lr: float
    weight_decay: float
    betas: tuple[float, float]
    optimizer_name: Literal['adamw'] = 'adamw'
    groups: list[ParameterGroupPlan] = field(default_factory=list)

@dataclass
class MuonPlan:
    lr: float
    weight_decay: float
    momentum: float
    optimizer_name: Literal['muon'] = 'muon'
    adjust_lr_fn: Literal['original', 'match_rms_adamw'] = 'match_rms_adamw'
    groups: list[ParameterGroupPlan] = field(default_factory=list)

@dataclass
class OptimizerPlan:
    parameter_buckets: ParameterBuckets
    adamw: AdamWPlan | None = None
    muon: MuonPlan | None = None

@dataclass
class Optimizers:
    adamw: torch.optim.Optimizer | None = None
    muon: torch.optim.Optimizer | None = None

def get_input_output_embeddings_ids(model):
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    input_embeddings_ids = set()
    output_embeddings_ids = set()
    for param in input_embeddings.parameters():
        input_embeddings_ids.add(id(param))
    for param in output_embeddings.parameters():
        output_embeddings_ids.add(id(param))
    return input_embeddings_ids, output_embeddings_ids

def classify_trainable_parameters(model) -> ParameterBuckets:
    named_trainable_parameters = model.get_named_trainable_parameters()
    input_embeddings_ids, output_embeddings_ids = get_input_output_embeddings_ids(model)

    buckets = ParameterBuckets()
    seen_param_ids = set()

    for name, param in named_trainable_parameters:
        pid = id(param)
        if not param.requires_grad or pid in seen_param_ids:
            continue
        seen_param_ids.add(pid)
        named_param = NamedParam(name, param)

        if is_lora_parameter_name(name):
            buckets.adapter.append(named_param)
        elif pid in input_embeddings_ids:
            buckets.embedding.append(named_param)
        elif pid in output_embeddings_ids:
            buckets.output.append(named_param)
        elif param.dim() == 2:
            buckets.matrix.append(named_param)
        else:
            buckets.scalar.append(named_param)

    buckets.num_adapter_params = sum(x.param.numel() for x in buckets.adapter)
    buckets.num_embedding_params = sum(x.param.numel() for x in buckets.embedding)
    buckets.num_output_params = sum(x.param.numel() for x in buckets.output)
    buckets.num_matrix_params = sum(x.param.numel() for x in buckets.matrix)
    buckets.num_scalar_params = sum(x.param.numel() for x in buckets.scalar)
    buckets.total_trainable_params = (
        buckets.num_adapter_params +
        buckets.num_embedding_params +
        buckets.num_output_params +
        buckets.num_matrix_params +
        buckets.num_scalar_params
    )

    logger.info(f'\nOptimizers')
    logger.info('----------------------------------------')
    logger.info(f'num trainable parameters: {buckets.total_trainable_params:,}')
    logger.info(f'num embedding parameters: {buckets.num_embedding_params:,}')
    logger.info(f'num matrix parameters: {buckets.num_matrix_params:,}')
    logger.info(f'num scalar parameters: {buckets.num_scalar_params:,}')
    logger.info(f'num output parameters: {buckets.num_output_params:,}')
    logger.info(f'num adapter (lora) parameters: {buckets.num_adapter_params:,}')


    return buckets

def extract_group(optimizer_kind, group_name, named_params, weight_decay, lr_scale=1.0):
    if not named_params:
        return None
    return ParameterGroupPlan(
        optimizer_kind=optimizer_kind,
        group_name=group_name,
        params=[x.param for x in named_params],
        param_names=[x.name for x in named_params],
        weight_decay=weight_decay,
        lr_scale=lr_scale
    )

def build_adamw_groups(config, parameter_buckets: ParameterBuckets) -> list[ParameterGroupPlan]:
    adamw_groups: list[ParameterGroupPlan] = []

    def _add(group):
        if group is not None:
            adamw_groups.append(group)

    _add(extract_group(optimizer_kind='adamw', group_name='adapter', named_params=parameter_buckets.adapter, weight_decay=0.0))
    _add(extract_group(optimizer_kind='adamw', group_name='embedding', named_params=parameter_buckets.embedding, weight_decay=config.adamw_weight_decay))
    _add(extract_group(optimizer_kind='adamw', group_name='output', named_params=parameter_buckets.output, weight_decay=config.adamw_weight_decay))
    if not config.use_muon:
        _add(extract_group(optimizer_kind='adamw', group_name='matrix', named_params=parameter_buckets.matrix, weight_decay=config.adamw_weight_decay))
    _add(extract_group(optimizer_kind='adamw', group_name='scalar', named_params=parameter_buckets.scalar, weight_decay=0.0))

    return adamw_groups

def build_muon_groups(config, parameter_buckets: ParameterBuckets) -> list[ParameterGroupPlan]:
    muon_groups: list[ParameterGroupPlan] = []

    def _add(group):
        if group is not None:
            muon_groups.append(group)

    _add(extract_group(optimizer_kind='muon', group_name='matrix', named_params=parameter_buckets.matrix, weight_decay=config.muon_weight_decay))

    return muon_groups

def build_optimizer_plan(config, parameter_buckets: ParameterBuckets) -> OptimizerPlan:
    adamw_groups = build_adamw_groups(config, parameter_buckets)
    adamw_plan = AdamWPlan(
        lr=config.adamw_max_lr,
        weight_decay=config.adamw_weight_decay,
        betas=config.adamw_betas,
        groups=adamw_groups
    )

    muon_plan = None
    if config.use_muon:
        muon_groups = build_muon_groups(config, parameter_buckets)
        if muon_groups:
            muon_plan = MuonPlan(
                lr=config.muon_max_lr,
                weight_decay=config.muon_weight_decay,
                momentum=config.muon_momentum,
                groups=muon_groups
            )
        else:
            logger.info('Muon requests but no matrix params were found. Will use AdamW only.')

    return OptimizerPlan(
        parameter_buckets=parameter_buckets,
        adamw=adamw_plan,
        muon=muon_plan
    )

def build_optimizers(config, optimizer_plan: OptimizerPlan) -> Optimizers:
    optimizers = Optimizers()
    logger.info('\nBuilding Optimizers')
    logger.info('----------------------------------------')
    if optimizer_plan.adamw:
        logger.info('initializing AdamW')
        adamw_groups = []
        for group in optimizer_plan.adamw.groups:
            adamw_groups.append({
                'params': group.params,
                'param_names': group.param_names,
                'weight_decay': group.weight_decay,
                'lr': optimizer_plan.adamw.lr * group.lr_scale
            })

        optimizers.adamw = AdamW(
            params=adamw_groups,
            lr=optimizer_plan.adamw.lr,
            betas=optimizer_plan.adamw.betas,
            fused=config.adamw_use_fused
        )
        logger.info('AdamW ready')
    if optimizer_plan.muon:
        logger.info('initializing Muon')
        muon_groups = []
        for group in optimizer_plan.muon.groups:
            muon_groups.append({
                'params': group.params,
                'param_names': group.param_names,
                'weight_decay': group.weight_decay,
                'lr': optimizer_plan.muon.lr * group.lr_scale
            })

        optimizers.muon = Muon(
            params=muon_groups,
            lr=optimizer_plan.muon.lr,
            weight_decay=optimizer_plan.muon.weight_decay,
            adjust_lr_fn=optimizer_plan.muon.adjust_lr_fn,
            momentum=optimizer_plan.muon.momentum
        )
        logger.info('Muon ready')

    return optimizers

def move_optimizer_state_to_param_device(optimizer):
    # This is to ensure the optimiser state respect the device for all params
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device=p.device)
