import math
import torch
import torch.nn.functional as F

from torch import nn


class CustomLoRA(nn.Module):
    def __init__(self, target_layer, rank=16, alpha=8, matrixA_standard_deviation=0.02, dropout=0.0):
        super().__init__()
        self.weight = target_layer.weight
        self.bias = target_layer.bias
        self.rank = rank
        self.alpha = alpha
        self.matrixA_standard_deviation = matrixA_standard_deviation
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)

        # Because of how pytorch stores weighs E.g: if Layer(in, out), weights are stored (out, in) which is the target.
        # We want BA so given layer (x, y) we want -> w(y, x) + (alpha/r) * (B(y, r) @ A(r, x)) note that B(y, r) @ A(r, x) has dim (y, x)
        self.A = nn.Parameter(torch.randn((rank, target_layer.in_features)) * matrixA_standard_deviation)
        self.B = nn.Parameter(torch.zeros((target_layer.out_features, rank)))

        # freeze original weights
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def forward(self, x):
        lora_weights = self.weight + self.scale * (self.B @ self.A)
        return F.linear(self.dropout(x), lora_weights, self.bias)

def apply_lora(
    model,
    device,
    target_modules=('wq', 'wk', 'wv', 'wo', 'w1', 'w3'),
    rank=16,
    alpha=8,
    dropout=0.0,
    is_master_process=True
):
    for name, module in model.named_modules():
        if any(name.endswith(t) for t in target_modules):
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = model.get_submodule(parent_name)

            lora_layer = CustomLoRA(module, rank=rank, alpha=alpha, dropout=dropout).to(device)

            setattr(
                parent_module,
                child_name,
                lora_layer
            )
    if is_master_process:
        print('\nLoRA applied with params:')
        print(f'- rank: {rank}')
        print(f'- alpha: {alpha}')
        print(f'- dropout: {dropout}')
