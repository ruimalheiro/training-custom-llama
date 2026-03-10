import torch
import torch.distributed as dist


def get_value_as_float32_tensor(value, device):
    if not torch.is_tensor(value):
        return torch.tensor(float(value), device=device, dtype=torch.float32)
    return value.detach().to(device=device, dtype=torch.float32)

def accumulate_weighted_metrics(weight, metrics, metrics_sum_acc, metrics_weights_acc, device):
    weight = weight.detach().to(device=device, dtype=torch.float32)

    for key, value in metrics.items():
        value = get_value_as_float32_tensor(value, device)

        if key not in metrics_sum_acc:
            metrics_sum_acc[key] = torch.tensor(0.0, device=device)
            metrics_weights_acc[key] = torch.tensor(0.0, device=device)

        metrics_sum_acc[key] += value * weight
        metrics_weights_acc[key] += weight

def combine_weighted_metrics(metrics_sum_acc, metrics_weights_acc, ddp):
    result = {}
    for key in metrics_sum_acc:
        if ddp:
            dist.all_reduce(metrics_sum_acc[key], op=dist.ReduceOp.SUM)
            dist.all_reduce(metrics_weights_acc[key], op=dist.ReduceOp.SUM)

        result[key] = (metrics_sum_acc[key] / metrics_weights_acc[key].clamp(min=1.0)).item()
    
    return result
