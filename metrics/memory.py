import torch

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryUsageMetrics:
    peak_allocated_mb: float
    peak_reserved_mb: float
    current_allocated_mb: float
    current_reserved_mb: float

def reset_memory_usage_metrics():
    if not torch.cuda.is_available():
        return
    torch.cuda.reset_peak_memory_stats()

def compute_memory_usage_metrics(ddp_local_rank):
    return MemoryUsageMetrics(
        peak_allocated_mb=torch.cuda.max_memory_allocated(ddp_local_rank) / 1024**2,
        peak_reserved_mb=torch.cuda.max_memory_reserved(ddp_local_rank) / 1024**2,
        current_allocated_mb=torch.cuda.memory_allocated(ddp_local_rank) / 1024**2,
        current_reserved_mb=torch.cuda.memory_reserved(ddp_local_rank) / 1024**2
    )
