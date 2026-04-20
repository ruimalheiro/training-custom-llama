from logger import logger
from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule
)


class NoOpProfiler:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def step(self):
        pass

def init_torch_profiler_context(config, distributed_ctx):
    if not config.torch_profiler_enabled or not distributed_ctx.is_master_process:
        return NoOpProfiler()

    def trace_handler(prof):
        logger.info(
            prof.key_averages().table(
                sort_by='cuda_time_total',
                row_limit=10
            ),
            True
        )

    logger.info('\nWARN: Torch profiler is enabled!\n')
    return (
        profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA
            ],
            record_shapes=True,
            profile_memory=True,
            acc_events=True,
            schedule=schedule(
                skip_first=config.torch_profiler_schedule_skip_first,
                wait=config.torch_profiler_schedule_wait,
                warmup=config.torch_profiler_schedule_warmup,
                active=config.torch_profiler_schedule_active,
                repeat=config.torch_profiler_schedule_repeat
            ),
            on_trace_ready=trace_handler
        )
    )
