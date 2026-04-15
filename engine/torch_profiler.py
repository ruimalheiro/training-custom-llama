from logger import logger
from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler
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

    tensor_board = None
    if config.torch_profiler_tensorboard_enabled:
        tensor_board = tensorboard_trace_handler(
            dir_name=config.torch_profiler_tensorboard_log_path,
            worker_name=f'rank{distributed_ctx.ddp_rank}'
        )

    def trace_handler(prof):
        if tensor_board is not None:
            tensor_board(prof)
        else:
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
