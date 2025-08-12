import wandb

from datetime import datetime
from config import config


class WandbWrapper():
    def __init__(self, enabled=True, is_master_process=True):
        self.WANDB = False
        self.is_master_process = is_master_process

        if enabled and self.is_master_process:
            WANDB_API_KEY = config.wandb_api_key
            if WANDB_API_KEY is not None:
                wandb.login(key=WANDB_API_KEY)
                self.WANDB = True
                print('Wandb enabled.')

    def init(self, project_name, *, job_name=None, config=None):
        if not self.WANDB:
            return
        
        if not job_name:
            job_start_time = datetime.now().strftime('%Y-%m-%d %H:%M')
            job_name = f'run_{job_start_time}'

        if config is None:
            wandb.init(project=project_name, name=job_name)
        else:
            wandb.init(
                project=project_name,
                name=job_name,
                config=config
            )

    def log(self, data):
        if not self.WANDB:
            return
        wandb.log(data)

    def finish(self):
        if not self.WANDB:
            return
        wandb.finish()
