import json


class DistributedLogger:
    def __init__(self):
        self.is_master_process = False
    
    def set_master(self, is_master_process):
        self.is_master_process = is_master_process

    def info(self, content, force=False, pbar=None, is_json=False):
        if is_json:
            content = json.dumps(content, indent=4)
        if self.is_master_process or force:
            if pbar is not None:
                pbar.write(content)
            else:
                print(content)

logger = DistributedLogger()
