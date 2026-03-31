from tasks.pretraining import PretrainingTask
from tasks.instruct import InstructTask
from tasks.dpo import DPOTask
from config import TrainingStage

def get_task(training_stage):
    if training_stage == TrainingStage.PRETRAIN:
        return PretrainingTask()
    elif training_stage == TrainingStage.INSTRUCT:
        return InstructTask()
    elif training_stage == TrainingStage.DPO:
        return DPOTask()
    else:
        raise ValueError('Unsupported training task')
