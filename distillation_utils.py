import torch
import torch.nn.functional as F


def distillation_loss(teacher_logits, student_logits, temperature=1.0):
    teacher_probabilities = F.softmax(teacher_logits.view(-1, teacher_logits.size(-1)) / temperature, dim=-1)
    student_log_probabilities = F.log_softmax(student_logits / temperature, dim=-1)

    kl_divergence = F.kl_div(student_log_probabilities, teacher_probabilities, reduction='batchmean') * (temperature ** 2)
    return kl_divergence
