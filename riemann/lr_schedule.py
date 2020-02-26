from sacred import Ingredient
from torch.optim.lr_scheduler import LambdaLR
from math import sqrt

lr_schedule_ingredient = Ingredient("lr_schedule")

@lr_schedule_ingredient.config
def config():
    """
    Possible schedule types:
        constant - Is fixed at base_lr
        linear - Linearly interpolates between scheduled_lrs over durations specified in
            lr_durations and remains constant at the last lr
        fixed_schedule - Iterates through scheduled_lrs over durations specified in
            lr_durations and remains constant at the last lr
    """
    schedule_type = "fixed_schedule"
    base_lr = 0.001
    fixed_embedding_lr = 50
    if schedule_type == "linear" or schedule_type == "fixed_schedule":
        base_lr = 1

    scheduled_lrs = [0.001, 0.0005]
    lr_durations = [1]
    decay_rate = 20

@lr_schedule_ingredient.capture
def get_lr_scheduler(optimizer, schedule_type, base_lr, scheduled_lrs, lr_durations, decay_rate):
    if schedule_type == "constant":
        return LambdaLR(optimizer, lambda epoch: 1)
    elif schedule_type == "linear" or schedule_type == "fixed_schedule":
        return LambdaLR(optimizer,
                lambda epoch: get_lr_from_schedule(epoch, scheduled_lrs, lr_durations,
                    linear_interpolate = (schedule_type == "linear")))
    elif schedule_type == "decay":
        return LambdaLR(optimizer, lambda epoch: 1/sqrt(1 + decay_rate * epoch))


@lr_schedule_ingredient.capture
def get_base_lr(base_lr):
    return base_lr

@lr_schedule_ingredient.capture
def get_fixed_embedding_lr(fixed_embedding_lr):
    return fixed_embedding_lr

def get_lr_from_schedule(epoch, scheduled_lrs, lr_durations, linear_interpolate=False):
    i = 0
    sum_epochs = 0
    for i in range(len(lr_durations)):
        sum_epochs += lr_durations[i]
        if epoch < sum_epochs:
            break
    if epoch >= sum_epochs:
        i += 1

    if not linear_interpolate or i == len(lr_durations):
        return scheduled_lrs[i]
    if linear_interpolate:
        progress = (epoch - sum_epochs + lr_durations[i])/lr_durations[i]
        return (1 - progress) * scheduled_lrs[i] + progress * scheduled_lrs[i+1]
