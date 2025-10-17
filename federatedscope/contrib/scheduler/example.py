from federatedscope.register import register_scheduler


def call_my_scheduler(optimizer, reg_type, **kwargs):
    # Accept but ignore additional kwargs to avoid conflicts with other schedulers
    try:
        import torch.optim as optim
    except ImportError:
        optim = None
        scheduler = None

    if reg_type == 'myscheduler':
        if optim is not None:
            lr_lambda = [lambda epoch: epoch // 30]
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return scheduler

    # Return None if this is not our scheduler type
    return None


register_scheduler('myscheduler', call_my_scheduler)

