from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_ondemfl_cfg(cfg):
    cfg.ondemfl = CN()
    cfg.ondemfl.enable = False
    cfg.ondemfl.pretrain_rounds = 0
    cfg.ondemfl.ondemand_rounds = 0
    cfg.ondemfl.target_distribution = []
    cfg.ondemfl.subset_size = -1
    cfg.ondemfl.weight_scheme = 'ratio_times_size'
    cfg.ondemfl.min_ratio = 1e-8
    cfg.ondemfl.nnls_max_iter = 500
    cfg.ondemfl.nnls_tol = 1e-9
    cfg.ondemfl.freeze_predictor_after_stage1 = True
    cfg.ondemfl.dp_loss = 'mse'
    cfg.ondemfl.dp_optimizer = CN()
    cfg.ondemfl.dp_optimizer.type = 'SGD'
    cfg.ondemfl.dp_optimizer.lr = 1e-3
    cfg.ondemfl.dp_optimizer.momentum = 0.0
    cfg.ondemfl.dp_optimizer.weight_decay = 0.0
    cfg.ondemfl.dp_optimizer.nesterov = False
    cfg.ondemfl.grad_layer = ''
    cfg.ondemfl.log_metrics = True


register_config('ondemfl', extend_ondemfl_cfg)
