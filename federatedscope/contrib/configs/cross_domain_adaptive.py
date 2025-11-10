from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_cross_domain_adaptive_cfg(cfg):
    cfg.cross_domain_adaptive = CN()
    cfg.cross_domain_adaptive.anchor_reweight = True
    cfg.cross_domain_adaptive.anchor_weight_momentum = 0.5
    cfg.cross_domain_adaptive.anchor_weight_eps = 1e-3


register_config('cross_domain_adaptive', extend_cross_domain_adaptive_cfg)
