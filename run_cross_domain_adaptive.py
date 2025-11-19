"""
Cross-Domain Adaptive Training Script

This script runs the cross-domain adaptive training with server-side evaluation.
"""
import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import FederatedScope main
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.auxiliaries.worker_builder import get_client_cls, get_server_cls
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.auxiliaries.runner_builder import get_runner


def main():
    # Parse arguments
    init_cfg = global_cfg.clone()
    args = parse_args()

    # Load config from file
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)

    # Merge from command line arguments
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    # Update logger
    update_logger(init_cfg, clear_before_add=True)

    # Set random seed
    setup_seed(init_cfg.seed)

    # Load data
    data, modified_cfg = get_data(config=init_cfg.clone())
    init_cfg.merge_from_other_cfg(modified_cfg)

    # Freeze config
    init_cfg.freeze()

    # Build runner
    runner = get_runner(data=data,
                       server_class=get_server_cls(init_cfg),
                       client_class=get_client_cls(init_cfg),
                       config=init_cfg.clone(),
                       client_configs=client_cfg_opt)

    # Run
    _ = runner.run()


if __name__ == "__main__":
    main()
