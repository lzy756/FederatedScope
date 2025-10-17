#!/usr/bin/env python
"""
Script to run FedLSA on multi-domain datasets (Office-Caltech, Digits, PACS)

Example usage:
    python run_fedlsa.py --cfg scripts/example_configs/fedlsa_office_caltech.yaml
    python run_fedlsa.py --cfg scripts/example_configs/fedlsa_office_caltech.yaml \
                         fedlsa.lambda_com 0.7 fedlsa.tau 0.2
"""
import sys
import os

# Add the FederatedScope directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.auxiliaries.worker_builder import get_client_cls, \
    get_server_cls
from federatedscope.core.configs.config import global_cfg, CfgNode
from federatedscope.core.auxiliaries.runner_builder import get_runner

# Import FedLSA specific modules to ensure they are registered
# The registration happens automatically when these modules are imported
from federatedscope.core.trainers.trainer_FedLSA import FedLSATrainer
from federatedscope.core.workers.server_FedLSA import FedLSAServer
from federatedscope.core.workers.client_FedLSA import FedLSAClient

# Note: FedLSA workers are automatically registered in client_FedLSA.py
# No need to manually register them here

if __name__ == '__main__':

    # Parse configuration
    init_cfg = global_cfg.clone()
    args = parse_args()

    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)

    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    # Update logger
    update_logger(init_cfg, clear_before_add=True)

    # Set random seed
    setup_seed(init_cfg.seed)

    # Load client configs if provided
    if args.client_cfg_file:
        client_cfgs = CfgNode.load_cfg(open(args.client_cfg_file, 'r'))
        client_cfgs.merge_from_list(client_cfg_opt)
    else:
        client_cfgs = None

    # Get data
    data, modified_cfg = get_data(config=init_cfg.clone(),
                                  client_cfgs=client_cfgs)
    init_cfg.merge_from_other_cfg(modified_cfg)

    # Freeze configuration
    init_cfg.freeze()

    # Determine server and client classes based on fedlsa usage
    # Check if fedlsa config exists and is enabled
    use_fedlsa = hasattr(init_cfg, 'fedlsa') and init_cfg.fedlsa.use

    if use_fedlsa:
        server_class = FedLSAServer
        client_class = FedLSAClient
        print("[INFO] Using FedLSA Server and Client")
    else:
        server_class = get_server_cls(init_cfg)
        client_class = get_client_cls(init_cfg)
        print(f"[INFO] Using {server_class.__name__} and {client_class.__name__}")

    # Get runner
    runner = get_runner(
        data=data,
        server_class=server_class,
        client_class=client_class,
        config=init_cfg.clone(),
        client_configs=client_cfgs
    )

    # Run federated learning
    _ = runner.run()

    print("\n" + "="*80)
    print("FedLSA Training Completed!")
    print("="*80)



