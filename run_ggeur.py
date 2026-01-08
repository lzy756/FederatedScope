#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run GGEUR_Clip Experiments

This script integrates GGEUR_Clip components into FederatedScope and runs experiments.
"""

import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from federatedscope.core.cmd_args import parse_args
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.worker_builder import get_client_cls, get_server_cls
from federatedscope.core.auxiliaries.runner_builder import get_runner
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.auxiliaries.logging import update_logger

# Import and register GGEUR_Clip components
print("=" * 60)
print("Registering GGEUR_Clip components...")
print("=" * 60)

try:
    # Import and auto-register GGEUR_Clip components
    import federatedscope.contrib.ggeur_register

    print("✓ GGEUR_Clip components registered successfully!")
except Exception as e:
    print(f"✗ Failed to register GGEUR_Clip components: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def main():
    # Parse command line arguments
    init_cfg = global_cfg.clone()
    args = parse_args()

    # Merge config file
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)

    # Merge command line options
    if args.opts:
        init_cfg.merge_from_list(args.opts)

    # Update logger
    update_logger(init_cfg, clear_before_add=True)

    # Build data BEFORE freezing config
    # (dataset loading may modify config, e.g., client_num)
    print("\n" + "=" * 60)
    print("Building data...")
    print("=" * 60)
    data, modified_cfg = get_data(init_cfg.clone())

    # Merge modifications back into init_cfg
    init_cfg.merge_from_other_cfg(modified_cfg)

    # NOW freeze configuration
    init_cfg.freeze()

    print("\n" + "=" * 60)
    print("Configuration loaded:")
    print("=" * 60)
    print(f"Method: {init_cfg.federate.method}")
    print(f"Dataset: {init_cfg.data.type}")

    # Only print GGEUR_Clip scenario if using GGEUR_Clip method
    if hasattr(init_cfg, 'ggeur') and init_cfg.ggeur.use:
        print(f"Scenario: {init_cfg.ggeur.scenario}")

    print(f"Total rounds: {init_cfg.federate.total_round_num}")
    print(f"Clients: {init_cfg.federate.client_num}")
    print(f"Output dir: {init_cfg.outdir}")
    print("=" * 60 + "\n")

    print(f"✓ Data loaded: {len(data)} clients\n")

    # Build and run
    print("Building runner...")
    runner = get_runner(data=data,
                        server_class=get_server_cls(init_cfg),
                        client_class=get_client_cls(init_cfg),
                        config=init_cfg.clone(),
                        client_configs=None)

    print("\n" + "=" * 60)
    print(f"Starting Federated Learning ({init_cfg.federate.method})...")
    print("=" * 60 + "\n")

    # Run federated learning
    runner.run()

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
