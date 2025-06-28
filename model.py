# model.py

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from omegaconf import OmegaConf                      # ← 新增
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.runner_builder import get_runner

if __name__ == "__main__":
    # 1. 用 OmegaConf 加载 config.yaml，而不是 yaml.safe_load
    cfg = OmegaConf.load("scripts/example_configs/cora_dirichlet_FedAvg.yaml")

    # 2. 构造 data & runner
    data, _ = get_data(cfg, None)
    runner = get_runner(data=data,
                        model_config=cfg.model,
                        backend=cfg.get("backend", "torch"))

    # 3. 拿到模型并打印层名
    model = runner.model
    print("=== Modules ===")
    for name, _ in model.named_modules():
        print(name)
    print("\n=== Parameters ===")
    for name, _ in model.named_parameters():
        print(name)
