# FedMM 在 FederatedScope 中的实现说明

本文档汇总了 `FederatedScope` 框架中 FedMM 的整体结构、关键配置以及复现步骤，便于在单域多客户端场景下快速扩展或复用该方法。

## 1. 组件概览

| 组件 | 位置 | 说明 |
| --- | --- | --- |
| 数据构建 | `federatedscope/fedmm/data/fedmm_mnistm.py` | 加载 MNIST/MNIST-M，抽取 `fedmm_data`，并支持自定义多客户端划分。 |
| 模型 | `federatedscope/fedmm/model/dann.py` | 实现 FedMM 论文中的 DANN 主干（特征提取 + 任务头 + 域判别头 + GRL）。 |
| 优化器 | `federatedscope/fedmm/optimizer/pdgd.py` | Primal-Dual Gradient Descent，维持 `v*` 与对偶变量。 |
| 训练器 | `federatedscope/fedmm/trainer/fedmm_trainer.py` | 定义 FedMM 特有的批处理逻辑、域分类损失、PD 同步等。 |
| 运行脚本 | `scripts/fedmm/*.yaml` | 提供 2 客户端的基础脚本以及 `multi_user` 下的对比实验。 |

## 2. 数据加载与多客户端划分

`fedmm_mnistm` 数据入口负责：

1. **下载 / 读取**  
   - 从 `cfg.data.root` 下加载 `MNIST`（torchvision 自动下载）。  
   - 从 `mnistm_cache.pt` 或 `mnistm_data.pkl` 读取 MNIST-M；需要提前转换好放在 `data/`。
2. **构造 `ClientData.fedmm_data`**  
   每个客户端都以 `{'source': (xs, ys), 'target': (xt, yt)}` 形式持有域内张量；训练时真正用到的批次来自这些张量。
3. **多客户端策略**
   - 传统 2 客户端划分由 `fedmm.source_ratio / target_ratio` 控制。
   - `fedmm.client_domain_specs`：直接声明每个客户端来自哪个域以及各自的样本权重，适合小规模自定义。
   - `fedmm.domain_groups`：以“域”为单位声明 `client_num` 和切分器（目前支持 `lda`/均匀/权重），并在内部调用 `dirichlet_distribution_noniid_slice` 为每个域生成指定数量的客户端数据。此方式已用于 “单域 10 客户端” 需求。

示例（20 客户端，单域 10 个，LDA 划分）：

```yaml
fedmm:
  domain_groups:
    - domain: target
      client_num: 10
      splitter: lda
      splitter_args:
        alpha: 0.5
    - domain: source
      client_num: 10
      splitter: lda
      splitter_args:
        alpha: 0.5
```

若提供 `domain_groups`，其 `client_num` 之和必须与 `federate.client_num` 一致；每个域目前仅支持一个分组。

## 3. 模型结构

`FedMMDANN` 由三部分组成：

1. **ToyFeatureExtractor**：两个 `Conv5x5+ReLU` 与 `MaxPool`，输入默认 3 通道。  
2. **Task Head**：`Linear -> ReLU -> Dropout -> Linear`，输出分类 logits。  
3. **Domain Head**：结构同 Task Head，输出 2 维域 logits，配合梯度反转层实现对抗训练。  

训练器会在每轮本地训练前调用 `model.set_grl_lambda(cfg.fedmm.domain_loss_coef)` 来调节域损失权重。

## 4. FedMM 训练流程

`FedMMTrainer` 的核心逻辑：

1. **批数据生成**：  
   - 依据源/目标样本数自动分配批大小（`cfg.fedmm.batch_size`），保证两域数据都被采样。  
   - 通过自定义 `CyclingBatchSampler` 无限循环并随机打乱。
2. **损失**：  
   - 分类损失：对源域样本计算 `CrossEntropyLoss`。  
   - 域判别损失：对源+目标样本统一训练域分类器。  
3. **优化**：  
   - PD 模式：`fedmm.enable_pd=True` 时使用 `PDOptimizer`。训练器会维护上一轮全局权重与对偶变量，并在返回模型前执行 `_adjust_with_dual`。  
   - 传统模式：退化为 `SGD`，用于基线对比。
4. **返回值**：  
   - `train()` 返回样本数、更新后的可共享参数以及 `{'loss': avg_loss}`，与 FederatedScope 通信流程对齐。

## 5. 关键配置项

| 配置 | 含义 |
| --- | --- |
| `fedmm.batch_size` | 每个客户端的混合批大小。 |
| `fedmm.lambda1` / `lambda1_decay` / `lambda1_decay_step` | 对偶变量更新与衰减控制。 |
| `fedmm.mu` | PD 优化器中的罚项系数。 |
| `fedmm.domain_loss_coef` | GRL 系数，也即域分类损失权重。 |
| `fedmm.source_ratio` / `target_ratio` | 仅在 2 客户端场景使用，用于按比例拆分 MNIST / MNIST-M。 |
| `fedmm.client_domain_specs` | 直接列出每个客户端的 `source/target` 权重或 `domain` 类型。 |
| `fedmm.domain_groups` | 新增的分组式配置，声明域、客户端数、切分器以及参数（如 `alpha`）。 |
| `trainer.type` | FedMM 需设置为 `fedmm_trainer`。 |
| `fedmm.enable_pd` | 是否启用 PDGD（设为 `False` 即 FedAvg-SGDA 基线）。 |

## 6. 实验脚本与运行方式

常用脚本：

- `scripts/fedmm/fedmm_on_mnistm.yaml`：两客户端、与论文一致的设置。  
  ```bash
  python federatedscope/main.py --cfg scripts/fedmm/fedmm_on_mnistm.yaml
  ```
- `scripts/fedmm/multi_user/fedmm_multi_user.yaml`：20 客户端（每域 10），使用 LDA 划分与 PDGD。  
- `scripts/fedmm/multi_user/fedavg_multi_user.yaml`：相同数据划分下的 FedAvg 基线（禁用 PD）。  
  ```bash
  python federatedscope/main.py --cfg scripts/fedmm/multi_user/fedmm_multi_user.yaml
  python federatedscope/main.py --cfg scripts/fedmm/multi_user/fedavg_multi_user.yaml
  ```

运行完成后，实验结果默认保存在 `exp/<exp_name>/<timestamp>/`，可通过 `eval_results.log` 和 `exp_print.log` 查看精度曲线及 loss。

## 7. 数据准备注意事项

1. **MNIST-M 数据**：`fedmm_mnistm` 期望在 `cfg.data.root` 中找到 `mnistm_cache.pt`（推荐）或 `mnistm_data.pkl`。若不存在会抛出异常，请先执行 `projects/FedMM/create_mnistm.py` 或自行转换。  
2. **缓存格式**：张量需为 `[N, 3, 28, 28]`，若通道维在最后，加载器会自动转置。  
3. **占位样本**：对仅含单域数据的客户端，加载器会自动插入零样本占位，从而满足框架的张量接口要求。

## 8. 自定义与扩展建议

1. **更换模型**：实现新模型并通过 `register_model` 注册即可在 YAML `model.type` 中引用。确保 `forward(x, return_domain=True)` 返回 `(logits, domain_logits)`。  
2. **新的数据集**：仿照 `fedmm_mnistm` 新建数据函数并通过 `register_data` 注册；只要 `ClientData.fedmm_data` 中提供源/目标张量即可复用现有训练器。  
3. **控制域划分难度**：调整 `splitter_args.alpha`（越小越非 IID）或改为自定义 `weights`，即可模拟不同的域不均衡程度。  
4. **融合其他优化策略**：可以在 `FedMMTrainer` 中添加新的优化分支（例如 Adam/AdamW），保持返回值与 PD 分支一致即可。

---  
如需进一步与论文或原版代码对照，可参考 `projects/FedMM` 和 `papers/FedMM/FedMM.md`，当前实现已在 FederatedScope 中独立完成，便于与其他算法在统一框架下对比。
