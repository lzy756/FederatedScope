# Cross-Domain Adaptive 方法说明

本文结合当前代码实现，对 `cross_domain_adaptive` 方法的设计目标、核心组件、运行流程及调参与排障进行详细说明，方便阅读者快速理解根目录下的跨域自适应方案。

---

## 1. 设计动机

跨域联邦任务通常同时面临两种异质性：

1. **Domain skew**：不同客户端来自不同域（Office-Caltech 的 amazon/webcam/dslr/caltech）。FedLSA 通过服务器端语义锚点与客户端超球面对比损失来学习域无关特征。
2. **Class distribution heterogeneity**：客户端所含类别极不均衡。On-Demand FL 通过梯度驱动的类别分布预测器推断本地类别比例，服务器根据预测结果按需抽取最贴近目标分布的客户端。

`cross_domain_adaptive` 即为 “FedLSA + On-Demand FL” 的融合：在 **语义锚点对齐** 的同时利用 **预测分布 + NNLS 子集调度** 解决类别不均衡。

---

## 2. 核心模块（依据实际代码）

| 模块 | 文件 | 关键职责 |
| --- | --- | --- |
| **模型构建** | `federatedscope/contrib/model/ondemfl.py` | 当 `model.type` 为 `cross_domain_adaptive` 时，创建 `OnDemFLModel`：包含 `task_model`（例如 `fedlsa_cnn`）与 `DistributionPredictor`。预测器输入为最后一层线性 layer 的梯度，输出为类别概率。 |
| **客户端 Trainer** | `federatedscope/contrib/trainer/cross_domain_adaptive_trainer.py` | 继承自 `GeneralTorchTrainer`，同时优化：① FedLSA 的任务模型/投影器；② 类别分布预测器。训练结束前将预测结果保存到 `ctx.ondem_pred_distribution`，但上传模型参数时过滤 `distribution_predictor.*`。 |
| **Worker（客户端 & 服务器）** | `federatedscope/contrib/worker/cross_domain_adaptive.py` | - 客户端类多重继承自 `FedLSAClient` 与 `OnDemFLClient`，从服务器接收语义锚点并缓存预测分布。<br>- 服务器类继承 `FedLSAServer`，额外实现 On-Demand 阶段：NNLS 求解 `_ratio_map`、选择 `_sampling_clients`、按需广播模型。 |
| **配置** | `scripts/example_configs/cross_domain_adaptive_office_caltech.yaml` | 设定 `federate.method/trainer.type/model.type = cross_domain_adaptive`，并提供 Office-Caltech 运行示例。关键字段包括 `ondemfl.*`（阶段长度、NNLS 参数、目标分布）、`fedlsa.*`（锚点超参数）等。 |

---

## 3. 运行流程详解

### 3.1 初始化

1. **数据划分**  
   - `federatedscope/cv/dataloader/dataloader.py` 会按照 `data.dirichlet_alpha` 将每个域划分给多个客户端。日志中 “Splitting domain 'amazon' among 5 clients using Dirichlet…” 即来自此处。

2. **模型与锚点**  
   - `get_model()`（`core/auxiliaries/model_builder.py`）调用 `call_ondemfl_model()`，生成 `task_model + distribution_predictor`。  
   - 服务器侧（`server_FedLSA.py`）初始化 `SemanticAnchorLearner` 与 `semantic_classifier`，用于在每轮前学习锚点。

### 3.2 客户端本地训练

在 `CrossDomainAdaptiveTrainer` 中，关键逻辑如下：

1. **FedLSA Forward**（`_hook_on_batch_forward_cross`）  
   - 取模型 embedding，经过 projector 投影至单位超球面；  
   - 使用语义锚点计算紧凑性损失 `loss_com`；  
   - 任务损失 `loss_ce + λ * loss_com` 传给 `ctx.loss_task`。

2. **分布预测器训练**（`_hook_on_batch_backward_cross`）  
   - 先对任务损失反向传播；  
   - 将选定线性层的梯度拼接成 `grad_vec`，输入 `distribution_predictor`；  
   - 与 batch 内真实类别频率对齐，依据 `ondemfl.dp_loss`（KL/MSE/MAE）更新预测器；  
   - 均值后的预测结果缓存在 `ctx.ondem_pred_distribution`，并经 `CrossDomainAdaptiveClient` 存为 `estimated_distribution`。

3. **上传参数**  
   - `get_model_para()` 会剥离 `distribution_predictor.*`，避免预测器参与聚合；若配置 `fedlsa.share_projector=True`，projector 参数也会被合并并上传。

### 3.3 服务器端阶段切换

服务器类的关键流程：

1. **预训练期（0 ~ `pretrain_rounds-1`）**  
   - `self._ondem_active=False`，继承自 FedLSA 的聚合流程，因此每轮都会：  
     1. 调用 `train_semantic_anchors()`（父类实现，文件 `server_FedLSA.py`），训练锚点若干 epoch；  
     2. 广播模型/锚点到全部客户端；  
     3. 聚合所有上传的 task model。
   - 此阶段虽然训练了预测器，但服务器只缓存其输出，不影响采样。

2. **触发 On-Demand 阶段**（`_perform_federated_aggregation`）  
   - 当 `self.state == ondemfl.pretrain_rounds` 时置 `_need_setup_after_stage1=True`，下一轮 `_start_new_training_round` 会调用 `_prepare_on_demand_phase()`：
     1. 读取每个客户端的 `estimated_distribution`，组成 `pred_matrix`；
     2. 若未配置 `ondemfl.target_distribution`，则根据客户端预测自动推断均匀目标；
     3. 通过 `nnls_solve(D, target)` 得到满足目标分布的权重；  
     4. 根据 `subset_size/min_ratio` 选出 `_sampling_clients` 与 `_ratio_map`；若数量少于原 `sample_client_num`，自动降级；  
     5. 将所有客户端切换状态 `set_phase('ondemand')`，并把 `_ondem_active` 置为 True。

3. **按需阶段（`_ondem_active=True`）**
   - **广播**：`broadcast_model_para()` 只向 `_sampling_clients` 发送模型/锚点，并根据 `_ratio_map` 动态采样、重权；  
   - **聚合**：`_perform_federated_aggregation()` 用 `_ratio_map` (或 `ratio_times_size/ratio/size`) 重写消息缓冲，实现“按需权重” FedAvg；  
   - **日志**：在 `server_FedLSA.py:331` 可看到每轮锚点训练的 `L_ACE / L_SEP / L_LSA` 进展，`eval_results.log` 中也会记录服务器整体测试/验证结果。

---

## 4. 配置说明（按 `scripts/example_configs/cross_domain_adaptive_office_caltech.yaml`）

| 字段 | 作用 | 常用值/备注 |
| --- | --- | --- |
| `federate.method` | 指定服务器/客户端类型 | 必须为 `cross_domain_adaptive` |
| `trainer.type` | 客户端 trainer | 设为 `cross_domain_adaptive` |
| `model.type` | 模型构建器 | `cross_domain_adaptive`（内部会创建 fedlsa_cnn + predictor） |
| `ondemfl.pretrain_rounds` / `ondemand_rounds` | 两阶段长度 | 示例中为 `60/60`，总和建议等于 `federate.total_round_num` |
| `ondemfl.subset_size` | 按需阶段最多采样客户端数 | 例：10；若为 -1 表示不限，退化为所有客户端 |
| `ondemfl.dp_loss` / `dp_optimizer` | 预测器损失与优化器 | 默认 KL + SGD(lr=1e-3, momentum=0.9) |
| `ondemfl.grad_layer` | 指定梯度提取的线性层 | 默认为空，自动使用 backbone 最后一层 `nn.Linear` |
| `fedlsa.projector_*` | projector 结构 | 例：input_dim=512, output_dim=128 |
| `fedlsa.anchor_train_epochs` | 每轮锚点训练 epoch 数 | 示例 5，日志中 `epoch 0/4 ...` 由此得来 |

若运行其他数据集，需保证：
- 数据配置中 `data.type` 对应已实现的数据载入器；
- 模型 `backbone` 支持 `fedlsa` 训练（具有可提取 embedding 的结构）。

---

## 5. 运行与日志

### 5.1 启动命令

```bash
python federatedscope/main.py \
  --cfg scripts/example_configs/cross_domain_adaptive_office_caltech.yaml
```

### 5.2 输出目录

- `exp_cda/office_caltech/<expname>/eval_results.log`：服务器/客户端训练、锚点日志；
- `.../eval_results.raw`：逐客户端/服务器的原始指标；
- `exp_cda/.../config.yaml`：最终冻结后的配置；
- `exp_cda/.../model_*.pth`：若启用保存模型，可在此查看。

---

## 6. 常见问题排查

1. **锚点训练出现 NaN**  
   - 查看 `eval_results.log` 中 `server_FedLSA` 的输出，确认是否在 On-Demand 阶段收集到 `estimated_distribution=None`。  
   - 预测器梯度可能全零：检查 `ondemfl.grad_layer` 是否指向真实存在的 `nn.Linear`，或在模型中添加 `assert layer is not None` 检查。

2. **按需阶段未触发**  
   - 总轮数必须 ≥ `pretrain_rounds + ondemand_rounds`；  
   - 当某些客户端迟迟未上传预测器结果时 `_prepare_on_demand_phase` 会直接返回，可通过客户端日志确认 `ctx.ondem_pred_distribution` 是否成功写入。

3. **性能波动大**  
   - 调整 `subset_size`、`ondemfl.min_ratio` 以确保 NNLS 后至少有与 `sample_client_num` 相同的客户端；  
   - 适当降低 `dp_loss` 的学习率可避免预测器过拟合。

---

## 7. 扩展建议

1. **支持更多 backbone**  
   - 在 `federatedscope/cv/model/model_builder.py` 注册新模型；保证存在可提取 embedding 的接口，或在 trainer 中添加自定义的 `_forward_with_embedding`。

2. **自定义目标分布**  
   - 在配置中设置 `ondemfl.target_distribution`（长度 = 类别数、和为 1），即可让按需阶段朝指定的类别比例调度客户端。

3. **debug 工具**  
   - 可在 `CrossDomainAdaptiveServer._prepare_on_demand_phase` 中临时添加日志（例如打印 `pred_matrix`、`ratios`），以分析 NNLS 结果。

---

如需更深入的理论背景，可参考 FedLSA 的论文说明（`FedLSA实现完整技术文档`）与 On-Demand FL 的 README；本仓库的实现旨在做一个工程化的组合，方便在 Office-Caltech 等跨域数据集上直接比较。欢迎根据业务需求继续扩展。***
