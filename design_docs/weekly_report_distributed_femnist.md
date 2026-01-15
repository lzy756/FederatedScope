# 周报（分布式 FEMNIST 数据加载改造）

## 1. 项目背景
- 目标：将模拟式 data_idx 切片升级为真实分布式数据加载机制，使每个客户端仅能访问本地 FEMNIST 分片。
- 场景：`scripts/distributed_scripts/run_distributed_conv_femnist_managed.sh` 驱动的分布式训练流程。

## 2. 设计与规划
- **分布式数据路径**：在规划文档 `distributed_data_loading_plan.md` 中明确了离线切分、运行时加载、配置扩展与回退策略。
- **Manifest 架构**：定义 `manifest.json` 记录 `dataset`、`strategy`、`total_clients`、`clients[*].shard_path/num_samples`，并通过 `core/auxiliaries/manifest_utils.py` 解析与校验。
- **Loader 逻辑**：新增 `core/data/distributed_loader.py`，在 `data_builder.get_data` 中检测 `cfg.distributed_data.enabled`，直接构造 FEMNIST 专用 Dataset，绕过原 data_idx 转换。

## 3. 主要开发内容
1. **配置体系扩展**（`core/configs/cfg_fl_setting.py`）
   - 新增 `cfg.distribute.client_id` 以及 `cfg.distributed_data.enabled/shard_path/manifest/...` 字段，并在 `assert_fl_setting_cfg` 中追加校验；例如：
     ```python
     cfg.distributed_data = CN()
     cfg.distributed_data.enabled = False
     cfg.distributed_data.shard_path = ''
     # ...
     if cfg.distributed_data.enabled:
         if cfg.distribute.role == 'client' and cfg.distribute.client_id < 0:
             raise ValueError(...)
     ```
2. **离线切分工具**（`scripts/tools/prepare_femnist_shards.py`）
   - 支持 round_robin/random/manual 分配策略，生成 `client_k/train|val|test.pt`、`meta.json` 及带 `strategy/total_clients` 字段的 Manifest：
     ```python
     save_split(shard_dir, 'train', train_pair)
     metadata = {'client_id': cid, 'tasks': [...], 'label_hist': ...}
     with open(os.path.join(shard_dir, 'meta.json'), 'w') as f:
         json.dump(metadata, f, indent=2)
     ```
3. **运行时整合**
   - `core/data/distributed_loader.py`：实现 `FemnistShardDataset` 和 `DistributedShardLoader`，提供 transform fallback 和 manifest 元数据：
     ```python
     train_ds = self._wrap_pair(train_pair, transforms)
     if train_ds is None:
         raise ValueError('No train split found ...')
     metadata = self._build_metadata(train_ds, val_ds, test_ds)
     ```
   - `core/auxiliaries/data_builder.py`：加入 `_use_distributed_loader` 判断，分支调用 `load_distributed_data(config)`。
   - `scripts/distributed_scripts/distributed_configs/distributed_femnist_*.yaml`：每个客户端新增 `distributed_data` 段与 `client_id`，服务器补充 `model.input_shape: [1, 28, 28]`。
   - `run_distributed_conv_femnist_managed.sh`：启动前校验 Manifest/分片目录并提示 `prepare_femnist_shards.py` 命令。
4. **文档与示例**
   - `scripts/README.md` 新增示例：
     ```bash
     python scripts/tools/prepare_femnist_shards.py --clients 3
     bash distributed_scripts/run_distributed_conv_femnist_managed.sh
     ```
5. **测试保障**
   - 新增 `tests/test_distributed_femnist_loader.py`：使用临时目录构造分片，调用 `get_data` 并断言 `metadata`、DataLoader batch 行为。

## 4. Bug 修复 & 调优
- Server 端不再误写 `manifest_cache` 至配置，避免 YACS KeyError。
- 显式设置 `model.input_shape`，解决 server 无数据时推断失败。
- `FemnistShardDataset` 默认提供张量 fallback（无 `ToTensor` 配置时仍可训练）。

## 5. 测试与验证
- 单测：`python -m pytest tests/test_distributed_femnist_loader.py -vv` 全量通过。
- 手动验证：
  - `python scripts/tools/prepare_femnist_shards.py --clients 3`
  - `bash scripts/distributed_scripts/run_distributed_conv_femnist_managed.sh`，确认 Server/Client 按各自分片加载并稳定运行。

## 6. 遗留与后续计划
- Manifest 目前仅支持 FEMNIST，可抽象出通用字段供 Celeba/DomainNet 复用。
<!-- - 仍需评估大规模客户端场景下的 Manifest 解析与磁盘占用优化（软链接/索引化）。 -->
<!-- - 建议在 CI 中新增 `prepare_femnist_shards.py` 的快速冒烟测试，防止未来回归。 -->
