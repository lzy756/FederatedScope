# 真分布式 FEMNIST 数据加载改造规划

## 1. 现状回顾（基于 `scripts/distributed_scripts/run_distributed_conv_femnist_managed.sh`）
- 启动脚本顺序化拉起 1 个服务器进程与 3 个客户端进程，所有进程运行 `python federatedscope/main.py --cfg …`。
- 服务器与客户端均指向同一套 YAML 配置（`scripts/distributed_scripts/distributed_configs/distributed_femnist_*.yaml`），通过 gRPC 端口 50051~50054 建立连接。
- 每个客户端都从 `data/` 目录加载完整 FEMNIST 数据，然后依赖 `distribute.data_idx` 在运行时从 `StandaloneDataDict` 中抽取指定切片。
- 数据虽然逻辑隔离，但仍是集中式读取：
  - **同一磁盘读取压力**，无法模拟异地数据存储。
  - **data_idx 静态切片**，不能反映真实客户端本地文件的可用性/缺失。
  - **无法支持分布式数据刷新**，因为转换发生在 `convert_data_mode`，所有原始样本都必须先进入内存。

## 2. 改造目标
- 客户端仅能看到本地磁盘上的 FEMNIST 子集，实现物理数据隔离。
- 服务器不再加载任何数据；客户端在加入联邦时自我声明数据规模与分片标识。
- 兼容现有模拟流程（可通过配置开关继续使用 data_idx），同时提供新路径实现“真分布式”模式。
- 支持离线数据切分、动态校验、容错处理（缺 shard 时给出清晰报错）。

## 3. 总体架构
```
数据准备阶段 (offline)
├─ scripts/tools/prepare_femnist_shards.py   # 读取原始 FEMNIST，生成 per-client 目录
├─ data/femnist/shards/client_1/{train.pt,test.pt,meta.json}
└─ manifest.json                             # 记录 shard -> 客户端 -> 样本数

运行阶段 (per process)
├─ Server (federatedscope/main.py, server cfg)
│  └─ 不加载数据，仅依据 manifest 校验客户端声明
└─ Client_k
   ├─ DistributedShardLoader                 # 根据 cfg.distributed_data.shard_path 直接读本地文件
   ├─ Trainer                                # 直接持有 torch Dataset/DataLoader
   └─ 心跳/上报                              # 报送 metadata 中的样本规模
```

## 4. 详细改造项

### 4.1 数据准备脚本
- **新增** `scripts/tools/prepare_femnist_shards.py`
  - 输入：原始 `data/femnist/`、目标客户端数、分配策略（顺序/随机/自定义 JSON）。
  - 输出：`data/femnist/shards/client_{k}/` 目录及 `manifest.json`，内含训练/验证/测试样本数、用户 ID 列表。
  - 额外生成 `meta.json`（含 `client_id`, `num_samples`, `label_hist`）。
- **可选**：支持断点续生成与 `--overwrite` 标志。

### 4.2 运行时加载链路
1. **新模块** `federatedscope/core/data/distributed_loader.py`
   - 类 `DistributedShardLoader`：
     - 根据 `cfg.distributed_data.shard_path` 直接创建 `torch.utils.data.TensorDataset`/`WrapDataset`。
     - 提供 `num_samples`, `label_hist`, `last_modified` 属性供客户端上报。
     - 支持流式加载：当 `cfg.distributed_data.streaming=true` 时采用 `IterableDataset` 读取磁盘切片。
2. **`load_cv_dataset` / `convert_data_mode` 调整**：
   - 当 `cfg.federate.mode == 'distributed'` 且 `cfg.distributed_data.enabled`：
     - 跳过集中式 translator，直接返回 `DistributedShardLoader` 产出的 `{train,val,test}`。
     - `convert_data_mode` 在检测到 `DistributedShardLoader` 时不再尝试 `data_idx` 切片。
3. **客户端初始化 (`federatedscope/core/workers/client.py`)**：
   - 如果数据对象是 `DistributedShardLoader`，直接传递底层 Dataset 给 Trainer。
   - 上报 `num_samples`/`sample_type` 给服务器，替换原先依据 `StandaloneDataDict` 推断的逻辑。

### 4.3 配置体系与启动脚本
- YAML 扩展：
  ```yaml
  distributed_data:
    enabled: true
    shard_path: data/femnist/shards/client_1
    manifest: data/femnist/shards/manifest.json
    streaming: false
    strict_check: true   # true 时缺文件即报错
  distribute:
    client_id: 1         # 替代 data_idx，唯一标识
  ```
- `scripts/distributed_scripts/distributed_configs/distributed_femnist_client_*.yaml`
  - 移除 `data_idx`，改为 `distributed_data.shard_path`（可由环境变量注入）。
  - 将 `data.root` 设为 shard 根路径或保留用于 transform（需在 loader 中处理）。
- `run_distributed_conv_femnist_managed.sh`
  - 读取 manifest 以生成 `CLIENT_K_CFG="… --client_id k --distributed_data.shard_path …"`（也可直接修改 YAML）。
  - 在启动前校验每个 `shard_path` 是否存在。

### 4.4 元数据与校验
- **Manifest 解析器**（服务器侧 `federatedscope/core/auxiliaries/manifest_utils.py`）：
  - 提供 `load_manifest(path) -> Dict[client_id, Metadata]`。
  - 服务器在每轮聚合前校验在线客户端是否与 manifest 匹配，若不匹配给出 warning/拒绝。
- **客户端自检**：`DistributedShardLoader` 初始化时验证 `meta.json` 内记录的样本数与实际张量维度一致。
- **日志优化**：`logger.info` 输出“Client k loaded 12456 train samples from data/femnist/shards/client_k/train.pt”。

### 4.5 回退与兼容
- 若 `distributed_data.enabled=false`：沿用旧 data_idx 逻辑，无需额外参数。
- 在 CLI 中提供 `--override distributed_data.enabled=false` 便于对比实验。

## 5. 文件改动清单
| 类型 | 文件 | 说明 |
| --- | --- | --- |
| 新增 | `scripts/tools/prepare_femnist_shards.py` | 离线切分 FEMNIST 并生成 metadata |
| 新增 | `federatedscope/core/data/distributed_loader.py` | 真分布式数据加载器实现 |
| 新增 | `federatedscope/core/auxiliaries/manifest_utils.py` | manifest 解析与校验工具 |
| 修改 | `federatedscope/core/auxiliaries/data_builder.py` | 接入分布式 loader 分支 |
| 修改 | `federatedscope/core/data/utils.py` | `convert_data_mode` 添加新分支，保留兼容逻辑 |
| 修改 | `federatedscope/core/workers/client.py` | 客户端初始化 & 元数据上报 |
| 修改 | `scripts/distributed_scripts/run_distributed_conv_femnist_managed.sh` | 启动时注入 shard 路径、添加健康检查 |
| 修改 | `scripts/distributed_scripts/distributed_configs/distributed_femnist_*.yaml` | 配置新增 `distributed_data` 段落 |
| 修改 | `doc/README.md` / `docs/source` | 文档说明如何准备分布式 FEMNIST |
| 修改 | `tests/test_femnist*.py` | 增补针对分布式 loader 的单测/集成测试 |

## 6. 实施里程碑
1. **阶段 A：离线数据切分（~0.5d）**
   - 实现 shard 准备脚本与 manifest；在 `data/` 目录验证输出结构。
2. **阶段 B：Loader 与配置接入（~1.5d）**
   - 编写 `DistributedShardLoader`，修改 `data_builder`、`convert_data_mode`、客户端初始化。
3. **阶段 C：脚本与 YAML 改造（~0.5d）**
   - 更新启动脚本、配置文件、README。
4. **阶段 D：验证与测试（~0.5d）**
   - 单测覆盖 loader；运行 `run_distributed_conv_femnist_managed.sh` 并截取日志确认每个客户端仅读自身数据。

## 7. 风险与应对
- **磁盘空间**：每个客户端独立存储数据，空间要求与客户端数线性相关；可在切分脚本中使用软链接或记录原始 user file 索引以降低复制量。
- **回退需求**：若需切回 data_idx，需确保配置中显式关闭 `distributed_data.enabled`；文档需突出说明。
- **跨平台差异**：Windows/WSL 环境路径差异，脚本内需统一使用 `os.path.join` 并支持相对路径。
- **大规模客户端**：manifest 与 loader 需懒加载，避免一次性读取成千上万 shard 的元数据；可使用生成器按需加载。

## 8. 下一步
- 与使用方确认 FEMNIST 原始数据来源与许可，确保可在本地重新切分。
- 决定 manifest 中是否记录用户标签直方图，以便服务器实现抽样/配额控制。
- 规划未来是否将该机制抽象到其它数据集（Celeba、DomainNet），以复用 `DistributedShardLoader`。
