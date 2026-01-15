# 周报（客户端自动重连与最小客户端阈值机制）

## 1. 目标&场景
- 目标：为分布式训练增加客户端自动重连能力和最小客户端阈值检测，提升训练的容错性和稳定性。
- 场景：`scripts/distributed_scripts/run_distributed_conv_femnist_managed.sh` 驱动的分布式训练流程。
- 前置工作：基于已实现的心跳机制，解决其遗留问题："当前客户端离线后不自动恢复"和"增加 min_clients_threshold 配置项"。

## 2. 设计调研

### 2.1 gRPC 指数退避算法（设备重连）
参考 gRPC 官方连接退避规范（[gRPC Connection Backoff Protocol](https://github.com/grpc/grpc/blob/master/doc/connection-backoff.md)），核心参数：
- `INITIAL_BACKOFF`: 1 秒
- `MULTIPLIER`: 2.0（每次重试翻倍）
- `MAX_BACKOFF`: 60 秒（上限）
- `JITTER`: ±20%（随机抖动，避免惊群效应）

### 2.2 PyTorch Elastic Training
参考 PyTorch `torchrun` 弹性训练机制：
- 支持 `--max-restarts` 限制重启次数
- Worker 失败后自动保存检查点并尝试恢复
- 提供 `TORCHELASTIC_MAX_RESTARTS` 环境变量控制

### 2.3 Circuit Breaker 模式
参考微服务熔断器设计（如 Netflix Hystrix）：
- **Closed**：正常工作，监控失败率
- **Open**：失败率超阈值后停止请求，进入冷却期
- **Half-Open**：冷却后尝试恢复

本实现借鉴上述设计：采用 gRPC 退避算法实现重连，结合阈值检测实现类似熔断器的暂停/恢复逻辑。

## 3. 主要开发内容

### 3.1 配置体系扩展（`core/configs/cfg_fl_setting.py`）
新增重连与阈值相关配置项：
```python
# 客户端重连配置（遵循 gRPC 退避算法）
cfg.distribute.reconnect_enabled = True
cfg.distribute.reconnect_max_attempts = 10
cfg.distribute.reconnect_initial_backoff = 1.0   # 秒
cfg.distribute.reconnect_max_backoff = 60.0      # 秒
cfg.distribute.reconnect_backoff_multiplier = 2.0
cfg.distribute.reconnect_jitter = 0.2            # 20% 抖动

# 服务器阈值配置
cfg.federate.min_clients_threshold = 0           # 0 表示禁用
cfg.federate.graceful_termination_timeout = 300  # 秒
```
在 `assert_fl_setting_cfg` 中追加参数校验逻辑。

### 3.2 通信层增强（`core/communication.py`）
- 新增连接丢失回调机制：
  ```python
  def register_connection_lost_callback(self, callback):
      self._connection_lost_callbacks.append(callback)

  def _notify_connection_lost(self, neighbor_id):
      for callback in self._connection_lost_callbacks:
          callback(neighbor_id)
  ```
- 新增在线客户端计数：`get_online_client_count()`
- 发送失败时触发回调通知

### 3.3 客户端重连机制（`core/workers/client.py`）
- **重连监控线程**：`_start_reconnect_monitor()` 启动后台线程监听连接丢失事件
- **连接丢失回调**：`_on_connection_lost()` 在检测到服务器连接断开时设置重连事件
- **指数退避重连**：`_attempt_reconnect()` 实现 gRPC 风格的退避算法：
  ```python
  for attempt in range(max_attempts):
      # 尝试发送 join_in 消息重连
      self.comm_manager.send(Message(msg_type='join_in', ...))
      # 计算下次退避时间（含 jitter）
      jitter_value = random.uniform(-jitter, jitter) * current_backoff
      sleep_time = min(current_backoff + jitter_value, max_backoff)
      current_backoff = min(current_backoff * multiplier, max_backoff)
  ```
- 在 `run()` 和 `callback_funcs_for_finish()` 中管理线程生命周期

### 3.4 服务器阈值检测（`core/workers/server.py`）
- **阈值检查**：`_check_clients_threshold()` 在每轮聚合前调用
- **暂停/恢复训练**：
  ```python
  def _pause_training(self, reason):
      self._is_training_paused = True
      self._pause_start_time = time.time()

  def _resume_training(self, online_count):
      self._is_training_paused = False
      self._pause_start_time = None
  ```
- **优雅终止**：超过 `graceful_termination_timeout` 后调用 `_save_checkpoint_and_terminate()` 保存模型并退出
- **重连客户端同步**：修改 `callback_funcs_for_join_in()` 处理已有 ID 客户端的重连请求，同步当前模型参数

### 3.5 三种终止场景
| 场景 | 触发条件 | 行为 |
|------|----------|------|
| 正常完成 | `state >= total_round_num` | 广播 finish 消息 |
| 早停 | `EarlyStopper` 触发 | 广播 converged 消息 |
| 优雅终止 | 在线客户端 < 阈值超过 timeout | 保存检查点并退出 |

## 4. 文件修改清单
| 文件 | 修改内容 |
|------|----------|
| `core/configs/cfg_fl_setting.py` | 新增 8 个配置项 + 校验逻辑 |
| `core/communication.py` | 连接丢失回调、在线计数 |
| `core/workers/client.py` | 重连监控线程、指数退避重连 |
| `core/workers/server.py` | 阈值检查、暂停/恢复、优雅终止、重连同步 |

## 5. 测试与验证

### 5.1 已验证功能
- 配置体系：8 个新增配置项可正确加载，校验逻辑生效
- 基础重连流程：单客户端断线后可自动重连并恢复训练
- 阈值检测：在线客户端低于阈值时可触发暂停

### 5.2 待验证场景
- 高并发重连（多客户端同时断线）
- 长时间运行下的超时精度
- 重连后模型状态一致性 

## 6. 问题排查

### 6.1 调试中的问题

#### 问题 1: 重连竞态条件导致偶发性死锁

**现象描述**：
在高并发场景下，多个客户端同时断线并尝试重连时，偶发性出现客户端进程无响应，必须手动终止重启。

**已排查方向**：
- 网络超时配置：调整各超时参数后问题依旧
- gRPC 连接池耗尽：监控显示连接数正常
- 消息队列阻塞：队列长度未达上限
- 初步怀疑 `client.py:290` 处 `_reconnect_lock` 与心跳线程存在锁竞争，但复现率约 5%，难以稳定触发

**当前状态**：
正在添加线程状态日志，等待下次复现时获取完整的锁获取顺序。

#### 问题 2: 重连后客户端训练数据异常

**现象描述**：
重连成功的客户端在后续训练轮次中，loss 出现异常飙升，其梯度更新方向与其他正常客户端存在明显偏差，影响全局模型收敛。

**已排查方向**：
- 网络传输丢包：对比模型参数 MD5 校验一致
- 数据加载器状态：重置 DataLoader 后问题依旧
- 随机种子问题：固定 seed 后仍可复现
- 初步怀疑与 optimizer 内部状态（如 momentum buffer）未正确重置有关，具体原因尚未确定

**当前状态**：
正在逐步打印 `trainer.optimizer.state` 各项数值，对比重连前后差异。

#### 问题 3: 优雅终止超时计算不准确

**现象描述**：
配置 `graceful_termination_timeout = 300` 秒，实际观测在约 180-200 秒时提前触发终止，误差达 30%-40%。

**已排查方向**：
- 配置读取错误：运行时打印确认配置值正确
- NTP 时间同步：服务器系统时间稳定
- 多次暂停累积：日志确认仅触发一次暂停事件
- 具体原因待定

**当前状态**：
正在添加细粒度时间戳日志，记录每次 `_check_clients_threshold` 调用时的 `elapsed` 值变化。

### 6.2 已解决的问题

#### 问题 4: 在线客户端计数与实际状态不一致

**现象描述**：
日志显示在线客户端数为 3，但实际仅有 2 个客户端正常工作，导致阈值判断失效，训练未按预期暂停。

**根本原因**：
- `communication.py:240-243` 的 `get_online_client_count` 基于 `_neighbor_status` 字典统计
- 客户端异常崩溃（如 `kill -9`）时，gRPC 连接不会立即断开
- 心跳超时检测存在延迟窗口，在此期间 `_neighbor_status` 仍为 `True`
- 若客户端在发送心跳后、收到 ACK 前崩溃，服务器仍认为其在线

**解决方案**：
结合 gRPC channel 状态检查与心跳超时机制进行双重判断，添加 `grpc_channel.get_state()` 作为快速失败路径。

#### 问题 5: 多客户端同时重连引发服务器 OOM

**现象描述**：
大量客户端（50+）同时断线并重连时，服务器内存迅速耗尽，被 OOM Killer 终止。

**根本原因**：
- 重连时服务器在 `callback_funcs_for_join_in` 中为每个客户端准备完整模型参数
- 50 个客户端同时发送 `join_in` 消息时，服务器需同时持有 50 份模型参数副本
- 大模型（如 ResNet-50）单份参数约 100MB，加上 protobuf 序列化开销，总内存占用超出限制

**解决方案**：
- 添加重连请求队列，限制并发处理数量
- 实现参数缓存复用，避免重复序列化
- 对重连请求实施 rate limiting

## 7. 使用说明

### 7.1 默认行为（无需修改配置）
- 客户端：自动重连已开启，最多 10 次，使用 gRPC 指数退避
- 服务器：阈值检测启用（`min_clients_threshold = 5`）

### 7.2 启用阈值检测（仅需修改服务器配置）
```yaml
# distributed_femnist_server.yaml
federate:
  min_clients_threshold: 2
  graceful_termination_timeout: 300
```

### 7.3 自定义重连参数（可选）
```yaml
# distributed_femnist_client_*.yaml
distribute:
  reconnect_enabled: True
  reconnect_max_attempts: 5
  reconnect_initial_backoff: 2.0
```

## 8. 实现总结
- **参考成熟实现**：gRPC 退避算法、PyTorch Elastic、Circuit Breaker 模式
- **非侵入式**：默认值保证向后兼容，无需修改现有配置文件
- **安全降级**：超时后保存检查点而非直接崩溃，支持后续手动恢复

## 9. 后续计划
- 解决第 6 节中仍在调试的问题
- 可考虑添加管理员 API 动态调整阈值
- 可扩展为支持客户端主动退出/加入的弹性伸缩机制
