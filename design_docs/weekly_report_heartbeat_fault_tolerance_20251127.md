# 周报（分布式心跳机制与通信容错）

## 1. 目标
- 目标：为分布式训练添加心跳机制检测客户端存活状态，增强 gRPC 通信层的错误处理能力，提升长时间训练的稳定性和可恢复性。
- 场景：`scripts/distributed_scripts/run_distributed_conv_femnist_managed.sh` 驱动的分布式训练流程。

## 2. 设计与规划
- **心跳机制**：客户端每 30 秒向服务器发送心跳消息，服务器维护 `client_last_heartbeat` 字典跟踪客户端存活状态，超时 90 秒未收到心跳则标记客户端离线。
- **通信层增强**：gRPC 发送失败时采用指数退避重试策略（最多 3 次），全部失败后标记邻居离线并跳过后续发送。
- **线程安全**：使用 `threading.Lock` 保护共享状态（邻居状态字典、心跳时间戳字典）。
- **离线处理策略**：客户端离线后直接标记并跳过，不实现自动重连恢复。

## 3. 主要开发内容
1. **配置体系扩展**（`core/configs/cfg_fl_setting.py`）
   - 新增心跳与重试相关配置项：
     ```python
     cfg.distribute.heartbeat_interval = 30  # 心跳间隔秒数
     cfg.distribute.heartbeat_timeout = 90   # 超时阈值秒数
     cfg.distribute.send_max_retries = 3     # 发送最大重试次数
     cfg.distribute.send_retry_base_delay = 1.0   # 重试基础延迟
     cfg.distribute.send_retry_max_delay = 30.0   # 重试最大延迟
     ```

2. **通信层增强**（`core/communication.py`）
   - `gRPCCommManager` 新增邻居状态管理：
     ```python
     self._neighbor_status = dict()  # neighbor_id -> bool (True=online)
     self._status_lock = threading.Lock()
     ```
   - 新增状态管理方法：`mark_neighbor_offline()`, `mark_neighbor_online()`, `is_neighbor_alive()`, `get_alive_neighbors()`
   - 重写 `_send()` 方法实现指数退避重试：
     ```python
     for attempt in range(self._max_retries):
         try:
             stub.sendMessage(request)
             return True
         except grpc._channel._InactiveRpcError:
             delay = self._exponential_backoff(attempt)
             time.sleep(delay)
     self.mark_neighbor_offline(receiver_id)
     return False
     ```
   - 修改 `send()` 方法跳过已离线邻居

3. **gRPC 服务优化**（`core/gRPC_server.py`）
   - `receive()` 方法从忙等循环改为带 `poll_interval=0.01` 的 sleep 循环，降低 CPU 占用：
     ```python
     def receive(self, timeout=None, poll_interval=0.01):
         while len(self.msg_queue) == 0:
             if timeout and elapsed >= timeout:
                 return None
             time.sleep(poll_interval)
         return self.msg_queue.popleft()
     ```

4. **消息处理器注册**（`core/workers/base_server.py`, `core/workers/base_client.py`）
   - 服务器注册 `heartbeat` → `callback_funcs_for_heartbeat()`
   - 客户端注册 `heartbeat_ack` → `callback_funcs_for_heartbeat_ack()`

5. **服务器心跳监控**（`core/workers/server.py`）
   - 新增心跳跟踪状态：
     ```python
     self._heartbeat_lock = threading.Lock()
     self._client_last_heartbeat = dict()
     self._heartbeat_checker_thread = None
     self._heartbeat_stop_event = threading.Event()
     ```
   - 实现 `_start_heartbeat_checker()` 启动守护线程，每 30 秒检查超时客户端
   - 实现 `callback_funcs_for_heartbeat()` 更新心跳时间戳并回复 ack
   - 在 `run()` 开始时启动心跳检查，结束时调用 `_stop_heartbeat_checker()` 停止

6. **客户端心跳发送**（`core/workers/client.py`）
   - 新增心跳发送状态：
     ```python
     self._heartbeat_stop_event = threading.Event()
     self._heartbeat_thread = None
     ```
   - 实现 `_start_heartbeat_sender()` 启动守护线程，每 30 秒发送心跳
   - 实现 `callback_funcs_for_heartbeat_ack()` 处理服务器确认
   - 在 `run()` 开始时启动心跳发送，`callback_funcs_for_finish()` 中停止

## 4. 技术要点
- 心跳相关日志使用 `DEBUG` 级别避免刷屏，仅在客户端离线时使用 `WARNING` 级别
- 指数退避重试加入随机 jitter（0-1 秒）避免惊群效应
- 守护线程（daemon=True）确保主进程退出时自动清理
- 使用 `threading.Event.wait(timeout)` 实现可中断的定时等待

## 5. 测试与验证
- 语法检查：所有修改文件通过 Pylance 静态分析，无错误
- 导入测试：
  ```bash
  python3 -c "from federatedscope.core.communication import gRPCCommManager; print('OK')"
  python3 -c "from federatedscope.core.workers.server import Server; print('OK')"
  python3 -c "from federatedscope.core.workers.client import Client; print('OK')"
  ```
- 配置验证：
  ```bash
  python3 -c "
  from federatedscope.core.configs.cfg_fl_setting import extend_fl_setting_cfg
  from federatedscope.core.configs.config import CN
  cfg = CN(); extend_fl_setting_cfg(cfg)
  print(f'heartbeat_interval: {cfg.distribute.heartbeat_interval}')  # 30
  print(f'heartbeat_timeout: {cfg.distribute.heartbeat_timeout}')    # 90
  print(f'send_max_retries: {cfg.distribute.send_max_retries}')      # 3
  "
  ```

## 6. 修改文件清单
| 文件 | 修改内容 |
|------|----------|
| `core/configs/cfg_fl_setting.py` | 新增 5 个心跳/重试配置项 |
| `core/communication.py` | 增强 gRPCCommManager：邻居状态管理、指数退避重试 |
| `core/gRPC_server.py` | 优化 receive() 从忙等改为 sleep 循环 |
| `core/workers/base_server.py` | 注册 heartbeat 消息处理器 |
| `core/workers/base_client.py` | 注册 heartbeat_ack 消息处理器 |
| `core/workers/server.py` | 心跳检查线程、callback_funcs_for_heartbeat |
| `core/workers/client.py` | 心跳发送线程、callback_funcs_for_heartbeat_ack |

## 7. 遗留与后续计划
- 当前客户端离线后不自动恢复，后续可考虑添加重连机制
- 增加 `min_clients_threshold` 配置项，当在线客户端数低于阈值时暂停训练
