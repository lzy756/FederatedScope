# 周报（修复重连日志引入的超时回归问题）

## 1. 背景
- 承接上次的问题2修复（优化器状态重置）
- 本周发现：问题3（超时不准确）在特定场景下复现

## 2. 问题回顾

### 2.1 现象描述
在有客户端重连的场景下，优雅终止超时再次出现不准确问题。配置 300 秒超时，实际在 260-280 秒时触发终止。
但在无重连的纯客户端掉线场景下，超时计算正常。

### 2.2 与上次修复的差异
- 上次修复后，纯掉线场景超时精度 < 1%（已验证）
- 本次问题仅在"掉线+重连成功"场景下复现

## 3. 根因分析

### 3.1 定位过程
对比有/无重连两种场景的日志，发现差异点：
- 重连场景下，`_check_clients_threshold` 被额外调用了一次
- 该次调用来自上次新增的重连成功回调逻辑

### 3.2 根本原因
上次修复问题2时，在 `_attempt_reconnect()` 成功分支添加了客户端重连耗时统计日志：

```python
# client.py:335 (上次新增)
reconnect_duration = time.time() - self._reconnect_start_time
logger.info(f'Client #{self.ID}: Reconnection took {reconnect_duration:.2f}s')
```

同时在服务器端 `callback_funcs_for_join_in` 处理重连请求时，新增了重连时间戳记录：

```python
# server.py:1231 (上次新增)
self._client_reconnect_time[sender] = time.time()
```

这个时间戳后续被用于计算"从暂停到恢复的总时长"，与 `_pause_start_time`（使用 `time.monotonic()`）混合计算，导致时间基准不一致。

### 3.3 问题代码
```python
# server.py:389 (问题代码)
def _resume_training(self, online_count=None):
    pause_duration = time.monotonic() - self._pause_start_time  # monotonic

    # 上次新增：统计重连客户端的平均重连耗时
    if self._client_reconnect_time:
        avg_reconnect = sum(
            time.time() - t  # 错误：混用 time.time()
            for t in self._client_reconnect_time.values()
        ) / len(self._client_reconnect_time)
        logger.info(f'Average reconnect time: {avg_reconnect:.2f}s')
```

## 4. 解决方案

### 4.1 修复方案
将所有时间间隔计算统一使用 `time.monotonic()`：

```python
# client.py - 修复
self._reconnect_start_time = time.monotonic()  # 改用 monotonic
# ...
reconnect_duration = time.monotonic() - self._reconnect_start_time

# server.py - 修复
self._client_reconnect_time[sender] = time.monotonic()  # 改用 monotonic
```

### 4.2 代码规范补充
在代码注释中添加规范说明，避免后续开发再次引入类似问题：
```python
# NOTE: 所有时间间隔计算必须使用 time.monotonic()，禁止使用 time.time()
# 原因：time.time() 受 NTP 同步影响，会导致计算不准确
```

## 5. 文件修改清单
| 文件 | 修改内容 |
|------|----------|
| `core/workers/client.py` | `_attempt_reconnect` 中时间记录改用 `time.monotonic()` |
| `core/workers/server.py` | `callback_funcs_for_join_in`、`_resume_training` 中时间记录改用 `time.monotonic()` |

## 6. 遗留问题
- 问题 1（重连竞态条件死锁）仍在排查中，复现率约 5%
- **新发现问题 4**：当多个客户端在短时间内同时重连时，服务器端的 `join_in` 消息处理可能出现竞态条件，导致部分客户端的模型参数同步丢失

## 7. 下周计划
- 继续排查问题 1，考虑引入 `RLock` 或使用 `asyncio` 重构锁逻辑
- 调研问题 4 的根因，初步怀疑与 gRPC 消息队列的并发处理有关