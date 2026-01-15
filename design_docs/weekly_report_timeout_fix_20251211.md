# 周报（优雅终止超时计算修复）

## 1. 目标&场景
- 目标：解决上周遗留的"优雅终止超时计算不准确"问题
- 场景：分布式训练中，当在线客户端低于阈值时触发暂停，超过 `graceful_termination_timeout` 后应优雅终止
- 问题表现：配置 300 秒超时，实际在约 180-200 秒时提前触发终止，误差达 30%-40%

## 2. 问题回顾

### 2.1 现象描述
配置 `graceful_termination_timeout = 300` 秒，实际观测在约 180-200 秒时提前触发终止，误差达 30%-40%。

### 2.2 上周已排查方向
- 配置读取错误：运行时打印确认配置值正确
- NTP 时间同步：服务器系统时间稳定
- 多次暂停累积：日志确认仅触发一次暂停事件

## 3. 根因分析

### 3.1 定位过程
通过添加细粒度时间戳日志，记录每次 `_check_clients_threshold` 调用时的时间值，发现：
- `time.time()` 返回值在某些时刻出现非单调变化
- 与系统 NTP 同步日志对比，时间跳变与 NTP 调整时刻吻合

### 3.2 根本原因
`time.time()` 返回的是系统墙上时钟（wall clock），存在以下问题：
- 受 NTP 时间同步影响，系统时间可能被向前或向后调整
- 在长时间运行的训练任务中，NTP 同步可能多次触发
- 当系统时间被向前调整时，`elapsed = time.time() - _pause_start_time` 计算结果偏大，导致提前触发终止

### 3.3 相关代码位置
```python
# server.py:357-358 (修复前)
elapsed = time.time() - self._pause_start_time
if elapsed > self._graceful_termination_timeout:
```

## 4. 解决方案

### 4.1 方案选择
采用 `time.monotonic()` 替代 `time.time()`：
- `time.monotonic()` 返回单调递增的时钟值，不受系统时间调整影响
- Python 官方文档推荐用于测量时间间隔
- 无需额外依赖，标准库原生支持

### 4.2 代码修改

#### `core/workers/server.py`
```python
# 修改 _pause_training 方法
def _pause_training(self, reason):
    self._is_training_paused = True
    self._pause_start_time = time.monotonic()  # 改用 monotonic
    self._pause_reason = reason
    logger.warning(f'Server: Training PAUSED - {reason}')

# 修改 _check_clients_threshold 方法中的超时检查
elapsed = time.monotonic() - self._pause_start_time  # 改用 monotonic
if elapsed > self._graceful_termination_timeout:
    ...

# 修改 _resume_training 方法中的日志计算
pause_duration = time.monotonic() - self._pause_start_time  # 改用 monotonic
```

## 5. 文件修改清单
| 文件 | 修改内容 |
|------|----------|
| `core/workers/server.py` | `_pause_training`、`_check_clients_threshold`、`_resume_training` 中的时间函数改用 `time.monotonic()` |

## 6. 测试与验证

### 6.1 测试方法
- 配置 `graceful_termination_timeout = 300` 秒
- 启动分布式训练后手动终止部分客户端，触发暂停状态
- 记录实际终止触发时间

### 6.2 测试结果
| 测试轮次 | 配置超时 | 实际触发时间 | 误差 |
|----------|----------|--------------|------|
| 1 | 300s | 299.8s | -0.07% |
| 2 | 300s | 300.2s | +0.07% |
| 3 | 300s | 300.1s | +0.03% |

修复后超时精度误差 < 1%，问题已解决。

## 7. 实现总结
- **问题根因**：`time.time()` 受 NTP 同步影响，在长时间运行场景下计算时间间隔不准确
- **解决方案**：改用 `time.monotonic()` 获取单调递增时钟
- **修改范围**：仅涉及 `server.py` 中 3 处时间相关代码
- **向后兼容**：无配置变更，无 API 变更
