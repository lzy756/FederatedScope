# 周报（修复重连后客户端训练异常问题）

## 1. 背景
- 承接之前的周报，解决其中遗留的调试问题
- 本周主要解决：**问题 2 - 重连后客户端训练数据异常**

## 2. 问题回顾

### 2.1 现象描述
重连成功的客户端在后续训练轮次中，loss 出现异常飙升，其梯度更新方向与其他正常客户端存在明显偏差，影响全局模型收敛。

### 2.2 已排查方向
- 网络传输丢包：对比模型参数 MD5 校验一致
- 数据加载器状态：重置 DataLoader 后问题依旧
- 随机种子问题：固定 seed 后仍可复现
- 初步怀疑与 optimizer 内部状态（如 momentum buffer）未正确重置有关

## 3. 根因分析

### 3.1 问题定位
通过对比重连前后 `trainer.ctx.optimizer.state` 的内容，确认了问题根因：

```python
# 打印 optimizer state 示例
for param_group in optimizer.param_groups:
    for p in param_group['params']:
        state = optimizer.state[p]
        if 'momentum_buffer' in state:
            print(f'momentum_buffer norm: {state["momentum_buffer"].norm()}')
```

**发现**：
- 重连后客户端从服务器同步了新的模型参数（`model.state_dict()`）
- 但优化器状态（`optimizer.state_dict()`）仍保留断线前的历史状态
- 以 SGD with momentum 为例，`momentum_buffer` 记录的是旧模型参数的梯度动量
- 新模型参数与旧动量方向不一致，导致更新方向出现严重偏差

### 3.2 根本原因
查看 `torch_trainer.py:199-200` 的优化器初始化逻辑：
```python
ctx.optimizer = get_optimizer(ctx.model, **ctx.cfg[ctx.cur_mode].optimizer)
ctx.scheduler = get_scheduler(ctx.optimizer, **ctx.cfg[ctx.cur_mode].scheduler)
```

优化器仅在 `_hook_on_fit_start_init` 中初始化一次。当客户端重连并接收新模型参数时，`trainer.update()` 只更新了 `model.state_dict()`，而 `optimizer.state` 未被重置。

### 3.3 影响范围
受影响的优化器（带有内部状态）：
| 优化器 | 受影响状态 |
|--------|-----------|
| SGD（momentum > 0） | `momentum_buffer` |
| Adam / AdamW | `exp_avg`, `exp_avg_sq`, `step` |
| RMSprop | `square_avg`, `momentum_buffer` |
| Adagrad | `sum` |

## 4. 解决方案

### 4.1 方案设计
在客户端重连成功后，重置优化器状态。具体实现：
1. 在 `client.py` 的 `_attempt_reconnect()` 成功分支中调用优化器重置
2. 通过清空 `optimizer.state` 字典实现状态重置
3. 保留优化器超参数（lr, momentum 等）不变

### 4.2 代码修改

#### `core/workers/client.py`
在重连成功后添加优化器状态重置：
```python
def _attempt_reconnect(self):
    # ... existing reconnection logic ...
    
    if self.comm_manager.is_neighbor_alive(self.server_id):
        logger.info(f'Client #{self.ID}: Reconnection successful!')
        
        # 重置优化器状态，避免历史动量干扰
        self._reset_optimizer_state()
        
        with self._reconnect_lock:
            self._is_reconnecting = False
        return

def _reset_optimizer_state(self):
    """
    Reset optimizer state after reconnection to avoid stale momentum
    interfering with the newly synchronized model parameters.
    """
    if hasattr(self.trainer, 'ctx') and hasattr(self.trainer.ctx, 'optimizer'):
        optimizer = self.trainer.ctx.optimizer
        if optimizer is not None:
            # Clear all parameter states (momentum_buffer, exp_avg, etc.)
            optimizer.state.clear()
            logger.info(f'Client #{self.ID}: Optimizer state reset after '
                        f'reconnection')
```

### 4.3 方案优势
- 最小侵入：仅在重连成功时触发，不影响正常训练流程
- 通用性强：`optimizer.state.clear()` 适用于所有 PyTorch 优化器
- 保留超参数：不影响 learning rate、momentum 等配置值

## 5. 文件修改清单
| 文件 | 修改内容 |
|------|----------|
| `core/workers/client.py` | 新增 `_reset_optimizer_state()` 方法，在重连成功后调用 |

## 6. 测试验证

### 6.1 测试场景
- 使用 SGD（momentum=0.9）训练 FEMNIST 数据集
- 在第 5 轮手动断开客户端 1，等待其自动重连
- 对比修复前后的 loss 曲线

### 6.2 测试结果
| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 重连后第 1 轮 loss | 2.34（异常飙升） | 0.52（正常） |
| 重连后第 5 轮 loss | 1.87（缓慢下降） | 0.31（正常收敛） |

### 6.3 验证结论
修复后重连客户端的梯度更新方向恢复正常，不再对全局模型收敛造成负面影响。

## 7. 遗留问题
- 问题 1（重连竞态条件导致死锁）仍在排查中
- 问题 3（终止超时不准确）仍存在部分问题

## 8. 下周计划
- 继续排查问题 1 的锁竞争问题，尝试引入 `RLock` 或调整锁获取顺序
- 为问题 3 添加独立的超时检查线程，避免依赖训练轮次触发
