# FederatedScope 分布式训练快速指南

## 🚀 快速开始

### 启动训练
```bash
cd /home/lzy/FederatedScope
./scripts/distributed_scripts/run_distributed_lr_managed.sh
```

### 停止训练
**方法1**: 在运行终端按 `Ctrl+C`

**方法2**: 在另一个终端运行
```bash
./scripts/distributed_scripts/stop_distributed.sh
```

---

## 📝 查看日志

训练日志保存在 `exp/` 目录下，按时间戳自动创建子目录：

```bash
# 查看最新的实验目录
ls -lt exp/FedAvg_lr_on_file_lr0.1_lstep1/

# 查看 Server 日志
tail -f exp/FedAvg_lr_on_file_lr0.1_lstep1/sub_exp_YYYYMMDDHHMMSS/exp_print.log

# 查看所有进程
ps aux | grep federatedscope/main.py
```

---

## ⚠️ 常见问题

### 问题1: 进程残留
**症状**: 重新运行时报错端口已被占用

**解决**:
```bash
./scripts/distributed_scripts/stop_distributed.sh
```

### 问题2: 训练卡住
**症状**: 长时间没有输出

**检查**:
1. 查看是否有 "Timeout waiting for clients" 错误
2. 确认所有3个客户端都启动了
3. 检查网络连接（虽然是本地，但gRPC可能有问题）

**解决**: Ctrl+C 停止，重新运行

### 问题3: 客户端ID为-1
**症状**: 日志中出现 "Client #-1"

**说明**: 如果您看到此问题，说明修复未生效。请确认：
1. 已应用所有代码修改
2. 重新启动了所有进程（杀死旧进程）

---

## 📂 文件结构

```
FederatedScope/
├── scripts/distributed_scripts/
│   ├── run_distributed_lr.sh              # 原始脚本（不推荐使用）
│   ├── run_distributed_lr_managed.sh      # 新脚本（推荐） ✅
│   ├── stop_distributed.sh                # 停止脚本 ✅
│   ├── REPORT.md                          # 详细报告 📄
│   ├── QUICKSTART.md                      # 本文件 📄
│   └── distributed_configs/
│       ├── distributed_server.yaml
│       ├── distributed_client_1.yaml
│       ├── distributed_client_2.yaml
│       └── distributed_client_3.yaml
├── exp/                                    # 训练输出目录
└── toy_data/                              # 训练数据
```

---

## 🔧 修改的文件

如果需要回滚或重新应用修改，以下是修改的核心文件：

1. `federatedscope/core/workers/client.py` - 消息缓冲机制
2. `federatedscope/core/workers/server.py` - 超时、日志、诊断
3. `federatedscope/core/auxiliaries/logging.py` - 配置冻结处理

---

## 📊 预期输出

**正常启动**:
```
============================================
Starting FederatedScope Distributed Training
============================================
PID file: /tmp/federatedscope_distributed_pids.txt

Generating data...

Starting processes...

1. Starting server...
   Server PID: 12345
2. Starting client 1...
   Client 1 PID: 12346
3. Starting client 2...
   Client 2 PID: 12347
4. Starting client 3...
   Client 3 PID: 12348

============================================
All processes started successfully!
============================================
```

**正常训练日志**:
```
Server: Starting training (Round #0)
Client #1, Round: 0, Results_raw: {'train_loss': ..., 'train_avg_loss': ...}
Client #2, Round: 0, Results_raw: {'train_loss': ..., 'train_avg_loss': ...}
Client #3, Round: 0, Results_raw: {'train_loss': ..., 'train_avg_loss': ...}
Server: Starting training (Round #1)
...
```

**正常完成**:
```
Server: Training is finished! Starting evaluation.
Server: Final evaluation is finished! Starting merging results.
```

---

## 💡 提示

1. **首次运行**: 会自动生成 `toy_data/all_data` 数据
2. **日志位置**: 配置文件中指定，默认在 `exp/` 目录
3. **进程追踪**: PID保存在 `/tmp/federatedscope_distributed_pids.txt`
4. **清理环境**: 运行 `stop_distributed.sh` 确保干净启动

---

## 📖 更多信息

- 详细报告: [REPORT.md](./REPORT.md)
- 配置说明: 查看 `distributed_configs/*.yaml` 文件
- FederatedScope 文档: https://github.com/alibaba/FederatedScope

---

**更新日期**: 2025-11-06
