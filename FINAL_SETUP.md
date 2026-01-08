# 最终配置总结

## ✅ 所有修改已完成

### 1. 配置文件
`scripts/example_configs/ggeur_officehome_lds.yaml`:
```yaml
ggeur:
  clip_model: ViT-B/16
  clip_pretrained: ''
  clip_custom_weights: ~/model/open_clip_vitb16.bin  # ✅ 自动展开 ~
```

### 2. 代码改进
- ✅ 使用 `open_clip` 库（替换 openai CLIP）
- ✅ 自动展开 `~` 路径（`os.path.expanduser`）
- ✅ 添加 fallback 机制（找不到自定义权重时使用默认权重）
- ✅ 更详细的日志输出

---

## 🚀 在服务器上运行（三步走）

### 步骤 1: 同步代码到服务器

在本地 PowerShell 运行：
```powershell
# 同步关键文件
scp scripts/example_configs/ggeur_officehome_lds.yaml root@10.112.81.135:~/FederatedScope/scripts/example_configs/

scp federatedscope/contrib/utils/clip_extractor.py root@10.112.81.135:~/FederatedScope/federatedscope/contrib/utils/

scp federatedscope/contrib/worker/ggeur_client.py root@10.112.81.135:~/FederatedScope/federatedscope/contrib/worker/

scp federatedscope/contrib/trainer/ggeur_trainer.py root@10.112.81.135:~/FederatedScope/federatedscope/contrib/trainer/

scp federatedscope/core/configs/cfg_ggeur.py root@10.112.81.135:~/FederatedScope/federatedscope/core/configs/
```

### 步骤 2: 登录服务器并准备环境

```bash
ssh root@10.112.81.135
cd ~/FederatedScope

# 验证模型文件
ls -lh ~/model/open_clip_vitb16.bin

# 安装依赖（如果还没安装）
pip install open-clip-torch

# 清除旧缓存
rm -rf exp/ggeur_officehome_lds/clip_cache
```

### 步骤 3: 运行实验

```bash
python run_ggeur.py --cfg scripts/example_configs/ggeur_officehome_lds.yaml
```

或者使用 tmux（推荐，防止断线）：
```bash
tmux new -s ggeur
python run_ggeur.py --cfg scripts/example_configs/ggeur_officehome_lds.yaml

# 断开: Ctrl+B 然后 D
# 重连: tmux attach -t ggeur
```

---

## 📊 预期输出

### 启动时（前几行）
```
INFO: Loading open_clip model: ViT-B/16
INFO: Loading custom weights from: /root/model/open_clip_vitb16.bin  ← ✅ 路径正确
INFO: Successfully loaded custom weights  ← ✅ 加载成功
INFO: open_clip model loaded: ViT-B-16, embedding dim: 512  ← ✅ 模型正确
```

### 数据增强阶段
```
================================================================================
CLIENT #1 - DATA DISTRIBUTION REPORT
================================================================================
Original data: XXX samples
Augmented data: 3250 samples  ← ✅ 应该约 3250

Per-class distribution:
  Class 0: 50 samples  ← ✅ 每个类 50 个样本
  Class 1: 50 samples
  ...
  Class 64: 50 samples
================================================================================
```

### 第1轮训练
```
Round 1 - CLIENT #1 - EVALUATION RESULTS
================================================================================
TEST     Accuracy: 0.25 (25%)  ← ✅ 应该在 20-30% 范围
================================================================================
```

**如果第1轮准确率 > 20%，说明修复成功！** 🎉

---

## ⚠️ 故障排查

### 如果看到这个警告：
```
WARNING: Custom weights file not found: /root/model/open_clip_vitb16.bin
WARNING: Falling back to pretrained weights: laion2b_s34b_b88k
```

**原因**: 模型文件路径不对

**解决**:
```bash
# 检查文件是否存在
ls -lh ~/model/open_clip_vitb16.bin

# 检查展开后的路径
python -c "import os; print(os.path.expanduser('~/model/open_clip_vitb16.bin'))"

# 如果不存在，检查实际位置
find ~ -name "open_clip_vitb16.bin"

# 更新配置为正确路径
vim scripts/example_configs/ggeur_officehome_lds.yaml
```

### 如果准确率还是很低 (<5%)：
```bash
# 1. 检查模型是否真的加载了
grep "Successfully loaded custom weights" 日志文件

# 2. 检查数据增强
grep "Augmented data:" 日志文件

# 3. 检查 CLIP 特征维度
grep "embedding dim" 日志文件

# 4. 查看完整的错误信息
python run_ggeur.py --cfg scripts/example_configs/ggeur_officehome_lds.yaml 2>&1 | tee debug.log
grep -i "error\|warning\|failed" debug.log
```

---

## 💡 优化建议

### 1. 使用后台运行
```bash
nohup python run_ggeur.py --cfg scripts/example_configs/ggeur_officehome_lds.yaml > ggeur.log 2>&1 &

# 查看进度
tail -f ggeur.log

# 查看实时准确率
tail -f ggeur.log | grep "Accuracy:"
```

### 2. 监控资源使用
```bash
# 新开一个终端
watch -n 1 nvidia-smi  # GPU
htop  # CPU 和内存
```

### 3. 保存结果
实验结果会自动保存在：
- `exp/ggeur_officehome_lds/` - 实验输出目录
- CLIP 缓存在 `exp/ggeur_officehome_lds/clip_cache/`

---

## 📝 文件清单

已修改的文件：
1. `scripts/example_configs/ggeur_officehome_lds.yaml` - 配置文件
2. `federatedscope/contrib/utils/clip_extractor.py` - CLIP 提取器
3. `federatedscope/core/configs/cfg_ggeur.py` - GGEUR 配置
4. `federatedscope/contrib/worker/ggeur_client.py` - 客户端
5. `federatedscope/contrib/trainer/ggeur_trainer.py` - 训练器

---

现在可以开始同步和运行了！如有问题随时反馈日志。🚀
