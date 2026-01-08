# 服务器运行步骤

## ✅ 模型已上传
路径：`~/model/open_clip_vitb16.bin`

## 🔧 配置已更新
配置文件已修改为使用 `~/model/open_clip_vitb16.bin`

---

## 🚀 在服务器上运行（完整步骤）

### 1. 登录服务器
```bash
ssh root@10.112.81.135
```

### 2. 验证模型文件
```bash
ls -lh ~/model/open_clip_vitb16.bin
```
应该看到文件大小（约 300-400 MB）

### 3. 进入项目目录
```bash
cd ~/FederatedScope
# 或者
cd /root/FederatedScope
```

### 4. 同步配置文件（选择一种方法）

**方法 A: 手动编辑（推荐）**
```bash
vim scripts/example_configs/ggeur_officehome_lds.yaml
```
找到 `ggeur:` 部分，确保是：
```yaml
ggeur:
  clip_model: ViT-B/16
  clip_pretrained: ''
  clip_custom_weights: ~/model/open_clip_vitb16.bin
```

**方法 B: 从本地上传配置**
在本地 PowerShell 运行：
```powershell
scp scripts/example_configs/ggeur_officehome_lds.yaml root@10.112.81.135:~/FederatedScope/scripts/example_configs/
```

**方法 C: 使用 git**
```bash
git pull  # 如果代码在 git 仓库中
```

### 5. 验证配置
```bash
cat scripts/example_configs/ggeur_officehome_lds.yaml | grep -A5 "clip_"
```
应该看到：
```yaml
  clip_model: ViT-B/16
  clip_pretrained: ''
  clip_custom_weights: ~/model/open_clip_vitb16.bin
```

### 6. 安装依赖（如果还没安装）
```bash
pip install open-clip-torch
```

验证安装：
```bash
python -c "import open_clip; print('open_clip installed successfully')"
```

### 7. 清除旧缓存（重要！）
```bash
rm -rf exp/ggeur_officehome_lds/clip_cache
```

### 8. 测试模型加载（可选但推荐）
```bash
python -c "
import torch
import open_clip

print('Testing model loading...')
model_path = '~/model/open_clip_vitb16.bin'
model_path = model_path.replace('~', '/root')  # 展开波浪号

# 创建模型
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-16',
    pretrained=None,
    device='cpu'
)

# 加载权重
checkpoint = torch.load(model_path, map_location='cpu')
print(f'Checkpoint keys: {list(checkpoint.keys())[:5]}...')

# 尝试加载
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
elif 'model' in checkpoint:
    state_dict = checkpoint['model']
else:
    state_dict = checkpoint

model.load_state_dict(state_dict, strict=False)
print('✓ Model loaded successfully!')
print(f'Embedding dim: {model.visual.output_dim}')
"
```

### 9. 运行实验
```bash
python run_ggeur.py --cfg scripts/example_configs/ggeur_officehome_lds.yaml
```

---

## 📊 检查日志输出

### 启动时应该看到：
```
Loading open_clip model: ViT-B/16
Loading custom weights from: /root/model/open_clip_vitb16.bin
Successfully loaded custom weights
open_clip model loaded: ViT-B-16, embedding dim: 512
```

### 第1轮应该看到：
```
CLIENT #1 - DATA DISTRIBUTION REPORT
  Original data: XXX samples
  Augmented data: 3250 samples
  Per-class distribution:
    Class 0: 50 samples
    Class 1: 50 samples
    ...

Round 1:
  Test Acc: 0.20-0.30 (20-30%)  ← 应该在这个范围！
```

---

## ⚠️ 如果出现问题

### 问题 1: 模型加载失败
```
FileNotFoundError: ~/model/open_clip_vitb16.bin
```
**原因**: 波浪号 `~` 可能没有正确展开

**解决**:
```bash
# 检查实际路径
echo ~
# 应该输出 /root

# 检查文件
ls -lh ~/model/open_clip_vitb16.bin
ls -lh /root/model/open_clip_vitb16.bin

# 如果需要，改用绝对路径
vim scripts/example_configs/ggeur_officehome_lds.yaml
# 改为: clip_custom_weights: /root/model/open_clip_vitb16.bin
```

### 问题 2: 准确率还是很低 (<5%)
**检查**:
1. 模型是否成功加载（看日志中的 "Successfully loaded custom weights"）
2. CLIP 特征维度是否正确（应该是 512）
3. 数据增强是否成功（应该有 3250 samples）

**调试**:
```bash
# 查看详细日志
python run_ggeur.py --cfg scripts/example_configs/ggeur_officehome_lds.yaml 2>&1 | tee ggeur_debug.log

# 检查关键信息
grep -i "load.*weight\|clip.*model\|augment.*data" ggeur_debug.log
```

### 问题 3: open_clip 未安装
```
ModuleNotFoundError: No module named 'open_clip'
```
**解决**:
```bash
pip install open-clip-torch
```

---

## 💡 小贴士

1. **使用 tmux/screen 避免断线**:
```bash
tmux new -s ggeur
# 运行实验
python run_ggeur.py --cfg scripts/example_configs/ggeur_officehome_lds.yaml

# 断开: Ctrl+B 然后按 D
# 重新连接: tmux attach -t ggeur
```

2. **后台运行**:
```bash
nohup python run_ggeur.py --cfg scripts/example_configs/ggeur_officehome_lds.yaml > ggeur.log 2>&1 &

# 查看日志
tail -f ggeur.log
```

3. **监控 GPU 使用**:
```bash
# 另开一个终端
watch -n 1 nvidia-smi
```

---

现在可以直接在服务器上运行步骤 3-9 了！🚀
