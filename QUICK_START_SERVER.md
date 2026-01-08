# 快速上传和运行指南

## 📦 方法 1: 使用 PowerShell 脚本（推荐）

在本地 Windows 上，打开 PowerShell，运行：

```powershell
cd D:\Projects\FederatedScope
.\upload_model.ps1
```

脚本会自动：
- ✓ 检查模型文件是否存在
- ✓ 上传模型到服务器
- ✓ 验证上传成功

---

## 📦 方法 2: 手动上传

### 在本地 Windows PowerShell 中运行：

```powershell
scp "C:\Users\Dbook\Downloads\open_clip_vitb16.bin" root@10.112.81.135:/root/FederatedScope/
```

### 验证上传成功：

```bash
ssh root@10.112.81.135
ls -lh /root/FederatedScope/open_clip_vitb16.bin
```

应该看到文件大小（约 300-400 MB）

---

## 🚀 在服务器上运行

### 1. 登录服务器
```bash
ssh root@10.112.81.135
cd /root/FederatedScope
```

### 2. 同步代码（选择一种方法）

**方法 A: 如果使用 git**
```bash
git add .
git commit -m "Update GGEUR config for ViT-B-16"
git pull  # 或 git push 然后在服务器上 pull
```

**方法 B: 直接同步整个目录**
在本地 PowerShell 运行：
```powershell
# 使用 rsync（如果安装了）
rsync -avz --exclude '.git' --exclude '__pycache__' `
  D:/Projects/FederatedScope/ root@10.112.81.135:/root/FederatedScope/

# 或使用 scp（较慢但更兼容）
scp -r D:/Projects/FederatedScope/federatedscope root@10.112.81.135:/root/FederatedScope/
scp -r D:/Projects/FederatedScope/scripts root@10.112.81.135:/root/FederatedScope/
```

### 3. 安装依赖（如果还没安装）
```bash
pip install open-clip-torch
```

### 4. 清除旧缓存（重要！）
```bash
rm -rf exp/ggeur_officehome_lds/clip_cache
```

### 5. 验证配置
```bash
cat scripts/example_configs/ggeur_officehome_lds.yaml | grep -A5 "ggeur:"
```

应该看到：
```yaml
ggeur:
  clip_model: ViT-B/16
  clip_pretrained: ''
  clip_custom_weights: /root/FederatedScope/open_clip_vitb16.bin
```

### 6. 运行实验
```bash
python run_ggeur.py --cfg scripts/example_configs/ggeur_officehome_lds.yaml
```

---

## ✅ 检查日志

实验开始时应该看到：

```
Loading open_clip model: ViT-B/16
Loading custom weights from: /root/FederatedScope/open_clip_vitb16.bin
Successfully loaded custom weights
open_clip model loaded: ViT-B-16, embedding dim: 512
```

如果看到这些，说明模型加载成功！✅

---

## ⚠️ 常见问题

### 问题 1: 上传失败
```
scp: Connection refused
```
**解决**: 检查服务器 IP 和 SSH 是否可访问

### 问题 2: 权限问题
```
Permission denied
```
**解决**: 确保有 root 权限或使用正确的用户

### 问题 3: 模型加载失败
```
Failed to load open_clip model
```
**解决**:
1. 检查模型文件路径是否正确
2. 检查模型文件是否完整（大小约 300-400 MB）
3. 尝试在 Python 中手动加载：
```python
import torch
checkpoint = torch.load('/root/FederatedScope/open_clip_vitb16.bin')
print(checkpoint.keys())
```

---

## 📊 预期结果

修复后应该看到：

```
Round 1:
  CLIENT #1 - DATA DISTRIBUTION REPORT
    Augmented data: 3250 samples ✅
  Test Acc: 20-30% ✅

Round 10:
  Test Acc: 40-50% ✅

Round 50:
  Test Acc: 55-65% ✅
```

如果准确率还是很低（<5%），请提供：
1. 模型加载的日志
2. 第1轮的完整输出
3. 数据分布报告
