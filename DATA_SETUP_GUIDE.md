# Office-Caltech-10 数据集配置指南

## 问题诊断

通过运行 `python diagnose_server_test_data.py`，发现配置文件中的数据路径不存在：
- 配置路径: `/root/data/office_caltech_10` (Linux 路径)
- 当前系统: Windows

## 解决方案

### 步骤 1: 下载 Office-Caltech-10 数据集

Office-Caltech-10 数据集包含 4 个域的 10 类图像数据：
- **amazon**: 来自 Amazon.com 的图片
- **webcam**: 来自网络摄像头的低分辨率图片
- **dslr**: 来自数码单反相机的高分辨率图片
- **caltech**: 来自 Caltech-256 的图片

**下载链接:**
- Google Drive: https://drive.google.com/file/d/14OIlzWFmi5455AjeBZLak2Ku-cFUrfEo/view
- 百度网盘: https://pan.baidu.com/s/14JEGQ56LJX7LMbd6GLtxCw (提取码: aole)

**或者使用官方链接:**
1. 访问: https://people.eecs.berkeley.edu/~jhoffman/domainadapt/
2. 下载 Office-Caltech-10 数据集

### 步骤 2: 解压数据集

将下载的数据集解压到本地目录，例如:
```
D:\Data\office_caltech_10\
```

解压后的目录结构应该是:
```
D:\Data\office_caltech_10\
├── amazon\
│   ├── back_pack\
│   ├── bike\
│   ├── calculator\
│   ├── headphones\
│   ├── keyboard\
│   ├── laptop_computer\
│   ├── monitor\
│   ├── mouse\
│   ├── mug\
│   └── projector\
├── webcam\
│   ├── back_pack\
│   ├── ...
├── dslr\
│   ├── back_pack\
│   ├── ...
└── caltech\
    ├── back_pack\
    ├── ...
```

### 步骤 3: 修改配置文件

编辑 `scripts/example_configs/cross_domain_adaptive_office_caltech.yaml`，修改数据路径:

```yaml
data:
  root: 'D:/Data/office_caltech_10'  # 改为你的实际路径
  type: 'office_caltech'
  splits: [0.8, 0.1, 0.1]
  batch_size: 4
  shuffle: True
  num_workers: 0
  dirichlet_alpha: 0.1
  server_test_samples_per_class: 10  # 服务器端每类测试样本数
```

**注意:**
- Windows 路径使用正斜杠 `/` 或双反斜杠 `\\`
- 推荐使用正斜杠: `D:/Data/office_caltech_10`
- 避免使用单反斜杠: `D:\Data\office_caltech_10` (可能导致转义问题)

### 步骤 4: 验证数据配置

运行诊断脚本验证数据加载:
```bash
python diagnose_server_test_data.py
```

如果一切正常，应该看到:
```
[成功] 所有检查通过！服务器端测试数据加载应该可以正常工作。
```

### 步骤 5: 运行训练

配置正确后，运行训练:
```bash
python federatedscope/main.py --cfg scripts/example_configs/cross_domain_adaptive_office_caltech.yaml
```

## 服务器端测试集说明

配置中的 `server_test_samples_per_class: 10` 确保:
1. 服务器端为每个域创建类别均衡的测试集
2. 每个类别精确包含 10 个样本
3. 总共 4 个域 × 10 个类别 × 10 个样本/类 = 400 个测试样本

这样可以公平地评估模型在不同域上的性能，避免类别不平衡导致的偏差。

## 常见问题

### Q1: 数据路径不存在
**错误:** `[错误] 数据根目录不存在`

**解决:**
- 确认已下载并解压数据集
- 修改配置文件中的 `data.root` 为实际路径
- 使用正斜杠或双反斜杠

### Q2: 域目录不存在
**错误:** `Domain directory not found`

**解决:**
- 检查解压后的目录结构
- 确保包含 amazon, webcam, dslr, caltech 四个文件夹
- 每个文件夹内应包含 10 个类别的子文件夹

### Q3: 类别目录不存在
**错误:** `Class directory not found`

**解决:**
- 检查每个域文件夹内是否包含所有 10 个类别
- 类别名称必须完全匹配: back_pack, bike, calculator, headphones, keyboard, laptop_computer, monitor, mouse, mug, projector

### Q4: 评估时没有数据
**警告:** `No data for evaluation in split 'test', returning zero metrics`

**可能原因:**
1. 服务器端没有正确加载测试数据
2. Standalone 模式下服务器可能没有 trainer
3. `federate.make_global_eval` 未设置为 True

**解决:**
- 确保配置中 `federate.make_global_eval: True`
- 运行 `diagnose_server_test_data.py` 验证数据加载
- 检查日志中的服务器端评估信息

## 性能优化建议

### 数据加载
- `num_workers: 0` 适合 Windows 系统，避免多进程问题
- 如果在 Linux 上运行，可以设置为 2-4 提高速度

### 批次大小
- `batch_size: 4` 是保守设置，适合小显存
- 如果显存足够（>= 8GB），可以增加到 16 或 32

### 样本数量
- `server_test_samples_per_class: 10` 每类 10 个样本，总共 400 个
- 可以根据需要调整（5-20 都是合理范围）
- 更多样本 = 更准确的评估，但速度更慢

## 示例: 完整的数据配置

```yaml
data:
  root: 'D:/Data/office_caltech_10'  # 你的数据路径
  type: 'office_caltech'
  splits: [0.8, 0.1, 0.1]  # 训练:验证:测试 = 8:1:1
  batch_size: 4
  shuffle: True
  num_workers: 0  # Windows 使用 0，Linux 可用 2-4
  dirichlet_alpha: 0.1  # Non-IID 程度，越小越不均匀
  server_test_samples_per_class: 10  # 服务器端每类样本数

federate:
  mode: 'standalone'
  client_num: 20
  total_round_num: 300
  sample_client_num: 5
  method: 'cross_domain_adaptive'
  make_global_eval: True  # 启用服务器端全局评估
  share_local_model: False
  online_aggr: False
```

## 后续步骤

1. ✓ 已修复空评估数据错误
2. ✓ 已实现服务器端类别均衡评估
3. ⚠ **需要配置数据路径** (当前步骤)
4. 运行训练并观察服务器端评估结果
5. 调整超参数优化性能
