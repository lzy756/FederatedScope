#!/bin/bash
# Bus Error 诊断脚本

echo "========================================="
echo "Bus Error 诊断工具"
echo "========================================="

# 1. 检查GPU显存
echo -e "\n[1] GPU显存状态:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used --format=csv
else
    echo "未检测到nvidia-smi，跳过GPU检查"
fi

# 2. 检查系统内存
echo -e "\n[2] 系统内存状态:"
free -h

# 3. 检查数据路径
echo -e "\n[3] 检查PACS数据集路径:"
DATA_ROOT="/root/data/PACS"
if [ -d "$DATA_ROOT" ]; then
    echo "✓ 数据根目录存在: $DATA_ROOT"
    if [ -d "$DATA_ROOT/pacs_data" ]; then
        echo "✓ pacs_data目录存在"
        ls -lh "$DATA_ROOT/pacs_data/"
    else
        echo "✗ pacs_data目录不存在！"
    fi
else
    echo "✗ 数据根目录不存在: $DATA_ROOT"
fi

# 4. 检查PyTorch
echo -e "\n[4] 检查PyTorch环境:"
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name}')
        print(f'  总显存: {props.total_memory / 1024**3:.2f} GB')
        print(f'  已用显存: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB')
"

echo -e "\n========================================="
echo "诊断完成"
echo "========================================="
