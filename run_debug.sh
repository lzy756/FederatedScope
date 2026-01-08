#!/bin/bash
# Bus Error 调试启动脚本

echo "========================================="
echo "开始运行CDA - Bus Error调试模式"
echo "========================================="

# 设置环境变量
export CUDA_LAUNCH_BLOCKING=1
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1

# 检查数据集
echo -e "\n[检查] PACS数据集:"
if [ -d "/root/data/PACS/pacs_data" ]; then
    echo "✓ 数据集目录存在"
    for domain in photo art_painting cartoon sketch; do
        if [ -d "/root/data/PACS/pacs_data/$domain" ]; then
            count=$(find /root/data/PACS/pacs_data/$domain -type f | wc -l)
            echo "  - $domain: $count 个文件"
        else
            echo "  ✗ $domain 目录不存在"
        fi
    done
else
    echo "✗ 数据集目录不存在: /root/data/PACS/pacs_data"
    exit 1
fi

# 检查GPU
echo -e "\n[检查] GPU状态:"
nvidia-smi --query-gpu=index,name,memory.free --format=csv

# 清理GPU缓存
echo -e "\n[清理] GPU缓存..."
python -c "import torch; torch.cuda.empty_cache()"

# 运行配置
CONFIG="${1:-scripts/example_configs/cda_pacs_minimal_debug.yaml}"

echo -e "\n[运行] 配置文件: $CONFIG"
echo "========================================="
echo ""

# 运行并捕获错误
python -u federatedscope/main.py --cfg "$CONFIG" 2>&1 | tee run_debug.log

# 检查退出状态
if [ $? -ne 0 ]; then
    echo ""
    echo "========================================="
    echo "✗ 运行失败"
    echo "========================================="
    echo "错误日志已保存到: run_debug.log"
    echo ""
    echo "最后20行日志:"
    tail -20 run_debug.log
    exit 1
else
    echo ""
    echo "========================================="
    echo "✓ 运行成功"
    echo "========================================="
    exit 0
fi
