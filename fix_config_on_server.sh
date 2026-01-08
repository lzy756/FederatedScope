#!/bin/bash
# 配置文件修复脚本 - 在远程服务器上运行
# 使用方法：bash fix_config_on_server.sh

set -e  # 遇到错误立即退出

echo "=========================================="
echo "开始修复 FederatedScope 配置文件"
echo "=========================================="

# 1. 修复 cfg_data.py
echo ""
echo "[1/3] 修复 federatedscope/core/configs/cfg_data.py ..."
if grep -q "server_test_samples_per_class" federatedscope/core/configs/cfg_data.py; then
    echo "  ✓ server_test_samples_per_class 已存在，跳过"
else
    # 在 dirichlet_alpha 后添加新配置
    sed -i '/cfg.data.dirichlet_alpha/a\    cfg.data.server_test_samples_per_class = 10  # Number of samples per class\n    # for server-side balanced test sets (used in domain-specific evaluation)\n' federatedscope/core/configs/cfg_data.py
    echo "  ✓ 已添加 server_test_samples_per_class 配置"
fi

# 2. 修复 cfg_model.py
echo ""
echo "[2/3] 修复 federatedscope/core/configs/cfg_model.py ..."
if grep -q "cfg.model.num_classes" federatedscope/core/configs/cfg_model.py; then
    echo "  ✓ num_classes 已存在，跳过"
else
    # 在 hidden 后添加新配置
    sed -i '/cfg.model.hidden = 256/a\    cfg.model.num_classes = 10  # Number of classes for classification tasks' federatedscope/core/configs/cfg_model.py
    echo "  ✓ 已添加 num_classes 配置"
fi

# 3. 修复 cfg_fedlsa.py
echo ""
echo "[3/3] 修复 federatedscope/core/configs/cfg_fedlsa.py ..."
if grep -q "hasattr(cfg.model, 'num_classes')" federatedscope/core/configs/cfg_fedlsa.py; then
    echo "  需要移除错误的 hasattr 代码..."
    # 创建备份
    cp federatedscope/core/configs/cfg_fedlsa.py federatedscope/core/configs/cfg_fedlsa.py.bak

    # 删除包含 hasattr 和 cfg.model.num_classes 的行
    sed -i '/if not hasattr(cfg.model/d' federatedscope/core/configs/cfg_fedlsa.py
    sed -i '/cfg.model.num_classes = 10/d' federatedscope/core/configs/cfg_fedlsa.py

    # 在 "类别数量" 注释后添加说明
    sed -i '/# 类别数量.*用于锚点初始化/a\    # Note: This is handled by the model config, we just document it here\n    # cfg.model.num_classes should be set by the model configuration' federatedscope/core/configs/cfg_fedlsa.py

    echo "  ✓ 已修复 cfg_fedlsa.py（备份保存在 cfg_fedlsa.py.bak）"
else
    echo "  ✓ cfg_fedlsa.py 无需修复"
fi

# 4. 验证修改
echo ""
echo "=========================================="
echo "验证修改结果"
echo "=========================================="

echo ""
echo "检查 server_test_samples_per_class:"
if grep -q "cfg.data.server_test_samples_per_class" federatedscope/core/configs/cfg_data.py; then
    echo "  ✓ cfg_data.py 包含 server_test_samples_per_class"
else
    echo "  ✗ 错误：未找到 server_test_samples_per_class"
    exit 1
fi

echo ""
echo "检查 num_classes:"
if grep -q "cfg.model.num_classes" federatedscope/core/configs/cfg_model.py; then
    echo "  ✓ cfg_model.py 包含 num_classes"
else
    echo "  ✗ 错误：未找到 num_classes"
    exit 1
fi

echo ""
echo "检查 cfg_fedlsa.py 错误修复:"
if grep -q "hasattr(cfg.model" federatedscope/core/configs/cfg_fedlsa.py; then
    echo "  ✗ 错误：cfg_fedlsa.py 仍包含 hasattr 错误代码"
    exit 1
else
    echo "  ✓ cfg_fedlsa.py 已正确修复"
fi

# 5. 测试配置加载
echo ""
echo "=========================================="
echo "测试配置加载"
echo "=========================================="
python -c "
import sys
sys.path.insert(0, '.')
from federatedscope.core.configs.config import global_cfg

cfg = global_cfg.clone()
cfg.merge_from_file('scripts/example_configs/fedavg_office_caltech_domain_eval.yaml')

print('')
print('✓ 配置文件加载成功！')
print(f'✓ data.server_test_samples_per_class = {cfg.data.server_test_samples_per_class}')
print(f'✓ model.num_classes = {cfg.model.num_classes}')
print(f'✓ dataloader.batch_size = {cfg.dataloader.batch_size}')
print('')
" 2>&1 | grep -E "✓|配置"

echo ""
echo "=========================================="
echo "✓ 所有修复完成！"
echo "=========================================="
echo ""
echo "现在可以运行："
echo "  python federatedscope/main.py --cfg scripts/example_configs/fedavg_office_caltech_domain_eval.yaml"
echo ""
