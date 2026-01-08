#!/bin/bash
# 全自动配置修复脚本 - 智能检测并修复正确的代码库
# 解决多个 FederatedScope 目录导致的问题

set -e

echo "=========================================="
echo "FederatedScope 配置自动修复工具"
echo "=========================================="

# 1. 首先运行诊断脚本
echo ""
echo "[步骤 1/4] 运行诊断，找出 Python 实际使用的代码库..."
echo ""

DIAGNOSTIC_OUTPUT=$(python3 << 'PYEOF'
import sys
import os

# 找出 Python 实际导入的 federatedscope 路径
try:
    import federatedscope
    actual_path = federatedscope.__file__
    module_dir = os.path.dirname(os.path.dirname(actual_path))
    print(module_dir)
except ImportError:
    # 如果无法导入，尝试从当前目录
    print(os.getcwd())
PYEOF
)

TARGET_DIR="$DIAGNOSTIC_OUTPUT"
echo "检测到的 FederatedScope 目录: $TARGET_DIR"

if [ ! -d "$TARGET_DIR/federatedscope/core/configs" ]; then
    echo "错误：无法找到配置目录"
    echo "请手动指定 FederatedScope 根目录："
    echo "  bash $0 /path/to/FederatedScope"
    exit 1
fi

# 如果用户提供了参数，使用用户指定的目录
if [ $# -eq 1 ]; then
    TARGET_DIR="$1"
    echo "使用用户指定的目录: $TARGET_DIR"
fi

echo ""
echo "将在以下目录进行修复:"
echo "  $TARGET_DIR"
echo ""
read -p "确认继续? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 1
fi

cd "$TARGET_DIR"

# 2. 备份原文件
echo ""
echo "[步骤 2/4] 备份原始配置文件..."
BACKUP_DIR=".config_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

for file in \
    federatedscope/core/configs/cfg_data.py \
    federatedscope/core/configs/cfg_model.py \
    federatedscope/core/configs/cfg_fedlsa.py; do
    if [ -f "$file" ]; then
        cp "$file" "$BACKUP_DIR/"
        echo "  ✓ 已备份: $file"
    fi
done

# 3. 修复配置文件
echo ""
echo "[步骤 3/4] 修复配置文件..."

# 3.1 修复 cfg_data.py
echo ""
echo "  [3.1] 修复 cfg_data.py ..."
if grep -q "cfg.data.server_test_samples_per_class" federatedscope/core/configs/cfg_data.py 2>/dev/null; then
    echo "    ✓ server_test_samples_per_class 已存在"
else
    # 找到 dirichlet_alpha 行并在其后插入新配置
    python3 << 'PYEOF'
import re

with open('federatedscope/core/configs/cfg_data.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
inserted = False

for i, line in enumerate(lines):
    new_lines.append(line)
    # 在 dirichlet_alpha 的注释行之后插入
    if 'dirichlet_alpha' in line and '# 0.5 = moderate heterogeneity' in line:
        new_lines.append('    cfg.data.server_test_samples_per_class = 10  # Number of samples per class\n')
        new_lines.append('    # for server-side balanced test sets (used in domain-specific evaluation)\n')
        new_lines.append('\n')
        inserted = True
        print('    ✓ 已添加 server_test_samples_per_class 配置')
        break

if not inserted:
    print('    ✗ 警告: 无法自动添加，请手动修改')
else:
    # 添加剩余的行
    new_lines.extend(lines[i+1:])

    with open('federatedscope/core/configs/cfg_data.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
PYEOF
fi

# 3.2 修复 cfg_model.py
echo ""
echo "  [3.2] 修复 cfg_model.py ..."
if grep -q "cfg.model.num_classes" federatedscope/core/configs/cfg_model.py 2>/dev/null; then
    echo "    ✓ num_classes 已存在"
else
    python3 << 'PYEOF'
with open('federatedscope/core/configs/cfg_model.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
inserted = False

for i, line in enumerate(lines):
    new_lines.append(line)
    if 'cfg.model.hidden = 256' in line:
        new_lines.append('    cfg.model.num_classes = 10  # Number of classes for classification tasks\n')
        inserted = True
        print('    ✓ 已添加 num_classes 配置')

if not inserted:
    print('    ✗ 警告: 无法自动添加，请手动修改')
else:
    with open('federatedscope/core/configs/cfg_model.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
PYEOF
fi

# 3.3 修复 cfg_fedlsa.py
echo ""
echo "  [3.3] 修复 cfg_fedlsa.py ..."
if grep -q "hasattr(cfg.model" federatedscope/core/configs/cfg_fedlsa.py 2>/dev/null; then
    echo "    需要移除错误的 hasattr 代码..."
    python3 << 'PYEOF'
with open('federatedscope/core/configs/cfg_fedlsa.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip_next = False

for line in lines:
    # 跳过包含 hasattr 和设置 num_classes 的行
    if 'if not hasattr(cfg.model' in line:
        skip_next = True
        # 添加修复后的注释
        new_lines.append('    # Note: This is handled by the model config, we just document it here\n')
        new_lines.append('    # cfg.model.num_classes should be set by the model configuration\n')
        print('    ✓ 已移除错误的 hasattr 代码并添加说明')
        continue

    if skip_next and 'cfg.model.num_classes = 10' in line:
        skip_next = False
        continue

    new_lines.append(line)

with open('federatedscope/core/configs/cfg_fedlsa.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
PYEOF
else
    echo "    ✓ 无需修复"
fi

# 4. 验证修复结果
echo ""
echo "[步骤 4/4] 验证修复结果..."

python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')

print("\n检查配置文件内容:")

# 检查 cfg_data.py
with open('federatedscope/core/configs/cfg_data.py', 'r') as f:
    content = f.read()
    has_config = 'server_test_samples_per_class' in content
    print(f"  {'✓' if has_config else '✗'} cfg_data.py 包含 server_test_samples_per_class")

# 检查 cfg_model.py
with open('federatedscope/core/configs/cfg_model.py', 'r') as f:
    content = f.read()
    has_config = 'cfg.model.num_classes' in content
    print(f"  {'✓' if has_config else '✗'} cfg_model.py 包含 num_classes")

# 检查 cfg_fedlsa.py
with open('federatedscope/core/configs/cfg_fedlsa.py', 'r') as f:
    content = f.read()
    has_error = 'hasattr(cfg.model' in content
    print(f"  {'✓' if not has_error else '✗'} cfg_fedlsa.py {'包含错误!' if has_error else '已修复'}")

print("\n测试配置加载:")
try:
    from federatedscope.core.configs.config import global_cfg
    cfg = global_cfg.clone()

    has_server_test = hasattr(cfg.data, 'server_test_samples_per_class')
    has_num_classes = hasattr(cfg.model, 'num_classes')

    print(f"  {'✓' if has_server_test else '✗'} cfg.data.server_test_samples_per_class: {cfg.data.server_test_samples_per_class if has_server_test else 'N/A'}")
    print(f"  {'✓' if has_num_classes else '✗'} cfg.model.num_classes: {cfg.model.num_classes if has_num_classes else 'N/A'}")

    if has_server_test and has_num_classes:
        print("\n✓✓✓ 配置修复成功！✓✓✓")
        sys.exit(0)
    else:
        print("\n✗ 配置加载失败，请检查错误")
        sys.exit(1)

except Exception as e:
    print(f"\n✗ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYEOF

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ 修复完成！"
    echo "=========================================="
    echo ""
    echo "备份文件保存在: $TARGET_DIR/$BACKUP_DIR"
    echo ""
    echo "现在可以运行:"
    echo "  cd $TARGET_DIR"
    echo "  python federatedscope/main.py --cfg scripts/example_configs/fedavg_office_caltech_domain_eval.yaml"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "✗ 修复失败"
    echo "=========================================="
    echo ""
    echo "可以从备份恢复:"
    echo "  cp $TARGET_DIR/$BACKUP_DIR/* $TARGET_DIR/federatedscope/core/configs/"
    echo ""
    exit 1
fi

