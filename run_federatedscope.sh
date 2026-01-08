#!/bin/bash
# FederatedScope 启动脚本 - 强制使用 CDA_new 目录的代码
# 解决 Python 导入路径问题

# 获取脚本所在目录（即 CDA_new 目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "=========================================="
echo "FederatedScope 启动脚本"
echo "=========================================="
echo "使用目录: $SCRIPT_DIR"
echo ""

# 清除可能存在的 PYTHONPATH
unset PYTHONPATH

# 设置 PYTHONPATH 为当前目录，优先级最高
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# 验证导入路径
echo "验证 Python 导入路径:"
python -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
import federatedscope
import os
actual_path = federatedscope.__file__
module_dir = os.path.dirname(os.path.dirname(actual_path))
print(f'  federatedscope 导入自: {module_dir}')

if module_dir == '$SCRIPT_DIR':
    print('  ✓ 路径正确！')
else:
    print('  ✗ 路径错误！')
    print(f'  期望: $SCRIPT_DIR')
    print(f'  实际: {module_dir}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ 导入路径验证失败"
    echo "请检查 Python 环境"
    exit 1
fi

echo ""
echo "=========================================="
echo "开始运行 FederatedScope"
echo "=========================================="
echo ""

# 运行 FederatedScope，传递所有参数
cd "$SCRIPT_DIR"
exec python -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
# 运行 main.py
exec(open('federatedscope/main.py').read())
" "$@"
