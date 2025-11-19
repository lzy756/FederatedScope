"""
数据集路径配置助手
帮助找到并配置正确的数据集路径
"""
import os
import sys

print("=" * 80)
print("Office-Caltech-10 数据集路径配置助手")
print("=" * 80)
print()

# 1. 询问数据集位置
print("请告诉我你的数据集实际位置。")
print()
print("常见位置示例:")
print("  1. /root/data/office_caltech_10")
print("  2. ~/data/office_caltech_10")
print("  3. /home/username/data/office_caltech_10")
print("  4. D:\\data\\office_caltech_10")
print("  5. 其他自定义路径")
print()

# 尝试自动检测一些常见位置
possible_paths = [
    "/root/data/office_caltech_10",
    os.path.expanduser("~/data/office_caltech_10"),
    os.path.join(os.getcwd(), "data", "office_caltech_10"),
    os.path.join(os.getcwd(), "root", "data", "office_caltech_10"),
    "D:\\data\\office_caltech_10",
    "C:\\data\\office_caltech_10",
]

print("正在自动检测可能的位置...")
found_paths = []
for path in possible_paths:
    if os.path.exists(path):
        # 检查是否包含domain目录
        domains = ['amazon', 'webcam', 'dslr', 'caltech']
        domain_found = sum(1 for d in domains if os.path.exists(os.path.join(path, d)))
        if domain_found > 0:
            found_paths.append((path, domain_found))
            print(f"  ✓ 找到: {path} (包含 {domain_found}/4 个域)")

print()

if found_paths:
    print("检测到以下可用路径:")
    for i, (path, count) in enumerate(found_paths, 1):
        print(f"  [{i}] {path} ({count}/4 个域)")
    print()

    choice = input(f"请选择路径 (1-{len(found_paths)}) 或输入 'c' 自定义路径: ").strip()

    if choice.lower() == 'c':
        dataset_path = input("请输入完整的数据集路径: ").strip()
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(found_paths):
                dataset_path = found_paths[idx][0]
            else:
                print("无效选择，请重新运行脚本")
                sys.exit(1)
        except ValueError:
            print("无效输入，请重新运行脚本")
            sys.exit(1)
else:
    print("未能自动检测到数据集路径")
    dataset_path = input("请输入完整的数据集路径: ").strip()

# 2. 验证路径
print()
print("=" * 80)
print("验证数据集路径")
print("=" * 80)

if not os.path.exists(dataset_path):
    print(f"✗ 路径不存在: {dataset_path}")
    print()
    print("请确认:")
    print("  1. 数据集已下载")
    print("  2. 路径拼写正确")
    print("  3. 有访问权限")
    sys.exit(1)

print(f"✓ 路径存在: {dataset_path}")

# 检查域
domains = ['amazon', 'webcam', 'dslr', 'caltech']
missing_domains = []
for domain in domains:
    domain_path = os.path.join(dataset_path, domain)
    if os.path.exists(domain_path):
        # 检查类别数量
        classes = [d for d in os.listdir(domain_path)
                  if os.path.isdir(os.path.join(domain_path, d))]
        print(f"✓ {domain}: {len(classes)} 个类别")
    else:
        print(f"✗ {domain}: 目录不存在")
        missing_domains.append(domain)

if missing_domains:
    print()
    print(f"✗ 缺少域: {', '.join(missing_domains)}")
    print("数据集可能不完整")
    response = input("是否继续使用此路径? (y/n): ").strip().lower()
    if response != 'y':
        sys.exit(1)

# 3. 计算相对路径或使用绝对路径
print()
print("=" * 80)
print("配置路径")
print("=" * 80)

project_root = os.path.dirname(os.path.abspath(__file__))
print(f"项目根目录: {project_root}")
print(f"数据集路径: {dataset_path}")
print()

# 尝试计算相对路径
try:
    rel_path = os.path.relpath(dataset_path, project_root)
    # 检查相对路径是否过于复杂
    if '..' in rel_path and rel_path.count('..') > 2:
        use_relative = False
        config_path = dataset_path
    else:
        use_relative = True
        config_path = rel_path
except ValueError:
    # 在不同的驱动器上（Windows）
    use_relative = False
    config_path = dataset_path

if use_relative:
    print(f"建议使用相对路径: {config_path}")
else:
    print(f"建议使用绝对路径: {config_path}")

print()
print("路径选项:")
print(f"  1. 相对路径: {rel_path if use_relative else '(不适用)'}")
print(f"  2. 绝对路径: {dataset_path}")

choice = input("选择使用哪种路径? (1/2, 默认推荐): ").strip()
if choice == '2':
    config_path = dataset_path
elif choice == '1' and use_relative:
    config_path = rel_path
# else 使用推荐的路径

# 4. 更新配置文件
print()
print("=" * 80)
print("更新配置文件")
print("=" * 80)

config_file = os.path.join(project_root, "scripts", "example_configs",
                          "cross_domain_adaptive_office_caltech.yaml")

if not os.path.exists(config_file):
    print(f"✗ 配置文件不存在: {config_file}")
    sys.exit(1)

print(f"配置文件: {config_file}")
print(f"将设置数据集路径为: {config_path}")
print()

# 读取配置文件
with open(config_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 查找并替换 data.root 行
updated = False
for i, line in enumerate(lines):
    if line.strip().startswith('root:'):
        old_path = line.strip()
        # 保持缩进
        indent = len(line) - len(line.lstrip())
        lines[i] = ' ' * indent + f"root: '{config_path}'\n"
        updated = True
        print(f"原路径: {old_path}")
        print(f"新路径: root: '{config_path}'")
        break

if not updated:
    print("✗ 未找到 data.root 配置项")
    sys.exit(1)

# 写回配置文件
response = input("\n确认更新配置文件? (y/n): ").strip().lower()
if response == 'y':
    with open(config_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print("✓ 配置文件已更新")
else:
    print("已取消更新")
    sys.exit(0)

# 5. 验证配置
print()
print("=" * 80)
print("验证配置")
print("=" * 80)

import yaml
with open(config_file, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

configured_path = config['data']['root']
print(f"配置中的路径: {configured_path}")

# 从项目根目录测试路径
if os.path.isabs(configured_path):
    test_path = configured_path
else:
    test_path = os.path.join(project_root, configured_path)

if os.path.exists(test_path):
    print(f"✓ 路径验证成功: {test_path}")
else:
    print(f"✗ 路径验证失败: {test_path}")
    print()
    print("请检查:")
    print(f"  - 从项目根目录 {project_root}")
    print(f"  - 访问路径 {configured_path}")
    print(f"  - 是否能找到数据集")
    sys.exit(1)

# 6. 完成
print()
print("=" * 80)
print("配置完成!")
print("=" * 80)
print()
print("现在可以运行训练:")
print("  python start_training.py")
print()
print("或先检查环境:")
print("  python setup_and_check.py")
