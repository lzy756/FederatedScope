"""
GGEUR_Clip 诊断脚本 - 请运行并提供输出结果

这个脚本会检查：
1. 数据增强是否正确执行
2. 训练数据的质量
3. 模型是否在学习
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import yaml
from federatedscope.core.configs import global_cfg
from federatedscope.core.cmd_args import parse_args

print("="*80)
print("GGEUR_Clip 诊断脚本")
print("="*80)

# Load config
config_path = "scripts/example_configs/ggeur_officehome_lds.yaml"
print(f"\n1. 加载配置文件: {config_path}")

with open(config_path, 'r') as f:
    cfg_dict = yaml.safe_load(f)

print(f"   - 数据集: {cfg_dict['data']['type']}")
print(f"   - 客户端数: {cfg_dict['federate']['client_num']}")
print(f"   - 通信轮数: {cfg_dict['federate']['total_round_num']}")
print(f"   - Local steps: {cfg_dict['train']['local_update_steps']}")
print(f"   - CLIP model: {cfg_dict['ggeur']['clip_model']}")
print(f"   - Use softmax: {cfg_dict['ggeur']['use_softmax_output']}")
print(f"   - Splitter: {cfg_dict['data']['splitter']}")

# Check if output directory exists
outdir = cfg_dict.get('outdir', 'exp/ggeur_officehome_lds')
print(f"\n2. 检查输出目录: {outdir}")

if os.path.exists(outdir):
    print(f"   ✓ 目录存在")

    # Check CLIP cache
    clip_cache = os.path.join(outdir, 'clip_cache')
    if os.path.exists(clip_cache):
        cache_files = os.listdir(clip_cache)
        print(f"   ✓ CLIP 缓存: {len(cache_files)} 个文件")
        for f in cache_files[:5]:
            fpath = os.path.join(clip_cache, f)
            size_mb = os.path.getsize(fpath) / 1024 / 1024
            print(f"     - {f} ({size_mb:.2f} MB)")
    else:
        print(f"   ✗ CLIP 缓存不存在")

    # Check for log files
    import glob
    log_files = glob.glob(os.path.join(outdir, '*.log'))
    if log_files:
        print(f"   ✓ 找到 {len(log_files)} 个日志文件")
        latest_log = max(log_files, key=os.path.getmtime)
        print(f"     最新日志: {os.path.basename(latest_log)}")

        # Parse log for key information
        with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # Look for augmentation info
        aug_lines = [l for l in lines if 'augment' in l.lower() or 'DATA DISTRIBUTION' in l]
        if aug_lines:
            print(f"\n3. 数据增强信息（从日志）:")
            for line in aug_lines[:10]:
                print(f"   {line.strip()}")

        # Look for training info
        train_lines = [l for l in lines if 'train_acc' in l.lower() or 'Training' in l]
        if train_lines:
            print(f"\n4. 训练信息（从日志）:")
            for line in train_lines[-5:]:
                print(f"   {line.strip()}")

        # Look for test accuracy
        test_lines = [l for l in lines if 'test_acc' in l.lower() or 'Accuracy:' in l]
        if test_lines:
            print(f"\n5. 测试准确率（从日志）:")
            for line in test_lines[-10:]:
                print(f"   {line.strip()}")
    else:
        print(f"   ✗ 没有找到日志文件")
else:
    print(f"   ✗ 输出目录不存在（可能还没运行过）")

print("\n" + "="*80)
print("诊断完成")
print("="*80)

print("\n请回答以下问题：")
print("1. 日志中是否看到 'CLIENT #X - DATA DISTRIBUTION REPORT'？")
print("2. 增强后每个客户端有多少样本？（应该是 65 类 × 50 = 3250 左右）")
print("3. 第1轮的 train_acc 是多少？（如果<5%说明有问题）")
print("4. test_acc 是否在提升？（应该从20%左右开始逐渐提升）")
print("\n如果上述检查都正常，但准确率还是很低，问题可能在：")
print("  → CLIP 模型版本不匹配（原始用的 open_clip？）")
print("  → 测试集构建有问题")
print("  → 优化器状态管理有问题")
