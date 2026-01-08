#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速诊断GGEUR准确率问题

使用方法:
    python quick_diagnose.py
"""

import os
import yaml
import sys

print("=" * 80)
print("GGEUR_Clip 准确率问题快速诊断")
print("=" * 80)

# 读取配置文件
config_file = 'scripts/example_configs/ggeur_officehome_lds.yaml'
print(f"\n[INFO] 正在分析配置文件: {config_file}")
print("-" * 80)

with open(config_file, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 检查关键配置
print("\n[CHECK] 关键配置检查:")
print("-" * 80)

# 1. 数据划分
splits = config['data']['splits']
print(f"[OK] 数据划分 (splits): {splits}")
print(f"  - 训练: {splits[0]*100:.0f}%")
print(f"  - 验证: {splits[1]*100:.0f}%")
print(f"  - 测试: {splits[2]*100:.0f}%")

# 2. 训练参数
local_steps = config['train']['local_update_steps']
print(f"\n{'[WARN]' if local_steps < 5 else '[OK]'} 本地训练步数 (local_update_steps): {local_steps}")
if local_steps < 5:
    print(f"  警告: 训练步数太少！建议改为 10-20")

lr = config['train']['optimizer']['lr']
print(f"[OK] 学习率 (lr): {lr}")

total_rounds = config['federate']['total_round_num']
print(f"[OK] 总轮数 (total_round_num): {total_rounds}")

# 3. GGEUR参数
print(f"\n[INFO] GGEUR增强参数:")
print("-" * 80)
ggeur = config['ggeur']
print(f"  scenario: {ggeur['scenario']}")
print(f"  n_samples_per_original: {ggeur['n_samples_per_original']}")
print(f"  m_samples_per_prototype: {ggeur['m_samples_per_prototype']}")
print(f"  step1_samples_per_class: {ggeur['step1_samples_per_class']}")
print(f"  use_full_covariance: {ggeur['use_full_covariance']}")
print(f"  eigenvalue_scaling: {ggeur['eigenvalue_scaling']}")

# 4. 数据集检查
print(f"\n[CHECK] 数据集检查:")
print("-" * 80)
data_root = config['data']['root']
if os.path.exists(data_root):
    print(f"[OK] 数据集路径存在: {data_root}")

    domains = ['Art', 'Clipart', 'Product', 'Real World']
    for domain in domains:
        domain_path = os.path.join(data_root, domain)
        if os.path.exists(domain_path):
            # 计算图片总数
            total_images = 0
            classes = [d for d in os.listdir(domain_path)
                      if os.path.isdir(os.path.join(domain_path, d))]

            for class_name in classes:
                class_path = os.path.join(domain_path, class_name)
                images = [f for f in os.listdir(class_path)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                total_images += len(images)

            print(f"  [OK] {domain}: {total_images} 张图片, {len(classes)} 个类别")
        else:
            print(f"  [FAIL] {domain}: 路径不存在")
else:
    print(f"[FAIL] 数据集路径不存在: {data_root}")

# 5. CLIP模型检查
print(f"\n[CHECK] CLIP模型检查:")
print("-" * 80)
clip_model_file = 'open_clip_vitb16.bin'
if os.path.exists(clip_model_file):
    file_size = os.path.getsize(clip_model_file) / (1024 * 1024)  # MB
    print(f"[OK] CLIP模型文件存在: {clip_model_file} ({file_size:.1f} MB)")
else:
    print(f"[WARN] CLIP模型文件不存在: {clip_model_file}")
    print(f"  程序会尝试自动下载，但可能需要网络连接")

# 6. 潜在问题总结
print(f"\n[WARN] 可能导致准确率低的问题:")
print("-" * 80)

issues = []

if local_steps < 5:
    issues.append(f"1. 本地训练步数太少 ({local_steps})，建议改为 10-20")

if ggeur['n_samples_per_original'] < 10:
    issues.append(f"2. 数据增强too少 (n_samples_per_original={ggeur['n_samples_per_original']})")

if not os.path.exists(clip_model_file):
    issues.append(f"3. CLIP模型文件不存在，可能影响特征提取")

if issues:
    for issue in issues:
        print(f"  {issue}")
else:
    print("  未发现明显配置问题")

# 7. 修复建议
print(f"\n[SUGGESTION] 修复建议:")
print("-" * 80)
print("""
1. 增加本地训练步数:
   train:
     local_update_steps: 10  # 从1改为10

2. 如果准确率仍然很低，尝试调整学习率:
   train:
     optimizer:
       lr: 0.0001  # 降低学习率

3. 运行测试命令:
   python run_ggeur.py --cfg scripts/example_configs/ggeur_officehome_lds.yaml

4. 查看详细日志:
   日志会保存在 exp/ggeur_officehome_lds/ 目录下
""")

print("\n" + "=" * 80)
print("诊断完成！")
print("=" * 80)
