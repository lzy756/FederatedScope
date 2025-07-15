#!/usr/bin/env python3
"""
测试改进后的FedGS_Splitter，验证人数均衡集成到层次聚类的效果
"""

import numpy as np
import sys
import os

# 添加FederatedScope路径
sys.path.append('/home/liruobin/FL/FederatedScope')

from federatedscope.contrib.splitter.fedgs_splitter import FedGS_Splitter

def create_synthetic_dataset(num_samples=2000, num_classes=10):
    """创建合成数据集用于测试，模拟真实的联邦学习数据分割场景"""
    dataset = []

    # 创建一个统一的数据集，然后让splitter来分割
    # 这样更符合实际的联邦学习场景
    for i in range(num_samples):
        # 创建不均匀的类别分布，模拟真实数据
        if i < num_samples * 0.3:
            # 前30%的数据主要是类别0-3
            label = np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])
        elif i < num_samples * 0.6:
            # 中间30%的数据主要是类别4-6
            label = np.random.choice([4, 5, 6], p=[0.5, 0.3, 0.2])
        else:
            # 后40%的数据主要是类别7-9
            label = np.random.choice([7, 8, 9], p=[0.4, 0.4, 0.2])

        # 创建虚拟特征向量
        feature = np.random.randn(10)
        dataset.append((feature, label))

    return dataset

def test_balance_improvement():
    """测试人数均衡改进的效果"""
    print("=" * 60)
    print("测试改进后的FedGS_Splitter人数均衡效果")
    print("=" * 60)
    
    # 创建测试数据
    num_clients = 20
    dataset = create_synthetic_dataset(num_samples=2000)
    
    # 测试不同的均衡权重
    balance_weights = [0.5, 1.0, 2.0, 5.0]
    balance_tolerances = [0.1, 0.2, 0.3]
    
    for balance_weight in balance_weights:
        for balance_tolerance in balance_tolerances:
            print(f"\n--- 测试配置: balance_weight={balance_weight}, balance_tolerance={balance_tolerance} ---")
            
            # 创建splitter
            splitter = FedGS_Splitter(
                client_num=num_clients,
                alpha=0.5,
                min_groups=3,
                max_groups=6,
                balance_weight=balance_weight,
                balance_tolerance=balance_tolerance
            )
            
            try:
                # 执行分割
                result = splitter(dataset)
                
                # 分析结果
                group_sizes = [len(group) for group in result]
                mean_size = np.mean(group_sizes)
                std_size = np.std(group_sizes)
                min_size = min(group_sizes)
                max_size = max(group_sizes)
                
                print(f"社区数量: {len(result)}")
                print(f"组大小: {group_sizes}")
                print(f"大小统计 - 平均: {mean_size:.2f}, 标准差: {std_size:.2f}")
                print(f"大小范围: [{min_size}, {max_size}]")
                print(f"均衡性评分 (标准差/平均值): {std_size/mean_size:.3f}")
                
                # 检查是否有显著改进
                balance_ratio = std_size / mean_size
                if balance_ratio < 0.2:
                    print("✅ 大小均衡性良好")
                elif balance_ratio < 0.4:
                    print("⚠️  大小均衡性一般")
                else:
                    print("❌ 大小均衡性较差")
                    
            except Exception as e:
                print(f"❌ 测试失败: {e}")

def compare_with_original():
    """比较原始方法和改进方法的效果"""
    print("\n" + "=" * 60)
    print("比较原始方法和改进方法")
    print("=" * 60)
    
    num_clients = 24
    dataset = create_synthetic_dataset(num_samples=2400)
    
    # 测试改进方法
    print("\n--- 改进方法 (balance_weight=2.0) ---")
    improved_splitter = FedGS_Splitter(
        client_num=num_clients,
        alpha=0.5,
        min_groups=4,
        max_groups=6,
        balance_weight=2.0,
        balance_tolerance=0.2
    )
    
    try:
        improved_result = improved_splitter(dataset)
        improved_sizes = [len(group) for group in improved_result]
        improved_std = np.std(improved_sizes)
        improved_mean = np.mean(improved_sizes)
        
        print(f"组大小: {improved_sizes}")
        print(f"均衡性评分: {improved_std/improved_mean:.3f}")
        
    except Exception as e:
        print(f"改进方法测试失败: {e}")
    
    # 测试低权重方法 (模拟原始方法)
    print("\n--- 低权重方法 (balance_weight=0.1, 模拟原始方法) ---")
    original_like_splitter = FedGS_Splitter(
        client_num=num_clients,
        alpha=0.5,
        min_groups=4,
        max_groups=6,
        balance_weight=0.1,  # 很低的权重，类似原始方法
        balance_tolerance=0.5
    )
    
    try:
        original_result = original_like_splitter(dataset)
        original_sizes = [len(group) for group in original_result]
        original_std = np.std(original_sizes)
        original_mean = np.mean(original_sizes)
        
        print(f"组大小: {original_sizes}")
        print(f"均衡性评分: {original_std/original_mean:.3f}")
        
        # 比较结果
        print(f"\n--- 比较结果 ---")
        print(f"改进方法均衡性评分: {improved_std/improved_mean:.3f}")
        print(f"原始方法均衡性评分: {original_std/original_mean:.3f}")
        
        if improved_std/improved_mean < original_std/original_mean:
            improvement = ((original_std/original_mean) - (improved_std/improved_mean)) / (original_std/original_mean) * 100
            print(f"✅ 改进方法均衡性提升了 {improvement:.1f}%")
        else:
            print("⚠️  改进效果不明显，可能需要调整参数")
            
    except Exception as e:
        print(f"原始方法测试失败: {e}")

if __name__ == "__main__":
    # 设置随机种子以获得可重现的结果
    np.random.seed(42)
    
    # 运行测试
    test_balance_improvement()
    compare_with_original()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
