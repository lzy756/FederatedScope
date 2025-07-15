#!/usr/bin/env python3
"""
直接测试改进后的FedGS_Splitter中的人数均衡集成效果
"""

import numpy as np
import sys
import os

# 添加FederatedScope路径
sys.path.append('/home/liruobin/FL/FederatedScope')

from federatedscope.contrib.splitter.fedgs_splitter import FedGS_Splitter

def create_simple_test_data():
    """创建简单的测试数据，确保能看到均衡效果"""
    # 创建一个简单的数据集，有明显的聚类结构
    dataset = []
    
    # 类别0-2：前600个样本
    for i in range(600):
        if i < 200:
            label = 0
        elif i < 400:
            label = 1
        else:
            label = 2
        dataset.append((np.random.randn(5), label))
    
    # 类别3-5：中间600个样本
    for i in range(600):
        if i < 200:
            label = 3
        elif i < 400:
            label = 4
        else:
            label = 5
        dataset.append((np.random.randn(5), label))
    
    # 类别6-9：最后800个样本
    for i in range(800):
        if i < 200:
            label = 6
        elif i < 400:
            label = 7
        elif i < 600:
            label = 8
        else:
            label = 9
        dataset.append((np.random.randn(5), label))
    
    return dataset

def test_balance_weights():
    """测试不同均衡权重的效果"""
    print("=" * 60)
    print("测试不同均衡权重对聚类结果的影响")
    print("=" * 60)
    
    dataset = create_simple_test_data()
    num_clients = 20
    
    # 测试不同的均衡权重
    weights_to_test = [0.1, 1.0, 5.0, 10.0]
    
    results = {}
    
    for weight in weights_to_test:
        print(f"\n--- 测试 balance_weight = {weight} ---")
        
        # 使用不同的随机种子确保结果不同
        np.random.seed(int(weight * 100))
        
        splitter = FedGS_Splitter(
            client_num=num_clients,
            alpha=0.5,
            min_groups=3,
            max_groups=7,
            balance_weight=weight,
            balance_tolerance=0.2
        )
        
        try:
            result = splitter(dataset)
            
            # 分析结果
            group_sizes = [len(group) for group in result]
            mean_size = np.mean(group_sizes)
            std_size = np.std(group_sizes)
            balance_score = std_size / mean_size
            
            results[weight] = {
                'group_sizes': group_sizes,
                'balance_score': balance_score,
                'num_groups': len(result)
            }
            
            print(f"社区数量: {len(result)}")
            print(f"组大小: {group_sizes}")
            print(f"平均大小: {mean_size:.1f}")
            print(f"标准差: {std_size:.1f}")
            print(f"均衡性评分: {balance_score:.3f}")
            
            if balance_score < 0.2:
                print("✅ 均衡性优秀")
            elif balance_score < 0.4:
                print("⚠️  均衡性良好")
            else:
                print("❌ 均衡性较差")
                
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            results[weight] = None
    
    # 比较结果
    print(f"\n--- 均衡权重效果比较 ---")
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_results) >= 2:
        sorted_by_balance = sorted(valid_results.items(), key=lambda x: x[1]['balance_score'])
        
        print("按均衡性排序（从好到差）:")
        for weight, result in sorted_by_balance:
            print(f"  权重 {weight}: 均衡评分 {result['balance_score']:.3f}, "
                  f"组数 {result['num_groups']}, 大小范围 [{min(result['group_sizes'])}, {max(result['group_sizes'])}]")
        
        best_weight = sorted_by_balance[0][0]
        worst_weight = sorted_by_balance[-1][0]
        improvement = (results[worst_weight]['balance_score'] - results[best_weight]['balance_score']) / results[worst_weight]['balance_score'] * 100
        
        print(f"\n最佳权重 {best_weight} 比最差权重 {worst_weight} 的均衡性提升了 {improvement:.1f}%")

def test_direct_clustering():
    """直接测试聚类算法的均衡效果"""
    print("\n" + "=" * 60)
    print("直接测试聚类算法的均衡集成效果")
    print("=" * 60)
    
    # 创建一个FedGS_Splitter实例来测试build_peer_communities方法
    splitter = FedGS_Splitter(
        client_num=15,
        alpha=0.5,
        min_groups=3,
        max_groups=5,
        balance_weight=3.0,
        balance_tolerance=0.15
    )
    
    # 创建模拟的客户端分布数据
    client_distributions = []
    
    # 创建3个明显的聚类
    # 聚类1：主要包含类别0,1
    for i in range(5):
        dist = np.array([40, 30, 5, 5, 5, 5, 5, 5])
        client_distributions.append(dist)
    
    # 聚类2：主要包含类别2,3,4
    for i in range(5):
        dist = np.array([5, 5, 25, 25, 25, 5, 5, 5])
        client_distributions.append(dist)
    
    # 聚类3：主要包含类别5,6,7
    for i in range(5):
        dist = np.array([5, 5, 5, 5, 5, 30, 25, 20])
        client_distributions.append(dist)
    
    # 设置data_info
    splitter.data_info = [{'distribution': dist} for dist in client_distributions]
    
    print("客户端数量:", len(client_distributions))
    print("期望的聚类结构: 3个聚类，每个5个客户端")
    
    try:
        # 调用build_peer_communities方法
        splitter.build_peer_communities(client_distributions)
        
        # 分析结果
        group_sizes = [len(pc) for pc in splitter.peer_communities]
        mean_size = np.mean(group_sizes)
        std_size = np.std(group_sizes)
        balance_score = std_size / mean_size
        
        print(f"\n实际结果:")
        print(f"社区数量: {len(splitter.peer_communities)}")
        print(f"组大小: {group_sizes}")
        print(f"平均大小: {mean_size:.1f}")
        print(f"标准差: {std_size:.1f}")
        print(f"均衡性评分: {balance_score:.3f}")
        
        # 检查每个社区的组成
        for i, community in enumerate(splitter.peer_communities):
            print(f"社区 {i+1}: 客户端 {[c+1 for c in community]}")
        
        if balance_score < 0.2:
            print("✅ 均衡集成效果优秀")
        elif balance_score < 0.4:
            print("⚠️  均衡集成效果良好")
        else:
            print("❌ 均衡集成效果需要改进")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_balance_weights()
    test_direct_clustering()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
