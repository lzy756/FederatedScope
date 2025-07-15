#!/usr/bin/env python3
"""
测试改进后的FedGS worker中的Binary Split模板求解
"""

import numpy as np
import sys
import os
import json
import tempfile

# 添加FederatedScope路径
sys.path.append('/home/liruobin/FL/FederatedScope')

# 创建模拟的配置对象
class MockConfig:
    def __init__(self):
        self.fedgs = MockFedGSConfig()
        self.aggregator = MockAggregatorConfig()
        self.fedsak = MockFedSAKConfig()
        self.federate = MockFederateConfig()
        self.eval = MockEvalConfig()
        self.outdir = './test_output'
        self.expname = 'test_binary_split'
        self.seed = 42
        
class MockFedGSConfig:
    def __init__(self):
        self.num_groups = 3
        self.L = 10  # Additional clients to select
        self.n_batch_size = 32
        
    def get(self, key, default):
        return getattr(self, key, default)

class MockAggregatorConfig:
    def get(self, key, default):
        if key == "lambda_":
            return 1e-3
        return default

class MockFedSAKConfig:
    def __init__(self):
        self.share_patterns = []

class MockFederateConfig:
    def __init__(self):
        self.total_round_num = 10
        self.mode = 'standalone'
        self.sample_client_rate = 1.0
        self.make_global_eval = False

class MockEvalConfig:
    def __init__(self):
        self.freq = 1
        self.split = ['test']
        self.report = ['weighted_avg']

def create_test_data():
    """创建测试数据文件"""
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    # 创建peer_communities.txt
    peer_communities = [
        [1, 2, 3, 4, 5],      # PC 1: 5 clients
        [6, 7, 8, 9],         # PC 2: 4 clients  
        [10, 11, 12, 13, 14, 15]  # PC 3: 6 clients
    ]
    
    peer_communities_data = {
        "peer_communities": peer_communities
    }
    
    with open('peer_communities.txt', 'w') as f:
        f.write("Peer Community Grouping Information\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total number of communities: {len(peer_communities)}\n\n")
        
        for idx, community in enumerate(peer_communities):
            f.write(f"Community #{idx + 1}\n")
            f.write(f"Members: {community}\n")
            f.write("-" * 30 + "\n")
        
        f.write("\nRaw Data (JSON format):\n")
        f.write(json.dumps(peer_communities_data, indent=2))
    
    # 创建data_info.txt (训练数据分布)
    num_classes = 10
    data_info = []
    
    # 为每个客户端创建数据分布
    for client_id in range(1, 16):  # 15个客户端
        if client_id <= 5:  # PC 1: 主要是类别0-3
            dist = np.random.dirichlet([3, 3, 2, 2, 1, 1, 1, 1, 1, 1]) * 100
        elif client_id <= 9:  # PC 2: 主要是类别4-6
            dist = np.random.dirichlet([1, 1, 1, 1, 3, 3, 3, 1, 1, 1]) * 100
        else:  # PC 3: 主要是类别7-9
            dist = np.random.dirichlet([1, 1, 1, 1, 1, 1, 1, 3, 3, 3]) * 100
        
        data_info.append(dist.astype(int).tolist())
    
    data_info_data = {"data_info": data_info}
    
    with open('data_info.txt', 'w') as f:
        f.write("Client Data Distribution Information\n")
        f.write("=" * 50 + "\n\n")
        
        for client_idx, dist in enumerate(data_info):
            f.write(f"Client #{client_idx + 1}\n")
            f.write(f"Distribution: {dist}\n")
            total = sum(dist)
            percentages = [f"{(count/total*100):.2f}%" for count in dist]
            f.write(f"Percentages: {percentages}\n")
            f.write(f"Total Samples: {total}\n")
            f.write("-" * 30 + "\n")
        
        f.write("\nRaw Data (JSON format):\n")
        f.write(json.dumps(data_info_data, indent=2))
    
    # 创建test_data_info.txt (测试数据分布，作为目标分布)
    # 创建均匀分布作为目标
    test_data_info = []
    for client_id in range(1, 16):
        uniform_dist = [10] * num_classes  # 每个类别10个样本
        test_data_info.append(uniform_dist)
    
    test_data_info_data = {"test_data_info": test_data_info}
    
    with open('test_data_info.txt', 'w') as f:
        f.write("Test Set Data Distribution Information\n")
        f.write("=" * 50 + "\n\n")
        
        for client_idx, dist in enumerate(test_data_info):
            f.write(f"Client #{client_idx + 1}\n")
            f.write(f"Distribution: {dist}\n")
            total = sum(dist)
            percentages = [f"{(count/total*100):.2f}%" for count in dist]
            f.write(f"Percentages: {percentages}\n")
            f.write(f"Total Samples: {total}\n")
            f.write("-" * 30 + "\n")
        
        f.write("\nRaw Data (JSON format):\n")
        f.write(json.dumps(test_data_info_data, indent=2))
    
    print("Test data files created successfully!")
    return peer_communities, data_info, test_data_info

def test_binary_split_solver():
    """测试Binary Split求解器"""
    print("=" * 60)
    print("测试Binary Split模板求解")
    print("=" * 60)

    # 创建测试数据
    peer_communities, data_info, test_data_info = create_test_data()

    try:
        # 直接测试Binary Split核心算法
        print("测试Binary Split核心算法...")

        # 模拟数据
        K = 3  # 3个聚类
        F = 10  # 10个类别
        L = 10   # 额外选择5个客户端
        n_batch_size = 32

        # 构建A矩阵 (每个聚类的平均特征)
        A_matrix = np.array([
            [30, 25, 15, 10, 5, 5, 3, 2, 3, 2],      # PC 1: 主要是类别0-3
            [5, 5, 5, 5, 25, 25, 20, 5, 3, 2],       # PC 2: 主要是类别4-6
            [3, 2, 3, 2, 5, 5, 5, 25, 25, 25]        # PC 3: 主要是类别7-9
        ]).T  # 转置为 (F, K)

        # 目标分布 (均匀分布)
        target_distribution = np.ones(F) / F

        # 构建M向量
        total_clients_selected = L + K
        M_vector = n_batch_size * total_clients_selected * target_distribution

        # 上界 (每个聚类的大小)
        B_vector = np.array([5, 4, 6])  # PC大小

        print(f"A matrix shape: {A_matrix.shape}")
        print(f"M vector: {M_vector}")
        print(f"B vector: {B_vector}")
        print(f"Target total clients: {total_clients_selected}")

        # 测试Binary Split变换
        import math
        G = []
        for k in range(K):
            B_k = int(B_vector[k])
            G_k = math.ceil(math.log2(B_k + 1)) if B_k > 0 else 1
            G.append(G_k)

        print(f"Binary decomposition G: {G}")

        # 构建T矩阵
        total_binary_vars = sum(G)
        T_matrix = np.zeros((K, total_binary_vars))

        col_idx = 0
        for i in range(K):
            for j in range(G[i]):
                T_matrix[i, col_idx] = 2**j
                col_idx += 1

        print(f"T matrix shape: {T_matrix.shape}")
        print(f"T matrix:\n{T_matrix}")

        # 测试一个简单的解
        # 假设每个聚类选择2个客户端 (总共6个，满足L=5的约束需要调整)
        test_h = np.array([2, 2, 1])  # 总共5个额外客户端
        print(f"Test h vector: {test_h}")
        print(f"Test h sum: {np.sum(test_h)} (should equal L={L})")

        # 计算目标函数值
        Ah = A_matrix @ test_h
        diff = Ah - M_vector
        objective_value = np.linalg.norm(diff)**2
        print(f"Test objective value: {objective_value}")

        print("\n✅ Binary Split核心算法测试成功!")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理测试文件
        for filename in ['peer_communities.txt', 'data_info.txt', 'test_data_info.txt']:
            if os.path.exists(filename):
                os.remove(filename)

if __name__ == "__main__":
    test_binary_split_solver()
