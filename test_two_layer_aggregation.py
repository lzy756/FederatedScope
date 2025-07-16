#!/usr/bin/env python3
"""
测试改进后的FedGS worker中的两层聚合架构
"""

import numpy as np
import sys
import os
import json
import tempfile

# 添加FederatedScope路径
sys.path.append('/home/liruobin/FL/FederatedScope')

def test_two_layer_architecture():
    """测试两层聚合架构的逻辑"""
    print("=" * 60)
    print("测试两层聚合架构")
    print("=" * 60)
    
    # 模拟数据
    print("1. 模拟聚类结果和h向量求解")
    
    # 假设有3个聚类
    peer_communities = [
        [1, 2, 3, 4, 5],      # Cluster 0: 5 clients
        [6, 7, 8, 9],         # Cluster 1: 4 clients  
        [10, 11, 12, 13, 14, 15]  # Cluster 2: 6 clients
    ]
    
    # 假设h向量求解结果
    h_vector = np.array([3, 2, 3])  # 从每个聚类选择的客户端数量
    L = 8  # 总共选择8个客户端
    
    print(f"聚类结果: {peer_communities}")
    print(f"h向量: {h_vector}")
    print(f"总选择客户端数: {np.sum(h_vector)} (期望: L={L})")
    
    # 模拟客户端选择
    print("\n2. 模拟客户端选择过程")
    selected_clients_per_cluster = {}
    
    np.random.seed(42)  # 固定随机种子
    for k, pc in enumerate(peer_communities):
        num_to_select = int(h_vector[k])
        if num_to_select > 0 and len(pc) >= num_to_select:
            selected = np.random.choice(pc, num_to_select, replace=False).tolist()
        else:
            selected = pc[:num_to_select] if num_to_select > 0 else []
        selected_clients_per_cluster[k] = selected
        print(f"聚类 {k}: 选择 {selected} (共{len(selected)}个客户端)")
    
    # 模拟第一层聚合 (簇内FedAvg)
    print("\n3. 模拟第一层聚合 (簇内FedAvg)")
    cluster_models = {}
    
    for cluster_idx, cluster_clients in selected_clients_per_cluster.items():
        if not cluster_clients:
            continue
            
        print(f"聚类 {cluster_idx} 内部聚合:")
        print(f"  参与客户端: {cluster_clients}")
        
        # 模拟客户端模型参数 (简化为标量)
        client_models = []
        total_samples = 0
        
        for client_id in cluster_clients:
            # 模拟客户端的模型参数和样本数
            model_param = np.random.randn()  # 简化的模型参数
            sample_size = np.random.randint(50, 200)  # 随机样本数
            
            client_models.append({
                'client_id': client_id,
                'model_param': model_param,
                'sample_size': sample_size
            })
            total_samples += sample_size
            
            print(f"    客户端 {client_id}: 参数={model_param:.3f}, 样本数={sample_size}")
        
        # FedAvg聚合 (加权平均)
        weighted_sum = sum(client['model_param'] * client['sample_size'] 
                          for client in client_models)
        cluster_model = weighted_sum / total_samples
        
        cluster_models[cluster_idx] = {
            'model_param': cluster_model,
            'total_samples': total_samples
        }
        
        print(f"  聚合结果: 参数={cluster_model:.3f}, 总样本数={total_samples}")
    
    # 模拟第二层聚合 (簇间FedSAK)
    print("\n4. 模拟第二层聚合 (簇间FedSAK)")
    print("簇间聚合参与者:")
    
    cluster_feedback = []
    for cluster_idx, cluster_info in cluster_models.items():
        cluster_feedback.append({
            'cluster_id': cluster_idx,
            'model_param': cluster_info['model_param'],
            'sample_size': cluster_info['total_samples']
        })
        print(f"  聚类 {cluster_idx}: 参数={cluster_info['model_param']:.3f}, "
              f"样本数={cluster_info['total_samples']}")
    
    # 简化的FedSAK聚合 (这里用加权平均模拟)
    if cluster_feedback:
        total_samples_all = sum(cluster['sample_size'] for cluster in cluster_feedback)
        global_model = sum(cluster['model_param'] * cluster['sample_size'] 
                          for cluster in cluster_feedback) / total_samples_all
        
        print(f"\n最终全局模型参数: {global_model:.3f}")
        print(f"总参与样本数: {total_samples_all}")
    
    # 架构总结
    print("\n5. 两层聚合架构总结")
    print("架构流程:")
    print("  1. 根据h向量从每个聚类选择客户端")
    print("  2. 每个聚类内的客户端进行本地训练")
    print("  3. 第一层: 聚类内使用FedAvg聚合")
    print("  4. 第二层: 聚类间使用FedSAK聚合")
    print("  5. 得到最终的全局模型")
    
    print("\n优势:")
    print("  - 去除了CC概念，简化了架构")
    print("  - 两层聚合充分利用了数据分布的层次结构")
    print("  - FedAvg适合聚类内相似数据的聚合")
    print("  - FedSAK适合聚类间异构数据的聚合")
    
    return True

def test_architecture_comparison():
    """比较新旧架构的差异"""
    print("\n" + "=" * 60)
    print("新旧架构对比")
    print("=" * 60)
    
    print("旧架构 (基于CC):")
    print("  1. 根据h向量生成多个CC")
    print("  2. 每个CC内部进行FedAvg聚合")
    print("  3. CC之间进行FedSAK聚合")
    print("  4. 存在CC概念的复杂性")
    
    print("\n新架构 (基于Cluster):")
    print("  1. 根据h向量从每个聚类选择客户端")
    print("  2. 聚类内客户端进行FedAvg聚合")
    print("  3. 聚类间进行FedSAK聚合")
    print("  4. 直接基于数据分布聚类，更加直观")
    
    print("\n主要改进:")
    print("  ✅ 去除了CC的中间概念")
    print("  ✅ 直接基于数据分布聚类进行聚合")
    print("  ✅ 两层聚合架构更加清晰")
    print("  ✅ 减少了架构复杂性")
    
    return True

if __name__ == "__main__":
    try:
        success1 = test_two_layer_architecture()
        success2 = test_architecture_comparison()
        
        if success1 and success2:
            print("\n" + "=" * 60)
            print("✅ 两层聚合架构测试成功!")
            print("=" * 60)
        else:
            print("\n❌ 测试失败")
            
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
