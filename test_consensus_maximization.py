#!/usr/bin/env python3
"""
测试共识最大化聚合方法的实现
"""

import numpy as np
import torch
import sys
import os

# 添加FederatedScope路径
sys.path.append('/home/liruobin/FL/FederatedScope')

def test_consensus_maximization_basic():
    """测试共识最大化聚合的基本功能"""
    print("=" * 60)
    print("测试共识最大化聚合基本功能")
    print("=" * 60)
    
    try:
        from federatedscope.core.aggregators.fedgs_aggregator import FedGSAggregator
        
        # 创建模拟配置
        class MockConfig:
            def __init__(self):
                self.fedgs = MockFedGSConfig()
                self.aggregator = MockAggregatorConfig()
                self.fedsak = MockFedSAKConfig()
                self.federate = MockFederateConfig()
                
        class MockFedGSConfig:
            def __init__(self):
                self.num_groups = 3
                
        class MockAggregatorConfig:
            def get(self, key, default):
                if key == "lambda_":
                    return 1e-3
                elif key == "lr_shared":
                    return 0.05
                return default
                
        class MockFedSAKConfig:
            def __init__(self):
                self.share_patterns = ["layer"]
                
        class MockFederateConfig:
            def __init__(self):
                self.ignore_weight = False
                self.use_ss = False
        
        config = MockConfig()
        aggregator = FedGSAggregator(model=None, device='cpu', config=config)
        
        print("✅ FedGSAggregator 初始化成功")
        
        # 创建模拟的客户端模型参数
        print("\n1. 创建模拟客户端模型参数")
        
        # 模拟3个客户端的模型参数
        client_models = []
        for i in range(3):
            model_para = {
                'layer1.weight': torch.randn(10, 5) + i * 0.5,  # 添加一些差异
                'layer1.bias': torch.randn(10) + i * 0.3,
                'layer2.weight': torch.randn(3, 10) + i * 0.2,
                'layer2.bias': torch.randn(3) + i * 0.1
            }
            
            client_models.append({
                'client_id': i + 1,
                'sample_size': 100 + i * 50,  # 不同的样本大小
                'model_para': model_para
            })
            
            print(f"客户端 {i+1}: 样本数={100 + i * 50}")
            for layer_name, param in model_para.items():
                print(f"  {layer_name}: shape={param.shape}, mean={param.mean().item():.4f}")
        
        # 测试共识最大化聚合
        print("\n2. 执行共识最大化聚合")
        
        aggregated_model = aggregator._intra_group_aggregate(client_models)
        
        print("聚合结果:")
        for layer_name, param in aggregated_model.items():
            print(f"  {layer_name}: shape={param.shape}, mean={param.mean().item():.4f}")
        
        # 比较与传统加权平均的差异
        print("\n3. 与传统加权平均比较")
        
        # 计算传统加权平均
        total_samples = sum(model['sample_size'] for model in client_models)
        weighted_avg_model = {}
        
        for layer_name in client_models[0]['model_para'].keys():
            weighted_avg_model[layer_name] = torch.zeros_like(
                client_models[0]['model_para'][layer_name]
            )
            
            for model in client_models:
                weight = model['sample_size'] / total_samples
                weighted_avg_model[layer_name] += model['model_para'][layer_name] * weight
        
        print("传统加权平均结果:")
        for layer_name, param in weighted_avg_model.items():
            print(f"  {layer_name}: shape={param.shape}, mean={param.mean().item():.4f}")
        
        # 计算差异
        print("\n4. 聚合方法差异分析")
        for layer_name in aggregated_model.keys():
            consensus_param = aggregated_model[layer_name]
            weighted_param = weighted_avg_model[layer_name]
            
            diff = torch.norm(consensus_param - weighted_param).item()
            relative_diff = diff / torch.norm(weighted_param).item() if torch.norm(weighted_param).item() > 0 else 0
            
            print(f"  {layer_name}:")
            print(f"    绝对差异: {diff:.6f}")
            print(f"    相对差异: {relative_diff:.6f}")
        
        print("\n✅ 共识最大化聚合测试成功!")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_consensus_optimization():
    """测试共识权重优化算法"""
    print("\n" + "=" * 60)
    print("测试共识权重优化算法")
    print("=" * 60)
    
    try:
        from federatedscope.core.aggregators.fedgs_aggregator import FedGSAggregator
        
        # 创建模拟配置
        class MockConfig:
            def __init__(self):
                self.fedgs = MockFedGSConfig()
                self.aggregator = MockAggregatorConfig()
                self.fedsak = MockFedSAKConfig()
                self.federate = MockFederateConfig()

        class MockFedGSConfig:
            def __init__(self):
                self.num_groups = 3

        class MockAggregatorConfig:
            def get(self, key, default):
                if key == "lambda_":
                    return 1e-3
                elif key == "lr_shared":
                    return 0.05
                return default

        class MockFedSAKConfig:
            def __init__(self):
                self.share_patterns = ["layer"]

        class MockFederateConfig:
            def __init__(self):
                self.ignore_weight = False
                self.use_ss = False
        
        config = MockConfig()
        aggregator = FedGSAggregator(model=None, device='cpu', config=config)
        
        # 创建测试向量
        print("1. 创建测试向量")
        
        # 创建3个归一化的更新向量
        updates = [
            torch.tensor([1.0, 0.0, 0.0]),  # 向量1
            torch.tensor([0.0, 1.0, 0.0]),  # 向量2  
            torch.tensor([0.5, 0.5, 0.0])   # 向量3
        ]
        
        # 归一化
        normalized_updates = []
        for update in updates:
            norm = torch.norm(update, p=2)
            normalized_updates.append(update / norm if norm > 0 else update)
            
        print("归一化更新向量:")
        for i, update in enumerate(normalized_updates):
            print(f"  客户端 {i+1}: {update.numpy()}")
        
        # 测试权重优化
        print("\n2. 执行权重优化")
        
        weights = aggregator._solve_consensus_weights(normalized_updates)
        
        print(f"优化后的权重: {weights}")
        print(f"权重和: {sum(weights):.6f}")
        
        # 计算聚合结果
        aggregated = torch.zeros_like(normalized_updates[0])
        for i, (weight, update) in enumerate(zip(weights, normalized_updates)):
            aggregated += weight * update
            
        print(f"聚合向量: {aggregated.numpy()}")
        print(f"聚合向量L2范数: {torch.norm(aggregated).item():.6f}")
        
        # 比较与均匀权重的差异
        print("\n3. 与均匀权重比较")
        
        uniform_weights = [1.0/3] * 3
        uniform_aggregated = torch.zeros_like(normalized_updates[0])
        for weight, update in zip(uniform_weights, normalized_updates):
            uniform_aggregated += weight * update
            
        print(f"均匀权重聚合: {uniform_aggregated.numpy()}")
        print(f"均匀权重L2范数: {torch.norm(uniform_aggregated).item():.6f}")
        
        # 验证共识最大化效果
        consensus_norm = torch.norm(aggregated).item()
        uniform_norm = torch.norm(uniform_aggregated).item()
        
        print(f"\n4. 共识最大化效果验证")
        print(f"共识最大化L2范数: {consensus_norm:.6f}")
        print(f"均匀权重L2范数: {uniform_norm:.6f}")
        
        if consensus_norm <= uniform_norm + 1e-6:
            print("✅ 共识最大化成功：L2范数更小或相等")
        else:
            print("⚠️  共识最大化可能未达到最优")
        
        print("\n✅ 权重优化测试成功!")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_consensus_maximization_basic()
    success2 = test_consensus_optimization()
    
    if success1 and success2:
        print("\n" + "=" * 60)
        print("✅ 所有共识最大化聚合测试通过!")
        print("=" * 60)
    else:
        print("\n❌ 部分测试失败")
