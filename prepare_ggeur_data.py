#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
准备GGEUR所需的所有离线数据

这个脚本会：
1. 提取训练集CLIP特征
2. 提取测试集CLIP特征
3. 计算协方差矩阵
4. 生成增强特征

完成后可以直接运行 run_ggeur_standalone.py
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# 添加FederatedScope到path
sys.path.insert(0, os.path.abspath('.'))

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.contrib.utils.clip_extractor import CLIPExtractor
from federatedscope.contrib.data_augmentation.ggeur_augmentation import GGEURAugmentation

def extract_train_features(cfg, data):
    """
    提取训练集CLIP特征

    保存到: clip_office_home_train_features/{domain}/client_{i}_class_{j}_original_features.npy
    """
    print("="*80)
    print("Step 1: 提取训练集CLIP特征")
    print("="*80)

    device = torch.device(f"cuda:{cfg.device}" if cfg.use_gpu else "cpu")

    clip_extractor = CLIPExtractor(
        model_name='ViT-B/16',
        device=device,
        batch_size=cfg.dataloader.batch_size
    )

    output_dir = './clip_office_home_train_features'
    os.makedirs(output_dir, exist_ok=True)

    domain_names = ['Art', 'Clipart', 'Product', 'Real_World']

    for client_id, client_data in data.items():
        if client_id == 0:  # server
            continue

        domain_name = domain_names[client_id - 1]
        print(f"\n处理客户端 {client_id} ({domain_name})...")

        # 获取训练数据
        train_dataset = client_data['train'].dataset

        # 提取所有特征
        result = clip_extractor.extract_dataset_features(
            train_dataset,
            use_cache=False
        )

        embeddings = result['embeddings'].cpu().numpy()
        labels = result['labels'].cpu().numpy()

        # 按类别保存
        for class_id in range(65):
            class_mask = (labels == class_id)
            class_embeddings = embeddings[class_mask]
            class_labels = labels[class_mask]

            if len(class_embeddings) > 0:
                class_dir = os.path.join(output_dir, domain_name)
                os.makedirs(class_dir, exist_ok=True)

                features_path = os.path.join(class_dir, f'client_{client_id-1}_class_{class_id}_original_features.npy')
                labels_path = os.path.join(class_dir, f'client_{client_id-1}_class_{class_id}_labels.npy')

                np.save(features_path, class_embeddings)
                np.save(labels_path, class_labels)

                print(f"  类别 {class_id}: {len(class_embeddings)} 样本")

    print(f"\n训练集特征已保存到: {output_dir}")

def extract_test_features(cfg, data):
    """
    提取测试集CLIP特征

    保存到: clip_office_home_test_features/{domain}/{domain}_test_features.npy
    """
    print("\n" + "="*80)
    print("Step 2: 提取测试集CLIP特征")
    print("="*80)

    device = torch.device(f"cuda:{cfg.device}" if cfg.use_gpu else "cpu")

    clip_extractor = CLIPExtractor(
        model_name='ViT-B/16',
        device=device,
        batch_size=cfg.dataloader.batch_size
    )

    output_dir = './clip_office_home_test_features'
    os.makedirs(output_dir, exist_ok=True)

    domain_names = ['Art', 'Clipart', 'Product', 'Real_World']

    for client_id, client_data in data.items():
        if client_id == 0:  # server
            continue

        domain_name = domain_names[client_id - 1]
        print(f"\n处理 {domain_name} 测试集...")

        # 获取测试数据
        test_dataset = client_data['test'].dataset

        # 提取特征
        result = clip_extractor.extract_dataset_features(
            test_dataset,
            use_cache=False
        )

        embeddings = result['embeddings'].cpu().numpy()
        labels = result['labels'].cpu().numpy()

        # 保存
        domain_dir = os.path.join(output_dir, domain_name)
        os.makedirs(domain_dir, exist_ok=True)

        features_path = os.path.join(domain_dir, f'{domain_name}_test_features.npy')
        labels_path = os.path.join(domain_dir, f'{domain_name}_test_labels.npy')

        np.save(features_path, embeddings)
        np.save(labels_path, labels)

        print(f"  测试集大小: {len(embeddings)} 样本")
        print(f"  已保存到: {domain_dir}")

    print(f"\n测试集特征已保存到: {output_dir}")

def compute_global_covariance(cfg):
    """
    计算全局协方差矩阵

    保存到: cov_matrix_output/class_{0~64}_cov_matrix.npy
    """
    print("\n" + "="*80)
    print("Step 3: 计算全局协方差矩阵")
    print("="*80)

    train_features_dir = './clip_office_home_train_features'
    output_dir = './cov_matrix_output'
    os.makedirs(output_dir, exist_ok=True)

    domain_names = ['Art', 'Clipart', 'Product', 'Real_World']

    for class_id in tqdm(range(65), desc="计算协方差矩阵"):
        # 收集所有domain的该类别特征
        all_class_features = []

        for domain_name in domain_names:
            for client_id in [0, 1, 2, 3]:
                if domain_name == 'Art' and client_id != 0:
                    continue
                if domain_name == 'Clipart' and client_id != 1:
                    continue
                if domain_name == 'Product' and client_id != 2:
                    continue
                if domain_name == 'Real_World' and client_id != 3:
                    continue

                features_path = os.path.join(train_features_dir, domain_name,
                                            f'client_{client_id}_class_{class_id}_original_features.npy')

                if os.path.exists(features_path):
                    features = np.load(features_path)
                    if len(features) > 0:
                        all_class_features.append(features)

        if len(all_class_features) > 0:
            # 合并所有特征
            all_features = np.vstack(all_class_features)

            # 计算协方差矩阵
            mean = all_features.mean(axis=0)
            centered = all_features - mean
            cov_matrix = (centered.T @ centered) / (len(all_features) - 1)

            # 添加正则化
            cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6

            # 保存
            cov_path = os.path.join(output_dir, f'class_{class_id}_cov_matrix.npy')
            np.save(cov_path, cov_matrix)

    print(f"\n协方差矩阵已保存到: {output_dir}")

def generate_augmented_features(cfg):
    """
    生成增强特征

    保存到: argumented_clip_features/{domain}/client_{i}_class_{j}/final_embeddings_filled.npy
    """
    print("\n" + "="*80)
    print("Step 4: 生成增强特征")
    print("="*80)

    # 后续实现...
    print("增强特征生成功能待实现...")
    print("请参考原始代码的 prototype_cov_matrix_generate_features.py")

def main():
    print("\n" + "="*80)
    print("GGEUR离线数据准备脚本")
    print("="*80 + "\n")

    # 加载配置
    cfg = global_cfg.clone()
    cfg.merge_from_file('scripts/example_configs/ggeur_officehome_lds.yaml')

    # 构建数据
    print("加载数据集...")
    data, modified_cfg = get_data(cfg)
    print(f"数据集加载完成: {len(data)-1} 个客户端\n")

    # Step 1: 提取训练集特征
    extract_train_features(cfg, data)

    # Step 2: 提取测试集特征
    extract_test_features(cfg, data)

    # Step 3: 计算协方差矩阵
    compute_global_covariance(cfg)

    # Step 4: 生成增强特征
    print("\n" + "="*80)
    print("注意：增强特征生成需要使用原始代码的 prototype_cov_matrix_generate_features.py")
    print("请在原始代码目录运行该脚本，然后将生成的 argumented_clip_features/ 复制到此目录")
    print("="*80 + "\n")

    print("离线数据准备完成！")
    print("\n下一步：")
    print("1. 复制 argumented_clip_features/ 到当前目录")
    print("2. 运行: python run_ggeur_standalone.py")

if __name__ == "__main__":
    main()
