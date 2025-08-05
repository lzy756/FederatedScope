#!/usr/bin/env python3
"""
UT-HAR数据集频率域划分脚本
按照频率特征将数据集划分为4个域，并按照频率高低命名文件夹
"""

import os
import sys
import numpy as np
import logging
import json
from pathlib import Path
from scipy import signal
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_ut_har_data(data_root):
    """加载原始UT-HAR数据"""
    import glob
    
    data_dir = os.path.join(data_root, 'ut_har', 'data')
    label_dir = os.path.join(data_root, 'ut_har', 'label')
    
    data_list = glob.glob(os.path.join(data_dir, '*.csv'))
    label_list = glob.glob(os.path.join(label_dir, '*.csv'))
    
    ut_har_data = {}
    
    logger.info("加载UT-HAR数据文件...")
    
    # 加载数据文件
    for data_file in data_list:
        data_name = os.path.splitext(os.path.basename(data_file))[0]
        logger.info(f"  正在加载 {data_name}...")
        with open(data_file, 'rb') as f:
            data = np.load(f)
            data = data.reshape(len(data), 1, 250, 90)
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        ut_har_data[data_name] = data_norm
    
    # 加载标签文件
    for label_file in label_list:
        label_name = os.path.splitext(os.path.basename(label_file))[0]
        logger.info(f"  正在加载 {label_name}...")
        with open(label_file, 'rb') as f:
            label = np.load(f)
        ut_har_data[label_name] = label
    
    logger.info("数据加载完成！")
    for key, value in ut_har_data.items():
        logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    
    return ut_har_data

def extract_frequency_features(csi_data, sample_rate=1000):
    """
    提取CSI数据的频率特征
    
    Args:
        csi_data: CSI数据 (250, 90)
        sample_rate: 采样率
    
    Returns:
        features: 8维频率特征向量
    """
    all_subcarrier_features = []
    
    for subcarrier in range(90):
        time_series = csi_data[:, subcarrier]  # (250,)
        
        # 计算功率谱密度
        freqs, psd = signal.welch(time_series, sample_rate, nperseg=min(64, len(time_series)//4))
        
        # 特征1: 谱质心
        spectral_centroid = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0
        
        # 特征2: 谱带宽
        if np.sum(psd) > 0:
            spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * psd) / np.sum(psd))
        else:
            spectral_bandwidth = 0
        
        # 特征3: 主频率
        dominant_freq = freqs[np.argmax(psd)] if len(psd) > 0 else 0
        
        # 特征4: 谱滚降 (85%能量所在频率)
        cumulative_psd = np.cumsum(psd)
        if cumulative_psd[-1] > 0:
            rolloff_idx = np.where(cumulative_psd >= 0.85 * cumulative_psd[-1])[0]
            spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
        else:
            spectral_rolloff = 0
        
        # 特征5-7: 频率范围内的能量比例
        total_energy = np.sum(psd)
        if total_energy > 0:
            # 低频 (0-10 Hz)
            low_freq_mask = (freqs >= 0) & (freqs <= 10)
            low_freq_ratio = np.sum(psd[low_freq_mask]) / total_energy
            
            # 中频 (10-50 Hz)  
            mid_freq_mask = (freqs > 10) & (freqs <= 50)
            mid_freq_ratio = np.sum(psd[mid_freq_mask]) / total_energy
            
            # 高频 (50+ Hz)
            high_freq_mask = freqs > 50
            high_freq_ratio = np.sum(psd[high_freq_mask]) / total_energy
        else:
            low_freq_ratio = mid_freq_ratio = high_freq_ratio = 0
        
        # 特征8: 谱熵
        psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
        psd_norm = psd_norm[psd_norm > 0]  # 避免log(0)
        if len(psd_norm) > 0:
            spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm))
        else:
            spectral_entropy = 0
        
        subcarrier_features = [
            spectral_centroid, spectral_bandwidth, dominant_freq, spectral_rolloff,
            low_freq_ratio, mid_freq_ratio, high_freq_ratio, spectral_entropy
        ]
        all_subcarrier_features.append(subcarrier_features)
    
    # 对所有子载波的特征取平均
    if len(all_subcarrier_features) > 0:
        features = np.mean(all_subcarrier_features, axis=0)
    else:
        features = np.zeros(8)
    
    return features

def calculate_frequency_score(features):
    """
    计算频率特征的综合得分，用于排序
    
    Args:
        features: 8维频率特征向量
        
    Returns:
        score: 频率综合得分（越高表示频率特征越强）
    """
    # 特征索引：0-谱质心, 1-谱带宽, 2-主频率, 3-谱滚降, 4-低频比, 5-中频比, 6-高频比, 7-谱熵
    spectral_centroid = features[0]
    spectral_bandwidth = features[1] 
    dominant_freq = features[2]
    spectral_rolloff = features[3]
    low_freq_ratio = features[4]
    mid_freq_ratio = features[5]
    high_freq_ratio = features[6]
    spectral_entropy = features[7]
    
    # 计算综合频率得分
    # 高频比重越高，主频越高，谱质心越高，得分越高
    frequency_score = (
        spectral_centroid * 0.25 +      # 谱质心权重
        dominant_freq * 0.25 +          # 主频率权重
        spectral_rolloff * 0.2 +        # 谱滚降权重
        high_freq_ratio * 0.15 +        # 高频比重权重
        mid_freq_ratio * 0.1 +          # 中频比重权重
        spectral_entropy * 0.05         # 谱熵权重
    )
    
    return frequency_score

def group_samples_by_frequency(X_data, y_data, num_groups=4):
    """
    按频率特征对样本进行分组，并按频率高低排序
    
    Args:
        X_data: 输入数据 (N, 1, 250, 90)
        y_data: 标签数据 (N,)
        num_groups: 分组数量
    
    Returns:
        groups: 按频率高低排序的分组结果字典
    """
    logger.info(f"开始按频率特征对 {len(X_data)} 个样本进行分组...")
    
    # 提取所有样本的频率特征
    all_features = []
    frequency_scores = []
    sample_indices = []
    
    for i in range(len(X_data)):
        if i % 500 == 0:
            logger.info(f"  提取特征进度: {i}/{len(X_data)}")
        
        csi_data = X_data[i, 0, :, :]  # (250, 90)
        features = extract_frequency_features(csi_data)
        score = calculate_frequency_score(features)
        
        all_features.append(features)
        frequency_scores.append(score)
        sample_indices.append(i)
    
    all_features = np.array(all_features)
    frequency_scores = np.array(frequency_scores)
    sample_indices = np.array(sample_indices)
    
    logger.info(f"特征提取完成，特征矩阵形状: {all_features.shape}")
    logger.info(f"频率得分范围: {np.min(frequency_scores):.3f} ~ {np.max(frequency_scores):.3f}")
    
    # 标准化特征
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features)
    
    # 使用K-means聚类进行分组
    logger.info(f"使用K-means进行{num_groups}组聚类...")
    kmeans = KMeans(n_clusters=num_groups, random_state=42, n_init=10)
    group_labels = kmeans.fit_predict(all_features_scaled)
    
    # 计算每个聚类的平均频率得分
    cluster_freq_scores = []
    for cluster_id in range(num_groups):
        cluster_mask = (group_labels == cluster_id)
        cluster_scores = frequency_scores[cluster_mask]
        avg_score = np.mean(cluster_scores)
        cluster_freq_scores.append((cluster_id, avg_score))
        logger.info(f"聚类 {cluster_id}: 平均频率得分 = {avg_score:.3f}")
    
    # 按频率得分排序聚类
    cluster_freq_scores.sort(key=lambda x: x[1])  # 从低到高排序
    
    # 创建聚类ID到频率等级的映射
    cluster_to_freq_level = {}
    freq_level_names = ['low_freq', 'mid_low_freq', 'mid_high_freq', 'high_freq']
    
    for i, (cluster_id, score) in enumerate(cluster_freq_scores):
        cluster_to_freq_level[cluster_id] = i
        logger.info(f"聚类 {cluster_id} -> {freq_level_names[i]} (得分: {score:.3f})")
    
    # 平衡分组（确保每组样本数量相近）
    target_size = len(sample_indices) // num_groups
    tolerance = max(1, target_size // 20)  # 5%的容忍度
    
    # 计算样本到各个聚类中心的距离
    distances = kmeans.transform(all_features_scaled)
    
    logger.info("平衡各组样本数量...")
    for iteration in range(100):
        group_sizes = [np.sum(group_labels == i) for i in range(num_groups)]
        
        if max(group_sizes) - min(group_sizes) <= tolerance:
            logger.info(f"  在第 {iteration} 次迭代后达到平衡")
            break
        
        # 从最大组移动样本到最小组
        max_group = np.argmax(group_sizes)
        min_group = np.argmin(group_sizes)
        
        max_group_mask = (group_labels == max_group)
        max_group_indices = np.where(max_group_mask)[0]
        
        distances_to_min = distances[max_group_indices, min_group]
        closest_sample_idx = max_group_indices[np.argmin(distances_to_min)]
        group_labels[closest_sample_idx] = min_group
    
    # 构建分组结果
    groups = {}
    class_names = ['lie_down', 'fall', 'walk', 'pickup', 'run', 'sit_down', 'stand_up']
    
    for cluster_id in range(num_groups):
        freq_level = cluster_to_freq_level[cluster_id]
        freq_name = freq_level_names[freq_level]
        
        group_mask = (group_labels == cluster_id)
        group_sample_indices = sample_indices[group_mask]
        group_freq_scores = frequency_scores[group_mask]
        
        # 统计类别分布
        group_labels_data = y_data[group_sample_indices]
        class_counts = np.bincount(group_labels_data.astype(int), minlength=7)
        
        groups[freq_level] = {
            'domain_name': freq_name,
            'cluster_id': cluster_id,
            'indices': group_sample_indices,
            'size': len(group_sample_indices),
            'class_distribution': class_counts,
            'cluster_center': kmeans.cluster_centers_[cluster_id],
            'avg_frequency_score': np.mean(group_freq_scores),
            'freq_score_range': (np.min(group_freq_scores), np.max(group_freq_scores))
        }
        
        logger.info(f"域 {freq_name}: {len(group_sample_indices)} 个样本")
        logger.info(f"  频率得分: {np.mean(group_freq_scores):.3f} (范围: {np.min(group_freq_scores):.3f}~{np.max(group_freq_scores):.3f})")
        for i, (class_name, count) in enumerate(zip(class_names, class_counts)):
            if count > 0:
                logger.info(f"  {class_name}: {count} ({count/len(group_sample_indices)*100:.1f}%)")
    
    return groups

def save_domain_data_by_frequency(ut_har_data, groups, output_root):
    """
    将按频率分组的数据保存到不同的文件夹
    
    Args:
        ut_har_data: 原始UT-HAR数据
        groups: 按频率分组的结果
        output_root: 输出根目录
    """
    logger.info(f"开始保存频率域数据到 {output_root}...")
    
    # 创建输出目录
    Path(output_root).mkdir(parents=True, exist_ok=True)
    
    # 类别名称映射
    class_names = ['lie_down', 'fall', 'walk', 'pickup', 'run', 'sit_down', 'stand_up']
    
    # 按频率等级排序保存
    sorted_groups = sorted(groups.items(), key=lambda x: x[0])  # 按频率等级排序
    
    for freq_level, group_info in sorted_groups:
        domain_name = group_info['domain_name']
        domain_dir = os.path.join(output_root, domain_name)
        
        # 创建域目录和子目录
        data_dir = os.path.join(domain_dir, 'data')
        label_dir = os.path.join(domain_dir, 'label')
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        Path(label_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"保存 {domain_name} ({group_info['size']} 个样本)...")
        
        # 获取该域的样本索引（只使用训练集的索引）
        indices = group_info['indices']
        
        # 只保存训练集数据
        X_key = 'X_train'
        y_key = 'y_train'
        
        # 训练集：直接使用聚类得到的索引
        domain_indices = indices[indices < len(ut_har_data[X_key])]
        if len(domain_indices) > 0:
            X_domain = ut_har_data[X_key][domain_indices]
            y_domain = ut_har_data[y_key][domain_indices]
        else:
            X_domain = np.empty((0, 1, 250, 90), dtype=ut_har_data[X_key].dtype)
            y_domain = np.empty((0,), dtype=ut_har_data[y_key].dtype)
        
        # 保存训练数据和标签
        data_file = os.path.join(data_dir, f'{X_key}.csv')
        label_file = os.path.join(label_dir, f'{y_key}.csv')
        
        with open(data_file, 'wb') as f:
            np.save(f, X_domain)
        
        with open(label_file, 'wb') as f:
            np.save(f, y_domain)
        
        logger.info(f"  训练集: {len(X_domain)} 个样本")
        
        # 保存域信息
        domain_info = {
            'domain_name': domain_name,
            'frequency_level': freq_level,
            'total_samples': group_info['size'],
            'avg_frequency_score': group_info['avg_frequency_score'],
            'freq_score_range': group_info['freq_score_range'],
            'class_distribution': group_info['class_distribution'].tolist(),
            'class_names': class_names,
            'cluster_center': group_info['cluster_center'].tolist(),
            'cluster_id': group_info['cluster_id'],
            'feature_names': [
                'spectral_centroid', 'spectral_bandwidth', 'dominant_freq', 
                'spectral_rolloff', 'low_freq_ratio', 'mid_freq_ratio', 
                'high_freq_ratio', 'spectral_entropy'
            ],
            'frequency_description': {
                0: '低频域 - 主要包含低频特征的信号',
                1: '中低频域 - 包含中等偏低频率特征的信号', 
                2: '中高频域 - 包含中等偏高频率特征的信号',
                3: '高频域 - 主要包含高频特征的信号'
            }[freq_level]
        }
        
        info_file = os.path.join(domain_dir, 'domain_info.json')
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(domain_info, f, indent=2, ensure_ascii=False)
    
    # 保存整体分组信息
    overall_info = {
        'description': 'UT-HAR数据集按频率特征划分的域',
        'total_samples': sum(group['size'] for group in groups.values()),
        'num_domains': len(groups),
        'class_names': class_names,
        'frequency_domains': {}
    }
    
    for freq_level, group_info in sorted_groups:
        domain_name = group_info['domain_name']
        overall_info['frequency_domains'][domain_name] = {
            'frequency_level': freq_level,
            'sample_count': group_info['size'],
            'avg_frequency_score': group_info['avg_frequency_score'],
            'freq_score_range': group_info['freq_score_range'],
            'class_distribution': group_info['class_distribution'].tolist()
        }
    
    overall_info_file = os.path.join(output_root, 'frequency_domains_overview.json')
    with open(overall_info_file, 'w', encoding='utf-8') as f:
        json.dump(overall_info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"频率域数据保存完成！输出目录: {output_root}")

def create_frequency_domain_datasets(data_root, output_root, num_domains=4):
    """
    主函数：创建频率域数据集
    
    Args:
        data_root: 原始数据根目录
        output_root: 输出根目录  
        num_domains: 域数量
    """
    logger.info("="*60)
    logger.info("UT-HAR数据集频率域划分开始")
    logger.info("="*60)
    
    # 检查输入数据
    ut_har_path = os.path.join(data_root, 'ut_har')
    if not os.path.exists(ut_har_path):
        logger.error(f"找不到UT-HAR数据目录: {ut_har_path}")
        return False
    
    try:
        # 1. 加载原始数据
        ut_har_data = load_ut_har_data(data_root)
        
        # 2. 基于训练集进行频率域划分
        logger.info("基于训练集进行频率域划分...")
        groups = group_samples_by_frequency(
            ut_har_data['X_train'], 
            ut_har_data['y_train'], 
            num_domains
        )
        
        # 3. 保存频率域数据
        save_domain_data_by_frequency(ut_har_data, groups, output_root)
        
        # 4. 打印总结信息
        logger.info("\n" + "="*60)
        logger.info("频率域划分完成！")
        logger.info("="*60)
        
        total_samples = sum(group['size'] for group in groups.values())
        logger.info(f"总样本数: {total_samples}")
        logger.info(f"域数量: {len(groups)}")
        
        # 按频率等级排序显示
        sorted_groups = sorted(groups.items(), key=lambda x: x[0])
        for freq_level, group_info in sorted_groups:
            percentage = (group_info['size'] / total_samples) * 100
            logger.info(f"{group_info['domain_name']}: {group_info['size']} 个样本 ({percentage:.1f}%), "
                       f"频率得分: {group_info['avg_frequency_score']:.3f}")
        
        logger.info(f"\n所有频率域数据已保存到: {output_root}")
        logger.info("频率域文件结构:")
        for freq_level, group_info in sorted_groups:
            domain_name = group_info['domain_name']
            logger.info(f"  {domain_name}/")
            logger.info(f"    ├── data/")
            logger.info(f"    │   └── X_train.csv")
            logger.info(f"    ├── label/")
            logger.info(f"    │   └── y_train.csv")
            logger.info(f"    └── domain_info.json")
        
        return True
        
    except Exception as e:
        logger.error(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 配置参数
    data_root = "/home/liruobin/FL/FederatedScope/data"  # 原始数据根目录
    output_root = "/home/liruobin/FL/FederatedScope/data/ut_har_frequency_domains"  # 输出根目录
    num_domains = 4  # 域数量
    
    # 执行频率域划分
    success = create_frequency_domain_datasets(data_root, output_root, num_domains)
    
    if success:
        print("\n✅ 频率域划分成功完成！")
    else:
        print("\n❌ 频率域划分失败！")
        sys.exit(1)
