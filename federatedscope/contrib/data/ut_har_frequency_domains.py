import os
import glob
import logging
import json
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataset import T_co
from federatedscope.core.data import BaseDataTranslator, ClientData, StandaloneDataDict
from federatedscope.register import register_data
from federatedscope.contrib.splitter.fedgs_splitter import FedGS_Splitter

logger = logging.getLogger(__name__)


##########################
# 聚类相关函数（使用FedGS_Splitter）
##########################

def get_label_distribution(subsets: list, num_classes=7):
    """
    计算每个子集的标签分布
    :param subsets: list of Subset objects or datasets
    :param num_classes: number of classes in the dataset (UT-HAR has 7 classes)
    :return: list of distributions for each subset
    """
    distributions = []
    for subset in subsets:
        labels = []
        # 获取标签
        if hasattr(subset, 'dataset'):
            # 如果是Subset对象
            for idx in subset.indices:
                _, label = subset.dataset[idx]
                labels.append(label)
        else:
            # 如果是普通dataset
            for i in range(len(subset)):
                _, label = subset[i]
                labels.append(label)

        labels = np.array(labels)
        dist = np.zeros(num_classes, dtype=int)
        for label in labels:
            dist[label] += 1
        distributions.append(dist)
    return distributions


def perform_clustering_with_fedgs(train_distributions,
                                  distance_threshold=0.5,
                                  min_groups=2,
                                  max_groups=10,
                                  balance_weight=2.0,
                                  balance_tolerance=0.2):
    """
    使用FedGS_Splitter进行聚类

    Args:
        train_distributions: 客户端训练数据分布列表
        distance_threshold: 距离阈值
        min_groups: 最小簇数量
        max_groups: 最大簇数量
        balance_weight: 大小均衡权重
        balance_tolerance: 大小均衡容忍度

    Returns:
        list: 聚类结果（peer communities）
    """
    n_clients = len(train_distributions)

    # 创建FedGS_Splitter实例
    splitter = FedGS_Splitter(
        client_num=n_clients,
        distance_threshold=distance_threshold,
        min_groups=min_groups,
        max_groups=max_groups,
        balance_weight=balance_weight,
        balance_tolerance=balance_tolerance
    )

    # 设置数据分布信息
    splitter.data_info = []
    for dist in train_distributions:
        splitter.data_info.append({"distribution": np.array(dist)})

    # 执行聚类
    logger.info(f"使用FedGS_Splitter进行聚类: {n_clients}个客户端")
    splitter.build_peer_communities(train_distributions)

    return splitter.peer_communities


class UTHarDataset(Dataset):
    def __init__(self, data: np.array, label: np.array):
        self.data = data
        self.targets = label

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index, :, :, :], int(self.targets[index])


def _load_frequency_domain_dataset(domain_name, data_root):
    """
    加载单个频率域数据集
    
    Args:
        domain_name: 频率域名称 (low_freq, mid_low_freq, mid_high_freq, high_freq)
        data_root: 数据根目录
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    try:
        # 构建频率域数据路径
        domain_path = os.path.join(data_root, 'ut_har_frequency_domains', domain_name)
        data_dir = os.path.join(domain_path, 'data')
        label_dir = os.path.join(domain_path, 'label')
        
        if not os.path.exists(domain_path):
            raise FileNotFoundError(f"频率域数据目录不存在: {domain_path}")
        
        # 加载该域的训练数据
        domain_data = {}
        
        # 只加载训练数据文件
        train_data_file = 'X_train.csv'
        train_label_file = 'y_train.csv'
        
        # 加载训练数据
        train_data_path = os.path.join(data_dir, train_data_file)
        if os.path.exists(train_data_path):
            with open(train_data_path, 'rb') as f:
                data = np.load(f)
                # 确保数据形状正确 (N, 1, 250, 90)
                if len(data.shape) == 3:
                    data = data.reshape(len(data), 1, 250, 90)
                # 归一化
                if data.size > 0:
                    data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
                else:
                    data_norm = data
            domain_data['X_train'] = torch.Tensor(data_norm)
            logger.info(f"加载 {domain_name}/X_train: shape={data_norm.shape}")
        
        # 加载训练标签
        train_label_path = os.path.join(label_dir, train_label_file)
        if os.path.exists(train_label_path):
            with open(train_label_path, 'rb') as f:
                label = np.load(f)
            domain_data['y_train'] = torch.Tensor(label)
            logger.info(f"加载 {domain_name}/y_train: shape={label.shape}")
        
        # 创建训练数据集
        train_dataset = UTHarDataset(domain_data['X_train'], domain_data['y_train'])
        
        logger.info(f"加载 {domain_name} 域: train={len(train_dataset)}")
        return train_dataset
        
    except Exception as e:
        logger.error(f"加载 {domain_name} 域失败: {e}")
        raise e


def _load_complete_val_test_datasets(data_root):
    """
    加载原始完整的验证集和测试集
    
    Args:
        data_root: 数据根目录
        
    Returns:
        tuple: (val_dataset, test_dataset)
    """
    try:
        # 原始数据路径
        original_data_dir = os.path.join(data_root, 'ut_har', 'data')
        original_label_dir = os.path.join(data_root, 'ut_har', 'label')
        
        if not os.path.exists(original_data_dir):
            raise FileNotFoundError(f"原始数据目录不存在: {original_data_dir}")
        
        # 加载原始验证集和测试集
        original_data = {}
        
        # 验证集和测试集文件
        data_files = ['X_val.csv', 'X_test.csv']
        label_files = ['y_val.csv', 'y_test.csv']
        
        # 加载数据文件
        for data_file in data_files:
            data_file_path = os.path.join(original_data_dir, data_file)
            if os.path.exists(data_file_path):
                data_name = os.path.splitext(data_file)[0]
                with open(data_file_path, 'rb') as f:
                    data = np.load(f)
                    # 确保数据形状正确 (N, 1, 250, 90)
                    if len(data.shape) == 3:
                        data = data.reshape(len(data), 1, 250, 90)
                    # 归一化
                    if data.size > 0:
                        data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
                    else:
                        data_norm = data
                original_data[data_name] = torch.Tensor(data_norm)
                logger.info(f"加载原始 {data_name}: shape={data_norm.shape}")
        
        # 加载标签文件
        for label_file in label_files:
            label_file_path = os.path.join(original_label_dir, label_file)
            if os.path.exists(label_file_path):
                label_name = os.path.splitext(label_file)[0]
                with open(label_file_path, 'rb') as f:
                    label = np.load(f)
                original_data[label_name] = torch.Tensor(label)
                logger.info(f"加载原始 {label_name}: shape={label.shape}")
        
        # 创建验证集和测试集
        val_dataset = UTHarDataset(original_data['X_val'], original_data['y_val'])
        test_dataset = UTHarDataset(original_data['X_test'], original_data['y_test'])
        
        logger.info(f"加载原始完整数据集: val={len(val_dataset)}, test={len(test_dataset)}")
        return val_dataset, test_dataset
        
    except Exception as e:
        logger.error(f"加载原始验证集和测试集失败: {e}")
        raise e


def _create_federated_data_dict(config, client_cfgs, train_dataset, val_dataset, test_dataset, domain_name, domain_client_num):
    """
    为单个频率域创建联邦数据字典
    
    Args:
        config: 配置对象
        client_cfgs: 客户端配置
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        test_dataset: 测试数据集
        domain_name: 域名称
        domain_client_num: 该域的客户端数量
        
    Returns:
        dict: 联邦数据字典
    """
    # 为该域创建临时配置
    tmp_config = config.clone()
    tmp_config.merge_from_list(['federate.client_num', domain_client_num])
    
    # 使用BaseDataTranslator转换为联邦格式
    translator = BaseDataTranslator(tmp_config, client_cfgs)
    
    # 创建数据集元组
    split_datasets = (train_dataset, val_dataset, test_dataset)
    data_dict = translator(split_datasets)
    
    logger.info(f"成功创建 {domain_name} 频率域联邦数据，客户端数量: {len(data_dict) - 1}")
    return data_dict


def _merge_data_dictionaries(config, data_dicts, domain_client_nums):
    """
    合并多个联邦数据字典
    
    Args:
        config: 配置对象
        data_dicts: 联邦数据字典列表
        domain_client_nums: 每个域的客户端数量列表
        
    Returns:
        StandaloneDataDict: 合并后的联邦数据字典
    """
    total_clients = sum(domain_client_nums)
    
    # 创建合并配置
    tmp_config_merged = config.clone()
    tmp_config_merged.merge_from_list(['federate.client_num', total_clients])
    
    # 创建合并数据字典
    merged_data_dict = {}
    
    # 合并服务器数据 (ID=0)
    server_train = ConcatDataset([data_dict[0].train_data for data_dict in data_dicts])
    server_val = ConcatDataset([data_dict[0].val_data for data_dict in data_dicts])
    server_test = ConcatDataset([data_dict[0].test_data for data_dict in data_dicts])
    merged_data_dict[0] = ClientData(tmp_config_merged, train=server_train, val=server_val, test=server_test)
    
    # 添加来自各个数据字典的客户端，使用正确的ID偏移
    client_id_offset = 0
    for data_dict, domain_client_num in zip(data_dicts, domain_client_nums):
        for client_id in range(1, domain_client_num + 1):
            new_client_id = client_id + client_id_offset
            merged_data_dict[new_client_id] = data_dict[client_id]
            # 更新客户端配置为合并配置
            merged_data_dict[new_client_id].setup(tmp_config_merged)
        
        # 更新下一个域的偏移量
        client_id_offset += domain_client_num
    
    # 创建最终的StandaloneDataDict
    return StandaloneDataDict(merged_data_dict, tmp_config_merged)


def _perform_clustering_analysis(data_dict, data_root):
    """
    使用FedGS_Splitter对联邦数据进行聚类分析
    
    Args:
        data_dict: 联邦数据字典
        data_root: 数据根目录，用于保存结果
    """
    try:
        # 收集训练和测试子集
        train_subsets = []
        test_subsets = []
        
        # 从每个客户端提取数据（跳过服务器id=0）
        for client_id in range(1, len(data_dict)):
            client_data = data_dict[client_id]
            
            # 访问原始数据集而不是DataLoaders
            if hasattr(client_data, 'train_data') and client_data.train_data is not None:
                train_subsets.append(client_data.train_data)
            if hasattr(client_data, 'test_data') and client_data.test_data is not None:
                test_subsets.append(client_data.test_data)
        
        logger.info(f"收集到 {len(train_subsets)} 个训练子集, {len(test_subsets)} 个测试子集")
        
        if train_subsets and test_subsets:
            # 计算标签分布
            logger.info("计算客户端标签分布...")
            train_distributions = get_label_distribution(train_subsets, num_classes=7)
            test_distributions = get_label_distribution(test_subsets, num_classes=7)
            
            # 使用FedGS_Splitter执行聚类
            logger.info("使用FedGS_Splitter执行客户端聚类...")
            peer_communities = perform_clustering_with_fedgs(
                train_distributions,
                distance_threshold=0.5,
                min_groups=2,
                max_groups=10,
                balance_weight=2.0,
                balance_tolerance=0.2
            )
            
            logger.info(f"聚类结果: {peer_communities}")
            
            # 使用FedGS_Splitter的保存方法保存聚类结果
            n_clients = len(train_distributions)
            splitter = FedGS_Splitter(client_num=n_clients)
            
            # 设置数据和社区信息用于保存
            splitter.data_info = [{"distribution": np.array(dist)} for dist in train_distributions]
            splitter.test_data_info = [{"distribution": np.array(dist)} for dist in test_distributions]
            splitter.peer_communities = peer_communities
            
            # 保存到指定位置
            cluster_file = os.path.join(data_root, "peer_communities.json")
            splitter.save_cluster_results(cluster_file)
            
        else:
            logger.warning("无法获取客户端数据子集，跳过聚类步骤")
            
    except Exception as e:
        logger.error(f"聚类过程中出现错误: {e}")
        logger.info("继续返回联邦数据，但没有聚类信息")


def load_ut_har_frequency_domains_dataset(config, client_cfgs=None):
    """
    加载UT-HAR频率域数据集进行联邦学习
    
    Args:
        config: 包含数据设置的配置对象
        client_cfgs: 客户端特定配置（可选）
        
    Returns:
        tuple: (datadict, config) 包含联邦数据集
    """
    data_root = config.data.root
    total_clients = config.federate.client_num
    
    # 首先加载原始完整的验证集和测试集
    logger.info("加载原始完整验证集和测试集...")
    complete_val_dataset, complete_test_dataset = _load_complete_val_test_datasets(data_root)
    
    # 定义频率域名称（按频率从低到高）
    domain_names = ['low_freq', 'mid_low_freq', 'mid_high_freq', 'high_freq']
    domain_train_datasets = []
    domain_sample_counts = []
    
    # 加载所有频率域的训练数据集并计算样本数量
    for domain_name in domain_names:
        # 加载频率域训练数据集
        train_dataset = _load_frequency_domain_dataset(domain_name, data_root)
        domain_train_datasets.append(train_dataset)
        
        # 计算该域的样本数量（使用训练样本进行比例计算）
        sample_count = len(train_dataset)
        domain_sample_counts.append(sample_count)
        logger.info(f"频率域 {domain_name}: {sample_count} 个训练样本")
    
    # 基于样本比例计算客户端分配
    total_samples = sum(domain_sample_counts)
    domain_client_nums = []
    allocated_clients = 0
    
    for i, sample_count in enumerate(domain_sample_counts):
        if i == len(domain_sample_counts) - 1:
            # 最后一个域获得剩余客户端
            domain_client_num = total_clients - allocated_clients
        else:
            # 计算比例分配
            proportion = sample_count / total_samples
            domain_client_num = max(1, round(proportion * total_clients))  # 每个域至少1个客户端
            allocated_clients += domain_client_num
        
        domain_client_nums.append(domain_client_num)
        logger.info(f"频率域 {domain_names[i]}: {domain_client_num} 个客户端 "
                   f"(比例: {sample_count/total_samples:.3f})")
    
    # 验证总分配
    actual_total = sum(domain_client_nums)
    if actual_total != total_clients:
        logger.warning(f"客户端分配不匹配: 期望 {total_clients}, 实际 {actual_total}")
        # 调整最后一个域以精确匹配
        domain_client_nums[-1] += (total_clients - actual_total)
        logger.info(f"调整最后域为 {domain_client_nums[-1]} 个客户端")
    
    # 为每个频率域创建联邦数据字典（使用相同的完整验证集和测试集）
    data_dicts = []
    for i, (domain_name, train_dataset, domain_client_num) in enumerate(
            zip(domain_names, domain_train_datasets, domain_client_nums)):
        
        # 为该域创建联邦数据字典，所有域使用相同的验证集和测试集
        data_dict = _create_federated_data_dict(
            config, client_cfgs, train_dataset, complete_val_dataset, complete_test_dataset, 
            domain_name, domain_client_num
        )
        data_dicts.append(data_dict)
    
    # 合并所有数据字典
    merged_data_dict = _merge_data_dictionaries(config, data_dicts, domain_client_nums)
    
    # 执行聚类分析
    _perform_clustering_analysis(merged_data_dict, data_root)
    
    return merged_data_dict, config


def call_ut_har_frequency_domains_data(config, client_cfgs=None):
    """
    UT-HAR频率域数据集注册入口点
    
    Args:
        config: 配置对象
        client_cfgs: 客户端特定配置（可选）
        
    Returns:
        tuple: (datadict, config) 如果数据集类型匹配，否则返回None
    """
    if config.data.type.lower() == "ut_har_frequency_domains":
        logger.info("加载UT-HAR频率域数据集...")
        return load_ut_har_frequency_domains_dataset(config, client_cfgs)
    return None


# 注册UT-HAR频率域数据集
register_data("ut_har_frequency_domains", call_ut_har_frequency_domains_data)