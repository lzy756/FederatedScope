import os
import numpy as np
from PIL import Image
from sklearn.datasets import load_svmlight_file
import scipy.io as sio

import torch
import torch.utils.data
from torch.utils.data import Subset, Dataset, TensorDataset
from torchvision import datasets, transforms

import logging
import json
from federatedscope.core.data import ClientData, StandaloneDataDict

logger = logging.getLogger(__name__)
import glob

from federatedscope.core.data import BaseDataTranslator
from federatedscope.register import register_data

class UTHarDataset(Dataset):
    def __init__(self, data: np.array, label: np.array):
        self.data = data
        self.targets = label

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index, :, :, :], int(self.targets[index])

    def __iter__(self):
        """Make the dataset iterable for model builder compatibility"""
        for i in range(len(self)):
            yield self[i]

def load_dataset(config, client_cfgs=None):
    data_dir = os.path.join(config.data.root, 'ut_har', 'data')
    label_dir = os.path.join(config.data.root, 'ut_har', 'label')
    
    data_list = glob.glob(os.path.join(data_dir, '*.csv'))
    label_list = glob.glob(os.path.join(label_dir, '*.csv'))
    
    ut_har_data = {}
    
    for data_dir in data_list:
        data_name = os.path.splitext(os.path.basename(data_dir))[0]
        with open(data_dir, 'rb') as f:
            data = np.load(f)
            data = data.reshape(len(data), 1, 250, 90)
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        ut_har_data[data_name] = torch.Tensor(data_norm)

    
    for label_dir in label_list:
        label_name = os.path.splitext(os.path.basename(label_dir))[0]
        with open(label_dir, 'rb') as f:
            label = np.load(f)
        ut_har_data[label_name] = torch.Tensor(label)
        
    
    # 输出每个数据项的类型、dtype和形状
    for key, value in ut_har_data.items():
        print(f"{key} 的类型: {type(value)}, dtype: {value.dtype}, shape: {value.shape}")
        
    train_dataset = UTHarDataset(ut_har_data['X_train'], ut_har_data['y_train'])
    val_dataset = UTHarDataset(ut_har_data['X_val'], ut_har_data['y_val'])
    test_dataset = UTHarDataset(ut_har_data['X_test'], ut_har_data['y_test'])
    
    split_datasets = (train_dataset, val_dataset, test_dataset)
    tmp_config = config.clone()
    tmp_config.merge_from_list(['federate.client_num', 25])
    translator = BaseDataTranslator(tmp_config, client_cfgs)
    data_dict_1 = translator(split_datasets)
    
    # 实现生成data_dict_3, 调用BaseDataTranslator但是使用两个ut_har数据集，前25个客户端为第一个ut_har数据集，后25个客户端为第二个ut_har数据集

    # 创建第一个数据集的BaseDataTranslator (客户端1-25)
    tmp_config_1 = config.clone()
    tmp_config_1.merge_from_list(['federate.client_num', 25])
    translator_1 = BaseDataTranslator(tmp_config_1, client_cfgs)
    data_dict_part1 = translator_1(split_datasets)

    # 创建第二个数据集 (复制相同的数据，实际应用中可以是不同的数据集)
    train_dataset_2 = UTHarDataset(ut_har_data['X_train'], ut_har_data['y_train'])
    val_dataset_2 = UTHarDataset(ut_har_data['X_val'], ut_har_data['y_val'])
    test_dataset_2 = UTHarDataset(ut_har_data['X_test'], ut_har_data['y_test'])
    split_datasets_2 = (train_dataset_2, val_dataset_2, test_dataset_2)

    # 创建第二个数据集的BaseDataTranslator (客户端26-50)
    tmp_config_2 = config.clone()
    tmp_config_2.merge_from_list(['federate.client_num', 25])
    translator_2 = BaseDataTranslator(tmp_config_2, client_cfgs)
    data_dict_part2 = translator_2(split_datasets_2)

    # 合并两个数据字典，调整客户端ID
    # 创建合并后的配置，总共50个客户端
    tmp_config_merged = config.clone()
    tmp_config_merged.merge_from_list(['federate.client_num', 50])

    # 合并数据字典
    merged_data_dict = {}

    # 添加服务器数据 (ID=0)，合并两个数据集的服务器数据
    from torch.utils.data import ConcatDataset
    server_train = ConcatDataset([data_dict_part1[0].train_data, data_dict_part2[0].train_data])
    server_val = ConcatDataset([data_dict_part1[0].val_data, data_dict_part2[0].val_data])
    server_test = ConcatDataset([data_dict_part1[0].test_data, data_dict_part2[0].test_data])
    merged_data_dict[0] = ClientData(tmp_config_merged, train=server_train, val=server_val, test=server_test)

    # 添加第一个数据集的客户端 (ID 1-25)
    for client_id in range(1, 26):
        merged_data_dict[client_id] = data_dict_part1[client_id]
        # 更新客户端配置为合并后的配置
        merged_data_dict[client_id].setup(tmp_config_merged)

    # 添加第二个数据集的客户端 (ID 26-50)
    for client_id in range(1, 26):
        new_client_id = client_id + 25  # 26-50
        merged_data_dict[new_client_id] = data_dict_part2[client_id]
        # 更新客户端配置为合并后的配置
        merged_data_dict[new_client_id].setup(tmp_config_merged)

    # 创建最终的StandaloneDataDict
    data_dict_3 = StandaloneDataDict(merged_data_dict, tmp_config_merged)

    # 实现生成 data_dict_2，不调用BaseDataTranslator但是使用完全相同的逻辑，要求得到与data_dict_1 完全一致
    # 1. 复制BaseDataTranslator的split_train_val_test逻辑
    dataset = split_datasets
    if isinstance(dataset, tuple):
        # No need to split train/val/test for tuple dataset.
        assert len(dataset) == 3, 'If dataset is tuple, it must contains train, valid and test split.'
        train, val, test = dataset[0], dataset[1], dataset[2]

    # 2. 复制BaseDataTranslator的split_to_client逻辑
    from federatedscope.core.auxiliaries.splitter_builder import get_splitter

    # 获取splitter
    splitter = get_splitter(tmp_config)

    # Initialization
    client_num = tmp_config.federate.client_num
    split_train, split_val, split_test = [[None] * client_num] * 3
    train_label_distribution = None

    # Split train/val/test to client
    if len(train) > 0:
        split_train = splitter(train)
        if tmp_config.data.consistent_label_distribution:
            try:
                train_label_distribution = [[j[1] for j in x] for x in split_train]
            except:
                logger.warning('Cannot access train label distribution for splitter.')
    if len(val) > 0:
        split_val = splitter(val, prior=train_label_distribution)
    if len(test) > 0:
        split_test = splitter(test, prior=train_label_distribution)

    # Build data dict with `ClientData`, key `0` for server.
    data_dict_raw = {
        0: ClientData(tmp_config, train=train, val=val, test=test)
    }
    for client_id in range(1, client_num + 1):
        if client_cfgs is not None:
            client_cfg = tmp_config.clone()
            client_cfg.merge_from_other_cfg(client_cfgs.get(f'client_{client_id}'))
        else:
            client_cfg = tmp_config
        data_dict_raw[client_id] = ClientData(client_cfg,
                                              train=split_train[client_id - 1],
                                              val=split_val[client_id - 1],
                                              test=split_test[client_id - 1])

    # 3. 创建StandaloneDataDict
    data_dict_2 = StandaloneDataDict(data_dict_raw, tmp_config)

    # 使用第三个data_dict作为最终结果
    data_dict = data_dict_3

    return data_dict, config

def call_cross_domain_data(config, client_cfgs=None):
    if config.data.type == "ut_har_test":
        return load_dataset(config, client_cfgs)

from federatedscope.register import register_data
register_data("ut_har_test", call_cross_domain_data)
