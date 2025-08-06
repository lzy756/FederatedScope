import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset
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


def load_dataset(config, client_cfgs=None):
    data_dir = os.path.join(config.data.root, "ut_har", "data")
    label_dir = os.path.join(config.data.root, "ut_har", "label")

    data_list = glob.glob(os.path.join(data_dir, "*.csv"))
    label_list = glob.glob(os.path.join(label_dir, "*.csv"))

    ut_har_data = {}

    for data_dir in data_list:
        data_name = os.path.splitext(os.path.basename(data_dir))[0]
        with open(data_dir, "rb") as f:
            data = np.load(f)
            data = data.reshape(len(data), 1, 250, 90)
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        ut_har_data[data_name] = torch.Tensor(data_norm)

    for label_dir in label_list:
        label_name = os.path.splitext(os.path.basename(label_dir))[0]
        with open(label_dir, "rb") as f:
            label = np.load(f)
        ut_har_data[label_name] = torch.Tensor(label)

    # 输出每个数据项的类型、dtype和形状
    for key, value in ut_har_data.items():
        print(
            f"{key} 的类型: {type(value)}, dtype: {value.dtype}, shape: {value.shape}"
        )

    train_dataset = UTHarDataset(ut_har_data["X_train"], ut_har_data["y_train"])
    val_dataset = UTHarDataset(ut_har_data["X_val"], ut_har_data["y_val"])
    test_dataset = UTHarDataset(ut_har_data["X_test"], ut_har_data["y_test"])

    split_datasets = (train_dataset, val_dataset, test_dataset)
    translator = BaseDataTranslator(config, client_cfgs)
    fs_data = translator(split_datasets)
    return fs_data, config


def call_ut_har_data(config, client_cfgs=None):
    if config.data.type == "ut_har":
        return load_dataset(config, client_cfgs)


# 注册数据集
register_data("ut_har", call_ut_har_data)
