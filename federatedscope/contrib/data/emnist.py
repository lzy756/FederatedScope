import os
import shutil
from torchvision.datasets import EMNIST
from torchvision import transforms
from torch.utils.data import Dataset
from federatedscope.core.data.base_translator import BaseDataTranslator
from federatedscope.register import register_data


from torch.utils.data import Dataset, Subset

class TupleWrapperDataset(Dataset):
    """
    Wrap torchvision dataset or subset to ensure it returns (data, label)
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        item = self.dataset[index]
        # 如果是 Subset 或者 TupleWrapperDataset 包装的，item 会是个 tuple
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            return item[0], item[1]
        else:
            raise ValueError(f"Unexpected data format at index {index}: {item}")

    def __len__(self):
        return len(self.dataset)


def load_emnist(config, client_cfgs=None):
    # 读取 split 类型
    if hasattr(config.data, 'args') and isinstance(config.data.args, dict):
        split_type = config.data.args.get('split', 'byclass')
    else:
        split_type = 'byclass'
        print(f"[Warning] config.data.args is not a dict, using default split='byclass'.")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载 EMNIST 数据集
    data_train = EMNIST(
        root=config.data.root,
        split=split_type,
        train=True,
        transform=transform,
        download=True
    )
    data_test = EMNIST(
        root=config.data.root,
        split=split_type,
        train=False,
        transform=transform,
        download=True
    )

    # ✅ 包装为仅含 (data, label)
    data_train = TupleWrapperDataset(data_train)
    data_test = TupleWrapperDataset(data_test)

    print(f"[Info] Loaded EMNIST dataset: split={split_type}")
    print(f"       Training samples: {len(data_train)}")
    print(f"       Testing samples:  {len(data_test)}")
    print(f"       Num classes:      {len(set([label for _, label in data_train]))}")

    # 构建 FederatedScope 格式的数据
    translator = BaseDataTranslator(config, client_cfgs)
    fs_data = translator([data_train, [], data_test])
    return fs_data, config


def call_emnist(config, client_cfgs):
    if config.data.type.lower() == "emnist":
        print("[Info] Triggered EMNIST data loader...")
        return load_emnist(config, client_cfgs)


# 注册数据加载函数
register_data("emnist", call_emnist)
