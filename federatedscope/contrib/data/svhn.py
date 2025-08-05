import os
from torch.utils.data import Dataset
from PIL import Image
from scipy.io import loadmat
import numpy as np
from torchvision import transforms
from federatedscope.register import register_data
import logging
from federatedscope.core.data import BaseDataTranslator
from sklearn.model_selection import train_test_split
import scipy.io as sio

logger = logging.getLogger(__name__)


def load_svhn(data_root, transform):
    def load_mat_svhn(split):
        mat = sio.loadmat(os.path.join(data_root, f"SVHN/{split}_32x32.mat"))
        X = np.transpose(mat['X'], (3, 0, 1, 2))
        y = mat['y'].reshape(-1).astype(int)
        y[y == 10] = 0
        class SVHNDataset(Dataset):
            def __init__(self, X, y, transform):
                self.X, self.y, self.transform = X, y, transform
            def __len__(self):
                return len(self.y)
            def __getitem__(self, idx):
                img = Image.fromarray(self.X[idx])
                img = self.transform(img)
                return img, int(self.y[idx])
        return SVHNDataset(X, y, transform)
    return load_mat_svhn('train'), load_mat_svhn('test')

def load_svhn_data(config, client_cfgs=None):
    data_root = config.data.root

    # 加载数据集
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),  # 统一缩放到32x32
            # 将灰度图复制为3通道（如果是RGB则无变化）
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),  # 转为Tensor并归一化到[0,1]
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),  # 标准化到[-1,1]
        ]
    )

    logger.info("Loading SVHN dataset with preprocessing...")

    # Load preprocessed SVHN data (returns train, test)
    svhn_train, svhn_test = load_svhn(data_root, transform)

    # Use validation set as the middle split for FederatedScope format
    split_datasets_svhn = (svhn_train, svhn_test, svhn_test)

    translator = BaseDataTranslator(config, client_cfgs)
    data_dict = translator(split_datasets_svhn)

    logger.info(f"SVHN dataset loaded successfully:")
    logger.info(f"  Training samples: {len(svhn_train)}")
    logger.info(f"  Validation samples: {len(svhn_test)}")
    logger.info(f"  Test samples: {len(svhn_test)}")

    return data_dict, config


def call_svhn_data(config, client_cfgs=None):
    if config.data.type == "svhn":
        return load_svhn_data(config, client_cfgs)

register_data("svhn", call_svhn_data)