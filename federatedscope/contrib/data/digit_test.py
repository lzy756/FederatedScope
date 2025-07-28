import os
from sklearn.datasets import load_svmlight_file
from torch.utils.data import Subset, Dataset, TensorDataset
from PIL import Image
import scipy.io as sio
import numpy as np
from torchvision import datasets, transforms
from federatedscope.register import register_data
import logging
from federatedscope.core.data import BaseDataTranslator
from federatedscope.core.data import ClientData, StandaloneDataDict

logger = logging.getLogger(__name__)


def load_usps(data_root, transform):
    usps_dir = os.path.join(data_root, 'USPS')
    X_tr_sp, y_tr = load_svmlight_file(os.path.join(usps_dir, 'usps'), n_features=256)
    X_te_sp, y_te = load_svmlight_file(os.path.join(usps_dir, 'usps.t'), n_features=256)
    X_tr = X_tr_sp.toarray().reshape(-1, 16, 16).astype('uint8')
    X_te = X_te_sp.toarray().reshape(-1, 16, 16).astype('uint8')
    y_tr = y_tr.astype(int) - 1
    y_te = y_te.astype(int) - 1

    class USPSDataset(TensorDataset):
        def __init__(self, X, y, transform):
            self.X, self.y, self.transform = X, y, transform
        def __len__(self):
            return len(self.y)
        def __getitem__(self, idx):
            img = Image.fromarray(self.X[idx], mode='L')
            img = self.transform(img)
            return img, int(self.y[idx])

    return USPSDataset(X_tr, y_tr, transform), USPSDataset(X_te, y_te, transform)

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

def load_syn(data_root, transform):
    from torchvision.datasets import ImageFolder
    syn_root = os.path.join(data_root, 'SYN')
    train_ds = ImageFolder(root=os.path.join(syn_root, 'imgs_train'), transform=transform)
    test_ds = ImageFolder(root=os.path.join(syn_root, 'imgs_valid'), transform=transform)
    return train_ds, test_ds

def load_cross_domain_data(config, client_cfgs=None):
    data_root = config.data.root

    # transform = transforms.Compose([
    #     transforms.Resize((28, 28)),
    #     transforms.Grayscale(num_output_channels=1),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5], std=[0.5]),
    # ])
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
    
    # MNIST: 40, USPS: 5, SVHN: 49, SYN: 6
    
    # 1. MNIST数据集
    logger.info("Loading MNIST dataset...")
    mnist_train = datasets.MNIST(root=data_root, train=True, transform=transform, download=True)
    mnist_test = datasets.MNIST(root=data_root, train=False, transform=transform, download=True)
    split_datasets_minist = (mnist_train, mnist_test, mnist_test)
    # 创建MNIST数据集的BaseDataTranslator (客户端1-40)
    tmp_config_1 = config.clone()
    tmp_config_1.merge_from_list(['federate.client_num', 40])
    translator_1 = BaseDataTranslator(tmp_config_1, client_cfgs)
    data_dict_part1 = translator_1(split_datasets_minist)
    
    # 2. USPS数据集
    logger.info("Loading USPS dataset...")
    usps_train, usps_test = load_usps(data_root, transform)
    split_datasets_usps = (usps_train, usps_test, usps_test)
    # 创建USPS数据集的BaseDataTranslator (客户端41-45)
    tmp_config_2 = config.clone()
    tmp_config_2.merge_from_list(['federate.client_num', 5])
    translator_2 = BaseDataTranslator(tmp_config_2, client_cfgs)
    data_dict_part2 = translator_2(split_datasets_usps)
    
    # # 3. SVHN数据集
    # logger.info("Loading SVHN dataset...")
    # svhn_train, svhn_test = load_svhn(data_root, transform)
    # split_datasets_svhn = (svhn_train, svhn_test, svhn_test)
    # # 创建SVHN数据集的BaseDataTranslator (客户端46-94)
    # tmp_config_3 = config.clone()
    # tmp_config_3.merge_from_list(['federate.client_num', 49])
    # translator_3 = BaseDataTranslator(tmp_config_3, client_cfgs)
    # data_dict_part3 = translator_3(split_datasets_svhn)
    
    # 合并三个数据字典，调整客户端ID
    # 创建合并后的配置
    tmp_config_merged = config.clone()
    tmp_config_merged.merge_from_list(['federate.client_num', 45])
    
    # 合并数据字典
    merged_data_dict = {}

    # 添加服务器数据 (ID=0)，合并三个数据集的服务器数据
    from torch.utils.data import ConcatDataset
    server_train = ConcatDataset([data_dict_part1[0].train_data, data_dict_part2[0].train_data])
    server_val = ConcatDataset([data_dict_part1[0].val_data, data_dict_part2[0].val_data])
    server_test = ConcatDataset([data_dict_part1[0].test_data, data_dict_part2[0].test_data])
    merged_data_dict[0] = ClientData(tmp_config_merged, train=server_train, val=server_val, test=server_test)

    # 添加MNIST数据集的客户端 (ID 1-40)
    for client_id in range(1, 41):
        merged_data_dict[client_id] = data_dict_part1[client_id]
        # 更新客户端配置为合并后的配置
        merged_data_dict[client_id].setup(tmp_config_merged)

    # 添加第二个数据集的客户端 (ID 41-45)
    for client_id in range(1, 6):
        new_client_id = client_id + 40  # 41-45
        merged_data_dict[new_client_id] = data_dict_part2[client_id]
        # 更新客户端配置为合并后的配置
        merged_data_dict[new_client_id].setup(tmp_config_merged)

    # # 添加第三个数据集的客户端 (ID 46-94)
    # for client_id in range(1, 50):
    #     new_client_id = client_id + 45  # 46-94
    #     merged_data_dict[new_client_id] = data_dict_part3[client_id]
    #     # 更新客户端配置为合并后的配置
    #     merged_data_dict[new_client_id].setup(tmp_config_merged)

    # 创建最终的StandaloneDataDict
    data_dict = StandaloneDataDict(merged_data_dict, tmp_config_merged)
    
    return data_dict, config

def call_cross_domain_data(config, client_cfgs=None):
    if config.data.type == "cross_domain_1":
        return load_cross_domain_data(config, client_cfgs)

register_data("cross_domain_1", call_cross_domain_data)