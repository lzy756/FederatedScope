from federatedscope.register import register_splitter
from federatedscope.core.splitters import BaseSplitter
import numpy as np
import random
from torch.utils.data import Subset

import logging

logger = logging.getLogger(__name__)


class RESISC45Splitter(BaseSplitter):
    def __init__(self, client_num):
        super(RESISC45Splitter, self).__init__(client_num)

    def __call__(self, dataset, *args, **kwargs):
        """
        为每个客户端分配随机的类别数据。允许数据在客户端之间重复。
        RESISC45有45个类别，可以通过kwargs指定每个客户端分配的类别数量。

        Args:
            dataset: 原始数据集的一个分区 (例如，训练集、验证集或测试集)，
                     它应该是一个 torch.utils.data.Dataset 实例。
            kwargs: 可选参数，包括 'classes_per_client'，用于指定每个客户端分配的类别数量。

        Returns:
            list: 每个元素是一个 torch.utils.data.Subset 对象，代表分配给一个客户端的数据。
        """
        random.seed(12345)
        np.random.seed(12345)

        # 获取数据集中每个样本的类别标签
        # 优化：尝试直接访问targets属性，如果不存在则逐个访问
        if hasattr(dataset, "targets"):
            # 对于torchvision数据集，直接使用targets属性
            labels = (
                dataset.targets.tolist()
                if hasattr(dataset.targets, "tolist")
                else list(dataset.targets)
            )
        elif hasattr(dataset, "labels"):
            # 对于某些数据集，标签存储在labels属性中
            labels = (
                dataset.labels.tolist()
                if hasattr(dataset.labels, "tolist")
                else list(dataset.labels)
            )
        else:
            # 回退到逐个访问（较慢）
            logger.info(
                f"Using slow label extraction for dataset of length {len(dataset)}"
            )
            labels = [dataset[i][1] for i in range(len(dataset))]

        # 获取所有唯一类别
        unique_classes = list(set(labels))
        total_classes = len(unique_classes)

        # 从kwargs获取每个客户端分配的类别数量，默认为2
        # client_num = kwargs.get("client_num", self.client_num)
        classes_per_client = kwargs.get("classes_per_client", 2)
        classes_per_client = min(classes_per_client, total_classes)

        # 为每个客户端选择类别
        client_selected_classes_list = []
        for _ in range(self.client_num):
            selected_classes_for_client = random.sample(
                unique_classes, classes_per_client
            )
            client_selected_classes_list.append(selected_classes_for_client)

        logger.info(client_selected_classes_list)

        # 为每个客户端收集样本索引 (相对于输入 dataset 的局部索引)
        client_sample_indices = [[] for _ in range(self.client_num)]

        # 按类别组织样本索引
        class_to_sample_indices = {c: [] for c in unique_classes}
        for idx, label in enumerate(labels):
            class_to_sample_indices[label].append(idx)

        # 为每个客户端分配其类别的样本
        # 如果多个客户端选择了相同的类别，这些类别下的所有样本会被分配给所有选择这些类别的客户端
        for client_idx, selected_classes in enumerate(client_selected_classes_list):
            current_client_indices_set = set()  # 用于确保每个客户端内部索引唯一
            for class_label in selected_classes:
                if class_label in class_to_sample_indices:
                    for sample_idx in class_to_sample_indices[class_label]:
                        current_client_indices_set.add(sample_idx)
            client_sample_indices[client_idx].extend(list(current_client_indices_set))

        # 将索引列表转换为 Subset 对象列表
        client_datasets = [Subset(dataset, idxs) for idxs in client_sample_indices]

        return client_datasets


def call_resisc45_splitter(splitter_type, client_num, **kwargs):
    if splitter_type == "RESISC45splitter":
        splitter = RESISC45Splitter(client_num)
        return splitter
    return None


register_splitter("RESISC45splitter", call_resisc45_splitter)
