import numpy as np
from federatedscope.core.splitters import BaseSplitter

class IIDSplitter(BaseSplitter):
    """
    This splitter splits dataset following the independent and identically
    distribution (IID).
    
    Args:
        client_num: the dataset will be split into ``client_num`` pieces
    """
    def __init__(self, client_num):
        super(IIDSplitter, self).__init__(client_num)

    def __call__(self, dataset, prior=None):
        from torch.utils.data import Dataset, Subset

        length = len(dataset)
        index = np.arange(length)
        np.random.shuffle(index)

        # ✅ 用索引列表而不是 dataset 本体来切分
        idx_slices = np.array_split(index, self.client_num)

        if isinstance(dataset, Dataset):
            data_list = [Subset(dataset, idxs.tolist()) for idxs in idx_slices]
        else:
            data_list = [[dataset[idx] for idx in idxs] for idxs in idx_slices]

        return data_list