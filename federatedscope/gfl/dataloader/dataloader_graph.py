from torch_geometric import transforms
from torch_geometric.datasets import TUDataset, MoleculeNet

from federatedscope.core.auxiliaries.transform_builder import get_transform
from federatedscope.gfl.dataset.cikm_cup import CIKMCUPDataset


def load_graphlevel_dataset(config=None):
    path = config.data.root
    name = config.data.type.upper()

    transforms_funcs, _, _ = get_transform(config, 'torch_geometric')

    # 单一子域
    if name in [
        'MUTAG','BZR','COX2','DHFR','PTC_MR','AIDS','NCI1',
        'ENZYMES','DD','PROTEINS','COLLAB','IMDB-BINARY','IMDB-MULTI',
        'REDDIT-BINARY'
    ]:
        if name in ['IMDB-BINARY','IMDB-MULTI'] and 'pre_transform' not in transforms_funcs:
            transforms_funcs['pre_transform'] = transforms.Constant(value=1.0, cat=False)
        dataset = TUDataset(path, name, **transforms_funcs)

    # MoleculeNet
    elif name in [
        'HIV','ESOL','FREESOLV','LIPO','PCBA','MUV','BACE','BBBP',
        'TOX21','TOXCAST','SIDER','CLINTOX'
    ]:
        dataset = MoleculeNet(path, name, **transforms_funcs)
        return dataset, config

    # 对 GRAPH_MULTI_DOMAIN_MOL，每个子域硬编码拆成 3 份
    elif name == 'GRAPH_MULTI_DOMAIN_MOL':
        # 子域列表
        sub_names = ['MUTAG','BZR','COX2','DHFR','PTC_MR','AIDS','NCI1']
        per_domain = 3  # 硬编码每个域拆成 3 份
        data_dict = {}
        cid = 1

        # runtime transform
        runtime_transform = transforms_funcs.get('transform', None)

        # 用于补 edge_attr 的辅助
        import torch

        from torch.utils.data import random_split, Dataset
        class ListDS(Dataset):
            def __init__(self, lst, transform=None):
                self.lst = lst
                self.transform = transform
            def __len__(self):
                return len(self.lst)
            def __getitem__(self, idx):
                d = self.lst[idx]
                # 补 edge_attr
                if (not hasattr(d, 'edge_attr')) or (d.edge_attr is None):
                    e = d.edge_index.size(1)
                    d.edge_attr = torch.zeros((e, 1), dtype=torch.float)
                # pad/truncate 到统一维度（如果你有 PadFeatures，可在 runtime_transform 中实现）
                if self.transform:
                    d = self.transform(d)
                return d

        # 对每个子域独立加载并拆分
        for dname in sub_names:
            ds = TUDataset(path, dname, **transforms_funcs)
            N = len(ds)
            base = N // per_domain
            rem  = N % per_domain
            sizes = [base + (1 if i < rem else 0) for i in range(per_domain)]
            splits = random_split(ds, sizes)
            for part in splits:
                data_dict[cid] = ListDS(part, transform=runtime_transform)
                cid += 1

        # 更新客户端数
        total_clients = cid - 1
        config.merge_from_list(['federate.client_num', total_clients])
        return data_dict, config

    # 其他 multi_domain，沿用默认或报错
    elif name.startswith('GRAPH_MULTI_DOMAIN'):
        raise ValueError(f'{name} 暂不支持该拆分方式')

    # CIKM
    elif name == 'CIKM':
        dataset = CIKMCUPDataset(path)

    else:
        raise ValueError(f'No dataset named: {name}!')

    # 默认单域多客户端
    client_num = (
        min(len(dataset), config.federate.client_num)
        if config.federate.client_num > 0 else len(dataset)
    )
    config.merge_from_list(['federate.client_num', client_num])
    data_dict = {i+1: dataset[i] for i in range(len(dataset))}
    return data_dict, config
