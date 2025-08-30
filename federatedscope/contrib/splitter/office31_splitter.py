from federatedscope.register import register_splitter
from federatedscope.core.splitters import BaseSplitter
import numpy as np
from torch.utils.data import Subset


class Office31DomainDirichletSplitter(BaseSplitter):
    """将“各域”分别分配给“同数量的客户端组”；
    在每个域内，按 Dirichlet(alpha) 在该组内的客户端之间分配样本。

    要点：
    - 数据集对象需包含 `domain_ids`（每个样本对应的域 id）与 `labels`。
    - 客户端按“出现的唯一域数”被均匀地划分为若干组；每个域映射到一个客户端组。
    - 仅将该域的样本分配给其对应组内的客户端；不同域互不混用。
    - 若客户端数少于域数，则某些组可能为空，会自动寻找最近的非空组。
    """
    def __init__(self, client_num, alpha: float = 0.5, **kwargs):
        super().__init__(client_num, **kwargs)
        self.alpha = float(alpha)

    @staticmethod
    def _split_clients_into_groups(total_clients: int, group_num: int = 3):
        # 尽量平均划分 client 到 group，返回
        # [group0_client_ids, group1_client_ids, group2_client_ids]
        base = total_clients // group_num
        rem = total_clients % group_num
        groups = []
        start = 0
        for g in range(group_num):
            size = base + (1 if g < rem else 0)
            groups.append(list(range(start, start + size)))
            start += size
        return groups

    def __call__(self, dataset, *args, **kwargs):
        # 提取底层含 domain_ids/labels 的 base 数据集，
        # 并找到链路中的 Subset（如果有）
        base = dataset
        subset_ref = None
        cur = dataset
        while hasattr(cur, 'dataset'):
            cur = cur.dataset
            if isinstance(cur, Subset):
                subset_ref = cur
        # 再继续下钻到最底层
        while hasattr(base, 'dataset'):
            base = base.dataset

        if not hasattr(base, 'domain_ids') or not hasattr(base, 'labels'):
            raise AttributeError('Office31DomainDirichletSplitter 需要数据集包含 '
                                 'domain_ids 和 labels 属性')

        # 构建本 split 的索引空间（使用全局索引），
        # 若有 Subset 则取其 indices
        if subset_ref is not None:
            global_indices = list(subset_ref.indices)
        else:
            global_indices = list(range(len(base)))

        # 将样本按域与类分桶：domain -> class -> [global_indices]
        domain_class_to_indices = {}
        for gi in global_indices:
            d = int(base.domain_ids[gi])
            c = int(base.labels[gi])
            domain_class_to_indices.setdefault(d, {}).setdefault(c,
                                                                 []).append(gi)

        # 按“出现的唯一域数”动态建组，并建立 dom_id -> group_idx 映射
        unique_domain_ids = sorted(
            {int(base.domain_ids[gi])
             for gi in global_indices})
        group_num = max(1, len(unique_domain_ids))
        dom_to_group_idx = {
            dom_id: i
            for i, dom_id in enumerate(unique_domain_ids)
        }
        groups = self._split_clients_into_groups(self.client_num, group_num)

        # 为每个客户端准备索引容器（相对当前 dataset 的局部索引）
        client_indices = [[] for _ in range(self.client_num)]

        rng = np.random.default_rng(12345)

        # 对每个域，在对应组内按类级 Dirichlet 分配样本
        for dom_id, cls_map in domain_class_to_indices.items():
            # 找到拥有此域的客户端组，若该组为空，则选择最近的非空组
            group_idx = dom_to_group_idx[int(dom_id)]
            search = 0
            clients_in_group = groups[group_idx]
            while len(clients_in_group) == 0 and search < group_num:
                group_idx = (group_idx + 1) % group_num
                clients_in_group = groups[group_idx]
                search += 1
            if len(clients_in_group) == 0:
                # 没有可用客户端，跳过（极端配置）
                continue

            for cls_id, idx_list in cls_map.items():
                idxs = list(idx_list)
                rng.shuffle(idxs)
                # 采样 Dirichlet 权重
                weights = rng.dirichlet(alpha=[self.alpha] *
                                        len(clients_in_group))
                n = len(idxs)
                shares = np.floor(weights * n).astype(int)
                # 补齐剩余
                while shares.sum() < n:
                    shares[np.argmax(weights)] += 1
                cursor = 0
                for cid_local, share in zip(clients_in_group, shares):
                    if share <= 0:
                        continue
                    part = idxs[cursor:cursor + share]
                    cursor += share
                    client_indices[cid_local].extend(part)

        # 构造 Subset 列表（相对于输入 dataset 的局部索引）
        # 若输入是 Subset，indices 已经是全局索引，传回 Subset(dataset.dataset, indices) 也可；
        # 但框架期望与输入同一个容器，所以直接用 Subset(dataset, reindexed) 的方式：
        # 需要将全局索引映射为当前 dataset 的局部索引
        # 情况 1：dataset 是 Subset：dataset.indices 是全局索引数组，本函数累积的也是全局索引。
        #        需把全局索引 -> 局部 [0..len(dataset)-1]。构造一个映射表。
        # 情况 2：dataset 不是 Subset：索引本身即为局部索引。
        # 将全局索引映射回当前传入 dataset 的局部索引空间
        if subset_ref is not None:
            # subset_ref.indices -> 全局索引数组；
            # 需要映射到 TransformWrapper(Subset(...)) 的局部索引
            global_to_local = {g: i for i, g in enumerate(subset_ref.indices)}
            local_client_indices = [[global_to_local[g] for g in idxs]
                                    for idxs in client_indices]
            client_datasets = [
                Subset(dataset, idxs) for idxs in local_client_indices
            ]
        else:
            # 传入 dataset 与 base 一致（无 Subset），可直接使用全局索引
            client_datasets = [
                Subset(dataset, idxs) for idxs in client_indices
            ]

        return client_datasets


def call_office31_splitter(splitter_type, client_num, **kwargs):
    if splitter_type == 'office31_domain_dirichlet':
        alpha = kwargs.get('alpha', 0.5)
        return Office31DomainDirichletSplitter(client_num,
                                               alpha=alpha,
                                               **kwargs)


register_splitter('office31_domain_dirichlet', call_office31_splitter)
