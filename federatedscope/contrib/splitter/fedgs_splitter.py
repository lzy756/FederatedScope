from federatedscope.register import register_splitter
from federatedscope.core.splitters import BaseSplitter
import numpy as np
import logging
from sklearn.cluster import AgglomerativeClustering
from scipy.optimize import linear_sum_assignment
import os
import json

logger = logging.getLogger(__name__)


class FedGS_Splitter(BaseSplitter):
    def __init__(self,
                 client_num,
                 alpha=0.5,
                 distance_threshold=2.0,
                 min_groups=2,
                 max_groups=10):
        self.alpha = alpha
        self.distance_threshold = distance_threshold  # 距离阈值 ζ，默认值调大
        self.min_groups = min_groups  # 最小社区数量 κ
        self.max_groups = max_groups  # 最大社区数量
        self.data_info = [
        ]  # 2D array storing training data distribution info for each client
        self.test_data_info = []  # Store test set data distribution
        self.peer_communities = []  # List of peer communities
        self.is_initialized = False  # Mark whether training data distribution has been recorded
        self.call_count = 0  # 记录__call__被调用的次数
        super(FedGS_Splitter, self).__init__(client_num)

    def save_data_info(self, save_path='data_info.txt'):
        """Save client data distribution information to text file
        
        Args:
            save_path: Save path, defaults to 'data_info.txt'
        """
        # Ensure directory exists
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
            exist_ok=True)

        # Format data
        formatted_data = []
        for client_idx, dist in enumerate(self.data_info):
            # Convert NumPy array to normal list and ensure numeric values
            if isinstance(dist, dict) and 'distribution' in dist:
                dist = dist['distribution']
            dist = np.array(dist, dtype=np.float32)
            client_total = float(np.sum(dist))
            percentages = [
                f"{(count/client_total*100):.2f}%" for count in dist
            ]

            formatted_data.append({
                'client_id': client_idx + 1,
                'distribution': dist.tolist(),
                'percentages': percentages,
                'total_samples': client_total
            })

        # Save as text file
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("Client Data Distribution Information\n")
            f.write("=" * 50 + "\n\n")
            for data in formatted_data:
                f.write(f"Client #{data['client_id']}\n")
                f.write(f"Distribution: {data['distribution']}\n")
                f.write(f"Percentages: {data['percentages']}\n")
                f.write(f"Total Samples: {data['total_samples']}\n")
                f.write("-" * 30 + "\n")

            # Save raw data in JSON format (for program reading)
            f.write("\nRaw Data (JSON format):\n")
            json_data = {
                "data_info": [data['distribution'] for data in formatted_data]
            }
            f.write(json.dumps(json_data, indent=2))

        logger.info(f'Data distribution information saved to: {save_path}')

    def save_test_data_info(self, save_path='test_data_info.txt'):
        """Save test set data distribution information to text file
        
        Args:
            save_path: Save path, defaults to 'test_data_info.txt'
        """
        # Ensure directory exists
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
            exist_ok=True)

        # Format data
        formatted_data = []
        for client_idx, dist in enumerate(self.test_data_info):
            # Convert NumPy array to normal list and ensure numeric values
            if isinstance(dist, dict) and 'distribution' in dist:
                dist = dist['distribution']
            dist = np.array(dist, dtype=np.float32)
            client_total = float(np.sum(dist))
            percentages = [
                f"{(count/client_total*100):.2f}%" for count in dist
            ]

            formatted_data.append({
                'client_id': client_idx + 1,
                'distribution': dist.tolist(),
                'percentages': percentages,
                'total_samples': client_total
            })

        # Save as text file
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("Test Set Data Distribution Information\n")
            f.write("=" * 50 + "\n\n")
            for data in formatted_data:
                f.write(f"Client #{data['client_id']}\n")
                f.write(f"Distribution: {data['distribution']}\n")
                f.write(f"Percentages: {data['percentages']}\n")
                f.write(f"Total Samples: {data['total_samples']}\n")
                f.write("-" * 30 + "\n")

            # Save raw data in JSON format (for program reading)
            f.write("\nRaw Data (JSON format):\n")
            json_data = {
                "test_data_info": [
                    data['distribution'] for data in formatted_data
                ]
            }
            f.write(json.dumps(json_data, indent=2))

        logger.info(
            f'Test set data distribution information saved to: {save_path}')

    def save_peer_communities(self, save_path='peer_communities.txt'):
        """Save peer community grouping information to text file
        
        Args:
            save_path: Save path, defaults to 'peer_communities.txt'
        """
        # Ensure directory exists
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
            exist_ok=True)

        # Save as text file
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("Peer Community Grouping Information\n")
            f.write("=" * 50 + "\n\n")
            f.write(
                f"Total number of communities: {len(self.peer_communities)}\n\n"
            )

            for idx, community in enumerate(self.peer_communities):
                # Convert NumPy array to normal list
                community = community.tolist() if isinstance(
                    community, np.ndarray) else community
                # 只在显示时将客户端ID加1
                display_community = [cid + 1 for cid in community]
                f.write(f"Community #{idx + 1}\n")
                f.write(f"Members: {display_community}\n")

                if hasattr(self, 'data_info') and self.data_info:
                    # 使用原始的0基索引访问数据
                    community_dist = np.sum(
                        [self.data_info[i]['distribution'] for i in community],
                        axis=0)
                    total_samples = np.sum(community_dist)
                    percentages = [
                        f"{(count/total_samples*100):.2f}%"
                        for count in community_dist
                    ]

                    f.write(
                        f"Community overall distribution: {community_dist.tolist()}\n"
                    )
                    f.write(f"Distribution percentages: {percentages}\n")
                    f.write(f"Total Samples: {total_samples}\n")

                f.write("-" * 30 + "\n")

            # Save raw data in JSON format (for program reading)
            f.write("\nRaw Data (JSON format):\n")
            # 将JSON数据中的客户端ID也改为1基索引
            json_data = {
                "peer_communities": [[
                    cid + 1 for cid in
                    (comm.tolist() if isinstance(comm, np.ndarray) else comm)
                ] for comm in self.peer_communities]
            }
            f.write(json.dumps(json_data, indent=2))

        logger.info(
            f'Peer community grouping information saved to: {save_path}')

    def load_data_info(self, load_path='data_info.txt'):
        """Load client data distribution information from text file
        
        Args:
            load_path: Load path, defaults to 'data_info.txt'
        """
        if os.path.exists(load_path):
            with open(load_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Find JSON data section
                json_start = content.find('\nRaw Data (JSON format):\n') + len(
                    '\nRaw Data (JSON format):\n')
                json_data = json.loads(content[json_start:])
                self.data_info = json_data['data_info']
                logger.info(
                    f'Loaded data distribution information from {load_path}')
                return True
        else:
            logger.warning(
                f'Data distribution file {load_path} does not exist')
            return False

    def load_peer_communities(self, load_path='peer_communities.txt'):
        """Load peer community grouping information from text file
        
        Args:
            load_path: Load path, defaults to 'peer_communities.txt'
        """
        if os.path.exists(load_path):
            with open(load_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Find JSON data section
                json_start = content.find('\nRaw Data (JSON format):\n') + len(
                    '\nRaw Data (JSON format):\n')
                json_data = json.loads(content[json_start:])
                # 将1基索引转换回0基索引
                self.peer_communities = [[
                    cid - 1 for cid in comm
                ] for comm in json_data['peer_communities']]
                logger.info(
                    f'Loaded peer community grouping information from {load_path}'
                )
                return True
        else:
            logger.warning(
                f'Peer community grouping file {load_path} does not exist')
            return False

    def build_peer_communities(self, client_distributions):
        """Build peer communities (PCs) using distance-based hierarchical clustering
        
        Args:
            client_distributions: List of client data distributions
        """
        # 计算客户端之间的欧氏距离矩阵
        n_clients = len(client_distributions)
        distance_matrix = np.zeros((n_clients, n_clients))
        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                dist = np.linalg.norm(client_distributions[i] -
                                      client_distributions[j])
                distance_matrix[i][j] = dist
                distance_matrix[j][i] = dist

        # 二分查找合适的阈值，使得社区数量在[min_groups, max_groups]范围内
        left, right = 0, np.max(distance_matrix)
        best_threshold = self.distance_threshold
        best_labels = None
        best_num_clusters = n_clients

        while left < right:
            mid = (left + right) / 2
            clustering = AgglomerativeClustering(n_clusters=None,
                                                 distance_threshold=mid,
                                                 metric='precomputed',
                                                 linkage='complete')
            cluster_labels = clustering.fit_predict(distance_matrix)
            num_clusters = len(np.unique(cluster_labels))
            if self.min_groups <= num_clusters <= self.max_groups:
                if abs(num_clusters - (self.min_groups + self.max_groups) / 2) < \
                   abs(best_num_clusters - (self.min_groups + self.max_groups) / 2):
                    best_threshold = mid
                    best_labels = cluster_labels
                    best_num_clusters = num_clusters
            if num_clusters > self.max_groups:
                left = mid + 1e-6
            else:
                right = mid - 1e-6
            if right - left < 1e-6:
                break
        if best_labels is None:
            clustering = AgglomerativeClustering(n_clusters=max(
                min(n_clients, self.max_groups), self.min_groups),
                                                 metric='precomputed',
                                                 linkage='complete')
            best_labels = clustering.fit_predict(distance_matrix)
            best_num_clusters = len(np.unique(best_labels))
            logger.warning(
                f"Could not find suitable threshold, forcing {best_num_clusters} communities"
            )
        # 构建对等社区
        unique_labels = np.unique(best_labels)
        self.peer_communities = []
        for label in unique_labels:
            community = np.where(best_labels == label)[0]
            self.peer_communities.append(community)
        # ========== 组内人数均衡后处理 ==========
        target_size = int(np.ceil(n_clients / len(self.peer_communities)))
        min_size = target_size - 1
        max_size = target_size + 1
        group_sizes = [len(pc) for pc in self.peer_communities]
        while True:
            changed = False
            max_idx = np.argmax(group_sizes)
            min_idx = np.argmin(group_sizes)
            if group_sizes[max_idx] > max_size and group_sizes[
                    min_idx] < min_size:
                big_group = self.peer_communities[max_idx]
                # 计算大组内每个客户端到组内其他客户端的平均距离
                avg_dists = []
                for i in big_group:
                    d = np.mean(
                        [distance_matrix[i][j] for j in big_group if j != i])
                    avg_dists.append(d)
                move_idx = np.argmax(avg_dists)
                client_to_move = big_group[move_idx]
                # 从大组移除，加入小组
                self.peer_communities[max_idx] = np.delete(big_group, move_idx)
                self.peer_communities[min_idx] = np.append(
                    self.peer_communities[min_idx], client_to_move)
                group_sizes[max_idx] -= 1
                group_sizes[min_idx] += 1
                changed = True
            if not changed:
                break
        # ========== 组内人数均衡后处理结束 ==========
        # 计算并保存每个社区的平均分布
        community_distributions = []
        for community in self.peer_communities:
            community_dist = np.zeros_like(client_distributions[0])
            total_samples = 0
            for client_idx in community:
                client_dist = self.data_info[client_idx]['distribution']
                client_samples = np.sum(client_dist)
                community_dist += client_dist
                total_samples += client_samples
            if total_samples > 0:
                community_dist = community_dist / total_samples
            community_distributions.append(community_dist)
        self.save_peer_communities()
        logger.info("\n========== Peer Communities Summary ==========")
        logger.info(
            f"Number of Peer Communities (PCs): {len(self.peer_communities)}")
        logger.info(f"Distance threshold used: {best_threshold:.4f}")
        for idx, (pc, pc_dist) in enumerate(
                zip(self.peer_communities, community_distributions)):
            logger.info(f"\nPeer Community {idx + 1}:")
            logger.info(f"Number of clients: {len(pc)}")
            # 显示时将客户端ID加1
            display_indices = [i + 1 for i in pc.tolist()]
            logger.info(f"Client indices: {display_indices}")
            logger.info(
                f"Normalized community distribution: {pc_dist.tolist()}")
            logger.info(
                f"Distribution percentages: {['%.2f%%' % (p*100) for p in pc_dist]}"
            )
            max_internal_dist = 0
            for i in pc:
                for j in pc:
                    if i < j:
                        max_internal_dist = max(max_internal_dist,
                                                distance_matrix[i][j])
            logger.info(f"Maximum internal distance: {max_internal_dist:.4f}")

            logger.info("\nDetailed client distributions in this community:")
            for client_idx in pc:
                client_dist = self.data_info[client_idx]['distribution']
                client_total = np.sum(client_dist)
                client_percentages = (client_dist / client_total *
                                      100).tolist()
                # 显示时将客户端ID加1
                display_id = client_idx + 1
                logger.info(f"Client {display_id}:")
                logger.info(f"  Sample counts: {client_dist.tolist()}")
                logger.info(
                    f"  Percentages: {['%.2f%%' % p for p in client_percentages]}"
                )
                logger.info(f"  Total samples: {client_total}")
        logger.info("\n============================================")

    def __call__(self, dataset, prior=None, **kwargs):
        from torch.utils.data import Dataset, Subset

        self.call_count += 1
        tmp_dataset = [ds for ds in dataset]
        label = np.array([y for x, y in tmp_dataset])
        idx_slice = dirichlet_distribution_noniid_slice(label,
                                                        self.client_num,
                                                        self.alpha,
                                                        prior=prior)

        # 用调用次数判断数据集类型
        if self.call_count == 1:
            # 训练集
            unique_labels = np.unique(label)
            num_classes = len(unique_labels)
            client_distributions = []
            for client_idx, client_indices in enumerate(idx_slice):
                client_indices = np.array(client_indices)
                client_labels = label[client_indices]
                client_dist = np.zeros(num_classes, dtype=int)
                for label_idx, label_class in enumerate(unique_labels):
                    mask = (client_labels == float(label_class))
                    client_dist[label_idx] = np.sum(mask)
                client_distributions.append(client_dist)
                if client_idx >= len(self.data_info):
                    self.data_info.append({'distribution': client_dist.copy()})
                else:
                    self.data_info[client_idx]['distribution'] = client_dist
            self.save_data_info()
            self.build_peer_communities(client_distributions)
        elif self.call_count == 3:
            # 测试集
            unique_labels = np.unique(label)
            num_classes = len(unique_labels)
            self.test_data_info = []
            for client_idx, client_indices in enumerate(idx_slice):
                client_indices = np.array(client_indices)
                client_labels = label[client_indices]
                client_dist = np.zeros(num_classes, dtype=int)
                for label_idx, label_class in enumerate(unique_labels):
                    mask = (client_labels == float(label_class))
                    client_dist[label_idx] = np.sum(mask)
                self.test_data_info.append(
                    {'distribution': client_dist.copy()})
            self.save_test_data_info()
            logger.info("Test set data distribution has been saved.")
        # 第二次调用为验证集，不做特殊处理

        if isinstance(dataset, Dataset):
            data_list = [Subset(dataset, idxs) for idxs in idx_slice]
        else:
            data_list = [[dataset[idx] for idx in idxs] for idxs in idx_slice]
        return data_list


def _split_according_to_prior(label, client_num, prior):
    assert client_num == len(prior)
    classes = len(np.unique(label))
    assert classes == len(np.unique(np.concatenate(prior, 0)))

    # counting
    frequency = np.zeros(shape=(client_num, classes))
    for idx, client_prior in enumerate(prior):
        for each in client_prior:
            frequency[idx][each] += 1
    sum_frequency = np.sum(frequency, axis=0)

    idx_slice = [[] for _ in range(client_num)]
    for k in range(classes):
        idx_k = np.where(label == k)[0]
        np.random.shuffle(idx_k)
        nums_k = np.ceil(frequency[:, k] / sum_frequency[k] *
                         len(idx_k)).astype(int)
        while len(idx_k) < np.sum(nums_k):
            random_client = np.random.choice(range(client_num))
            if nums_k[random_client] > 0:
                nums_k[random_client] -= 1
        assert len(idx_k) == np.sum(nums_k)
        idx_slice = [
            idx_j + idx.tolist() for idx_j, idx in zip(
                idx_slice, np.split(idx_k,
                                    np.cumsum(nums_k)[:-1]))
        ]

    for i in range(len(idx_slice)):
        np.random.shuffle(idx_slice[i])
    return idx_slice


def dirichlet_distribution_noniid_slice(label,
                                        client_num,
                                        alpha,
                                        min_size=1,
                                        prior=None):
    r"""Get sample index list for each client from the Dirichlet distribution.
    https://github.com/FedML-AI/FedML/blob/master/fedml_core/non_iid
    partition/noniid_partition.py

    Arguments:
        label (np.array): Label list to be split.
        client_num (int): Split label into client_num parts.
        alpha (float): alpha of LDA.
        min_size (int): min number of sample in each client
    Returns:
        idx_slice (List): List of splited label index slice.
    """
    if len(label.shape) != 1:
        raise ValueError('Only support single-label tasks!')

    if prior is not None:
        return _split_according_to_prior(label, client_num, prior)

    num = len(label)
    classes = len(np.unique(label))
    assert num > client_num * min_size, f'The number of sample should be ' \
                                        f'greater than' \
                                        f' {client_num * min_size}.'
    size = 0
    while size < min_size:
        idx_slice = [[] for _ in range(client_num)]
        for k in range(classes):
            # for label k
            idx_k = np.where(label == k)[0]
            np.random.shuffle(idx_k)
            prop = np.random.dirichlet(np.repeat(alpha, client_num))
            # prop = np.random.dirichlet(np.repeat(10, client_num))
            # prop = np.array([
            #    p * (len(idx_j) < num / client_num)
            #    for p, idx_j in zip(prop, idx_slice)
            # ])
            # prop = prop / sum(prop)
            prop = (np.cumsum(prop) * len(idx_k)).astype(int)[:-1]
            idx_slice = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_slice, np.split(idx_k, prop))
            ]
            size = min([len(idx_j) for idx_j in idx_slice])
    for i in range(client_num):
        np.random.shuffle(idx_slice[i])
    return idx_slice


def call_my_splitter(splitter_type, client_num, **kwargs):
    if splitter_type == 'fedgs':
        splitter = FedGS_Splitter(client_num, **kwargs)
        return splitter


register_splitter('fedgs', call_my_splitter)
