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
    def __init__(
        self,
        client_num,
        alpha=0.5,
        distance_threshold=0.5,  # 距离阈值 ζ，由于使用归一化向量，默认值设为0.5
        min_groups=2,
        max_groups=10,
        balance_weight=2.0,  # 大小均衡的权重，越大越重视均衡
        balance_tolerance=0.2,
    ):  # 大小均衡的容忍度，允许的偏差比例
        self.alpha = alpha
        self.distance_threshold = distance_threshold
        self.min_groups = min_groups  # 最小社区数量 κ
        self.max_groups = max_groups  # 最大社区数量
        self.balance_weight = balance_weight  # 大小均衡权重
        self.balance_tolerance = balance_tolerance  # 大小均衡容忍度
        # 2D array storing training data distribution info for each client
        self.data_info = []
        self.test_data_info = []  # Store test set data distribution
        self.peer_communities = []  # List of peer communities
        self.is_initialized = (
            False  # Mark whether training data distribution has been recorded
        )
        self.call_count = 0  # 记录__call__被调用的次数
        super(FedGS_Splitter, self).__init__(client_num)

    def save_cluster_results(self, save_path="data/peer_communities.json"):
        """Save all clustering results to a single JSON file

        Args:
            save_path: Save path, defaults to 'data/peer_communities.json'
        """
        # Ensure directory exists
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True,
        )

        # Prepare train distributions
        train_distributions = []
        for dist_info in self.data_info:
            if isinstance(dist_info, dict) and "distribution" in dist_info:
                dist = dist_info["distribution"]
            else:
                dist = dist_info
            train_distributions.append(
                dist.tolist() if isinstance(dist, np.ndarray) else dist
            )

        # Prepare test distributions
        test_distributions = []
        for dist_info in self.test_data_info:
            if isinstance(dist_info, dict) and "distribution" in dist_info:
                dist = dist_info["distribution"]
            else:
                dist = dist_info
            test_distributions.append(
                dist.tolist() if isinstance(dist, np.ndarray) else dist
            )

        # Prepare peer communities (keep 0-based indexing internally)
        peer_communities = []
        for community in self.peer_communities:
            community_list = (
                community.tolist() if isinstance(community, np.ndarray) else community
            )
            peer_communities.append(community_list)

        # Prepare metadata
        metadata = {
            "num_clients": len(train_distributions),
            "num_classes": len(train_distributions[0]) if train_distributions else 0,
            "num_communities": len(peer_communities),
        }

        # Prepare complete data structure
        data = {
            "metadata": metadata,
            "train_distributions": train_distributions,
            "test_distributions": test_distributions,
            "peer_communities": peer_communities,
        }

        # Write to JSON file
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"All clustering results saved to: {save_path}")

    def load_cluster_results(self, load_path="peer_communities.json"):
        """Load all clustering results from a single JSON file

        Args:
            load_path: Load path, defaults to 'peer_communities.json'

        Returns:
            bool: True if loaded successfully, False otherwise
        """
        if not os.path.exists(load_path):
            logger.warning(f"Clustering results file {load_path} does not exist")
            return False

        try:
            with open(load_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Load train distributions
            self.data_info = []
            for dist in data["train_distributions"]:
                self.data_info.append({"distribution": np.array(dist)})

            # Load test distributions
            self.test_data_info = []
            for dist in data["test_distributions"]:
                self.test_data_info.append({"distribution": np.array(dist)})

            # Load peer communities
            self.peer_communities = []
            for community in data["peer_communities"]:
                self.peer_communities.append(np.array(community))

            logger.info(f"All clustering results loaded from: {load_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load clustering results from {load_path}: {e}")
            return False



    def _normalize_distributions(self, client_distributions):
        """Normalize client distribution vectors"""
        normalized_distributions = []
        for dist in client_distributions:
            total = np.sum(dist)
            if total > 0:
                normalized_dist = dist / total
            else:
                normalized_dist = dist
            normalized_distributions.append(normalized_dist)
        return np.array(normalized_distributions)

    def _compute_distance_matrix(self, normalized_distributions):
        """Compute pairwise Euclidean distance matrix between clients"""
        n_clients = len(normalized_distributions)
        distance_matrix = np.zeros((n_clients, n_clients))
        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                dist = np.linalg.norm(
                    normalized_distributions[i] - normalized_distributions[j]
                )
                distance_matrix[i][j] = dist
                distance_matrix[j][i] = dist
        return distance_matrix

    def _calculate_balance_score(self, cluster_labels, n_clients):
        """计算聚类结果的综合评分，考虑簇数量和大小均衡性"""
        num_clusters = len(np.unique(cluster_labels))

        # 如果簇数量不在合理范围内，返回很大的惩罚分数
        if num_clusters < self.min_groups or num_clusters > self.max_groups:
            return float("inf"), num_clusters, 0

        # 计算每个簇的大小
        cluster_sizes = []
        for label in np.unique(cluster_labels):
            cluster_size = np.sum(cluster_labels == label)
            cluster_sizes.append(cluster_size)

        # 计算目标大小和大小均衡性指标
        target_size = n_clients / num_clusters
        size_variance = np.var(cluster_sizes)  # 大小方差，越小越均衡

        # 综合评分：簇数量偏离度 + 大小不均衡惩罚
        ideal_num_clusters = (self.min_groups + self.max_groups) / 2
        cluster_num_penalty = abs(num_clusters - ideal_num_clusters)
        balance_penalty = size_variance / (target_size**2)  # 归一化方差

        # 综合评分：使用可配置的权重
        total_score = cluster_num_penalty + self.balance_weight * balance_penalty
        return total_score, num_clusters, size_variance

    def _search_optimal_clustering(self, distance_matrix, n_clients):
        """搜索最优的聚类阈值"""
        best_threshold = min(self.distance_threshold, 1.0)
        best_labels = None
        best_score = float("inf")

        # 创建阈值候选列表
        threshold_candidates = np.linspace(0.01, 1.0, 50)

        logger.info("Searching for optimal distance threshold with balance consideration...")

        # 检查sklearn版本兼容性
        from sklearn import __version__ as sklearn_version
        def parse_version(version_str):
            parts = []
            for part in version_str.split("."):
                num_part = "".join(filter(str.isdigit, part))
                if num_part:
                    parts.append(int(num_part))
            return tuple(parts)

        use_metric = parse_version(sklearn_version) >= (1, 4)

        for threshold in threshold_candidates:
            try:
                # 根据sklearn版本选择参数
                if not use_metric:
                    clustering = AgglomerativeClustering(
                        n_clusters=None,
                        distance_threshold=threshold,
                        affinity="precomputed",
                        linkage="complete",
                    )
                else:
                    clustering = AgglomerativeClustering(
                        n_clusters=None,
                        distance_threshold=threshold,
                        metric="precomputed",
                        linkage="complete",
                    )

                cluster_labels = clustering.fit_predict(distance_matrix)
                current_score, num_clusters, size_variance = self._calculate_balance_score(
                    cluster_labels, n_clients
                )

                if current_score < best_score:
                    best_score = current_score
                    best_threshold = threshold
                    best_labels = cluster_labels

                    logger.debug(
                        f"New best: threshold={threshold:.4f}, clusters={num_clusters}, "
                        f"score={current_score:.4f}, variance={size_variance:.2f}"
                    )

            except Exception as e:
                logger.info(f"Failed at threshold {threshold:.4f}: {e}")
                continue

        return best_threshold, best_labels, best_score

    def _force_clustering(self, distance_matrix, n_clients):
        """当无法找到合适阈值时，强制聚类到指定数量"""
        target_clusters = max(min(n_clients, self.max_groups), self.min_groups)

        # 检查sklearn版本兼容性
        from sklearn import __version__ as sklearn_version
        def parse_version(version_str):
            parts = []
            for part in version_str.split("."):
                num_part = "".join(filter(str.isdigit, part))
                if num_part:
                    parts.append(int(num_part))
            return tuple(parts)

        use_metric = parse_version(sklearn_version) >= (1, 4)

        if not use_metric:
            clustering = AgglomerativeClustering(
                n_clusters=target_clusters,
                affinity="precomputed",
                linkage="complete",
            )
        else:
            clustering = AgglomerativeClustering(
                n_clusters=target_clusters,
                metric="precomputed",
                linkage="complete"
            )

        labels = clustering.fit_predict(distance_matrix)
        logger.warning(f"Could not find suitable threshold, forcing {target_clusters} communities")
        return labels

    def _balance_communities(self, distance_matrix, n_clients):
        """平衡社区大小"""
        target_size = n_clients / len(self.peer_communities)
        tolerance = max(1, int(target_size * self.balance_tolerance))
        min_size = int(target_size - tolerance)
        max_size = int(target_size + tolerance)

        group_sizes = [len(pc) for pc in self.peer_communities]
        max_iterations = 10
        iteration = 0

        logger.info(f"Initial group sizes: {group_sizes}")
        logger.info(f"Target size: {target_size:.1f}, allowed range: [{min_size}, {max_size}]")

        while iteration < max_iterations:
            # 找到最大和最小的组
            max_idx = np.argmax(group_sizes)
            min_idx = np.argmin(group_sizes)

            # 如果大小已经在可接受范围内，停止调整
            if group_sizes[max_idx] <= max_size or group_sizes[min_idx] >= min_size:
                break

            # 从最大组移动一个客户端到最小组
            big_group = self.peer_communities[max_idx]
            small_group = self.peer_communities[min_idx]

            # 选择距离小组最近的客户端进行移动
            best_move_idx = -1
            best_move_score = float('inf')

            for i, client in enumerate(big_group):
                # 计算该客户端到小组的平均距离
                dist_to_small = np.mean([distance_matrix[client][j] for j in small_group]) if len(small_group) > 0 else 0
                # 计算该客户端到大组其他成员的平均距离
                dist_to_big = np.mean([distance_matrix[client][j] for j in big_group if j != client])

                # 移动评分：更倾向于移动到小组距离更近的客户端
                score = dist_to_small - dist_to_big
                if score < best_move_score:
                    best_move_score = score
                    best_move_idx = i

            # 执行移动
            if best_move_idx >= 0:
                client_to_move = big_group[best_move_idx]
                self.peer_communities[max_idx] = np.delete(big_group, best_move_idx)
                self.peer_communities[min_idx] = np.append(small_group, client_to_move)
                group_sizes[max_idx] -= 1
                group_sizes[min_idx] += 1
            else:
                break

            iteration += 1

        logger.info(f"Final group sizes after {iteration} iterations: {[len(pc) for pc in self.peer_communities]}")

    def build_peer_communities(self, client_distributions):
        """Build peer communities (PCs) using distance-based hierarchical clustering

        Args:
            client_distributions: List of client data distributions
        """
        normalized_distributions = self._normalize_distributions(client_distributions)
        distance_matrix = self._compute_distance_matrix(normalized_distributions)
        n_clients = len(normalized_distributions)

        best_threshold, best_labels, best_score = self._search_optimal_clustering(
            distance_matrix, n_clients
        )

        if best_labels is None:
            best_labels = self._force_clustering(distance_matrix, n_clients)

        # 构建对等社区
        unique_labels = np.unique(best_labels)
        self.peer_communities = []
        for label in unique_labels:
            community = np.where(best_labels == label)[0]
            self.peer_communities.append(community)

        # 平衡社区大小
        self._balance_communities(distance_matrix, n_clients)
        # 计算并保存每个社区的平均分布
        community_distributions = []
        for community in self.peer_communities:
            community_dist = np.zeros_like(normalized_distributions[0])
            total_samples = 0
            for client_idx in community:
                client_dist = self.data_info[client_idx]["distribution"]
                client_samples = np.sum(client_dist)
                community_dist += client_dist
                total_samples += client_samples
            if total_samples > 0:
                community_dist = community_dist / total_samples
            community_distributions.append(community_dist)
        logger.info(
            "\n========== Peer Communities Summary (Improved with Balance-Aware Clustering) =========="
        )
        logger.info(f"Number of Peer Communities (PCs): {len(self.peer_communities)}")
        logger.info(f"Distance threshold used: {best_threshold:.4f}")
        logger.info(f"Balance score achieved: {best_score:.4f}")

        # 计算大小均衡性统计
        group_sizes = [len(pc) for pc in self.peer_communities]
        size_mean = np.mean(group_sizes)
        size_std = np.std(group_sizes)
        logger.info(f"Group sizes: {group_sizes}")
        logger.info(f"Size statistics - Mean: {size_mean:.2f}, Std: {size_std:.2f}")

        for idx, (pc, pc_dist) in enumerate(
            zip(self.peer_communities, community_distributions)
        ):
            logger.info(f"\nPeer Community {idx + 1}:")
            logger.info(f"Number of clients: {len(pc)}")
            # 显示时将客户端ID加1
            display_indices = [i + 1 for i in pc.tolist()]
            logger.info(f"Client indices: {display_indices}")
            logger.info(f"Normalized community distribution: {pc_dist.tolist()}")
            logger.info(
                f"Distribution percentages: {['%.2f%%' % (p*100) for p in pc_dist]}"
            )
            max_internal_dist = 0
            for i in pc:
                for j in pc:
                    if i < j:
                        max_internal_dist = max(
                            max_internal_dist, distance_matrix[i][j]
                        )
            logger.info(f"Maximum internal distance: {max_internal_dist:.4f}")

            logger.info("Detailed client distributions in this community:")
            for client_idx in pc:
                client_dist = self.data_info[client_idx]["distribution"]
                client_total = np.sum(client_dist)
                client_percentages = (client_dist / client_total * 100).tolist()
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
        idx_slice = dirichlet_distribution_noniid_slice(
            label, self.client_num, self.alpha, prior=prior
        )

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
                    mask = client_labels == float(label_class)
                    client_dist[label_idx] = np.sum(mask)
                client_distributions.append(client_dist)
                if client_idx >= len(self.data_info):
                    self.data_info.append({"distribution": client_dist.copy()})
                else:
                    self.data_info[client_idx]["distribution"] = client_dist
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
                    mask = client_labels == float(label_class)
                    client_dist[label_idx] = np.sum(mask)
                self.test_data_info.append({"distribution": client_dist.copy()})
            # 保存所有聚类结果到单个JSON文件
            self.save_cluster_results()
            logger.info("All clustering results have been saved to peer_communities.json.")
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
        nums_k = np.ceil(frequency[:, k] / sum_frequency[k] * len(idx_k)).astype(int)
        while len(idx_k) < np.sum(nums_k):
            random_client = np.random.choice(range(client_num))
            if nums_k[random_client] > 0:
                nums_k[random_client] -= 1
        assert len(idx_k) == np.sum(nums_k)
        idx_slice = [
            idx_j + idx.tolist()
            for idx_j, idx in zip(idx_slice, np.split(idx_k, np.cumsum(nums_k)[:-1]))
        ]

    for i in range(len(idx_slice)):
        np.random.shuffle(idx_slice[i])
    return idx_slice


def dirichlet_distribution_noniid_slice(
    label, client_num, alpha, min_size=1, prior=None
):
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
        raise ValueError("Only support single-label tasks!")

    if prior is not None:
        return _split_according_to_prior(label, client_num, prior)

    num = len(label)
    classes = len(np.unique(label))
    assert num > client_num * min_size, (
        f"The number of sample should be " f"greater than" f" {client_num * min_size}."
    )
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
    if splitter_type == "fedgs":
        splitter = FedGS_Splitter(client_num, **kwargs)
        return splitter


register_splitter("fedgs", call_my_splitter)