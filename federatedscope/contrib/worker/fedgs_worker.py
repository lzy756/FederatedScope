import logging
import copy
import numpy as np
import torch
import random
import os
import json
from pulp import *  # For linear programming solver

from federatedscope.core.message import Message
from federatedscope.core.workers.server import Server
from federatedscope.core.workers.client import Client
from federatedscope.core.auxiliaries.utils import merge_dict_of_results
from federatedscope.register import register_worker

logger = logging.getLogger(__name__)


class FedGSServer(Server):
    """
    FedGS server implementation with key features:
    - Uses predefined client grouping
    - Implements intra-group and inter-group aggregation
    - Uses FedGSAggregator for model aggregation
    """
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 unseen_clients_id=None,
                 **kwargs):
        super().__init__(ID, state, config, data, model, client_num,
                         total_round_num, device, strategy, unseen_clients_id,
                         **kwargs)

        # FedGS specific parameters
        self.num_groups = config.fedgs.num_groups  # K: Number of peer communities
        self.num_ccs = config.fedgs.num_groups  # M: Number of colleague communities

        # FedSAK specific parameters
        self.agg_lmbda = config.aggregator.get("lambda_",
                                               1e-3)  # FedSAK的lambda参数
        self.share_patterns = config.fedsak.share_patterns  # 共享层配置
        # 初始化客户端选择样本ID列表（当前轮次的）
        self.sample_client_ids = []
        # 新增：记录上一轮参与训练的客户端ID列表
        self.prev_sample_client_ids = []

        # Data distribution related
        self.peer_communities = None  # Peer communities
        self.client_features = {}  # Client features
        self.pc_features = None  # Peer community feature matrix P
        self.target_distribution = None  # Target distribution P^tar
        self.num_classes = None  # Number of classes

        # CC related
        self.ccs = [[] for _ in range(self.num_ccs)]
        self.client_ccs = {}
        self.h_vector = None  # CC template vector

        # Group related
        self.groups = [[] for _ in range(self.num_groups)
                       ]  # Store clients in each group
        self.client_groups = {}  # Record which group each client belongs to

        # Performance tracking
        self.client_weights = {}  # Store client sampling weights
        self.client_scores = {}  # Store client performance scores

        # Grouping state
        self.is_grouped = False
        self.is_cc_formed = False

        # Store group models after intra-group aggregation
        self.group_models = {}

        # Load data and initialize
        if not self._initialize_communities():
            raise ValueError("Failed to initialize communities")

        # Initialize groups
        self._initialize_groups()

        # Load saved grouping information
        if not self._load_saved_groups():
            raise ValueError(
                "Cannot find or load grouping information file peer_communities.txt"
            )

    def _initialize_communities(self):
        """Initialize communities and related data"""
        if not self._load_and_process_communities():
            return False

        # Calculate feature matrix P and target distribution
        if not self._calculate_features():
            return False

        # Build CC
        if not self._form_client_communities():
            logger.warning(
                "CC construction failed, will use fallback strategy")

        return True

    def _load_and_process_communities(self):
        """Load PC information from file and process data distribution"""
        peer_communities_path = 'peer_communities.txt'
        data_info_path = 'data_info.txt'
        test_data_path = 'test_data_info.txt'

        # Load test set distribution first (this will be our target)
        if not os.path.exists(test_data_path):
            logger.warning(
                f"Test set distribution file {test_data_path} does not exist")
            return False

        try:
            with open(test_data_path, 'r', encoding='utf-8') as f:
                content = f.read()
                json_start = content.find('\nRaw Data (JSON format):\n') + len(
                    '\nRaw Data (JSON format):\n')
                json_data = json.loads(content[json_start:])

                # Calculate test set distribution
                test_dist = {}
                total_samples = 0

                # Sum up all samples across clients
                for client_dist in json_data['test_data_info']:
                    for label, count in enumerate(client_dist):
                        if str(label) not in test_dist:
                            test_dist[str(label)] = 0
                        test_dist[str(label)] += count
                        total_samples += count

                # Set number of classes
                self.num_classes = len(json_data['test_data_info'][0])

                # Normalize to get distribution
                if total_samples > 0:
                    for label in test_dist:
                        test_dist[label] /= total_samples

                # Store test distribution as target
                self.target_distribution = test_dist
                logger.info(
                    f"Using test set distribution as target: {test_dist}")

        except Exception as e:
            logger.error(f"Error loading test set distribution: {str(e)}")
            return False

        # Load training data distribution
        if not os.path.exists(data_info_path):
            logger.warning(
                f"Data distribution file {data_info_path} does not exist")
            return False

        try:
            with open(data_info_path, 'r', encoding='utf-8') as f:
                content = f.read()
                json_start = content.find('\nRaw Data (JSON format):\n') + len(
                    '\nRaw Data (JSON format):\n')
                json_data = json.loads(content[json_start:])

                # Update client data distribution information
                self.client_data_dist = {}
                for client_id, dist in enumerate(json_data['data_info']):
                    self.client_data_dist[str(client_id)] = {
                        str(label): count
                        for label, count in enumerate(dist)
                    }

                logger.info(
                    f"Loaded training data distribution for {len(self.client_data_dist)} clients"
                )

        except Exception as e:
            logger.error(f"Error loading training data distribution: {str(e)}")
            return False

        # Load PC information
        if not os.path.exists(peer_communities_path):
            logger.warning(
                f"PC information file {peer_communities_path} does not exist")
            return False

        try:
            with open(peer_communities_path, 'r', encoding='utf-8') as f:
                content = f.read()
                json_start = content.find('\nRaw Data (JSON format):\n') + len(
                    '\nRaw Data (JSON format):\n')
                json_data = json.loads(content[json_start:])
                self.peer_communities = json_data['peer_communities']

                logger.info(
                    f"Loaded {len(self.peer_communities)} Peer Communities")

                # Build CC
                self._form_client_communities()
                return True
        except Exception as e:
            logger.error(f"Error loading PC information: {str(e)}")
            return False

    def _initialize_groups(self):
        """Initialize groups, assign clients to groups"""
        if not self.is_cc_formed:
            logger.warning(
                "CC not constructed, using simple round-robin grouping strategy"
            )
            # Use simple round-robin strategy to assign clients to groups
            all_clients = list(range(self.client_num))
            for i, client_id in enumerate(all_clients):
                group_id = i % self.num_groups
                self.groups[group_id].append(client_id)
                self.client_groups[client_id] = group_id
        else:
            # CC-based grouping strategy
            for cc_idx, cc_clients in enumerate(self.ccs):
                if not cc_clients:  # Skip empty CC
                    continue
                # Assign clients in CC to corresponding group
                group_id = cc_idx % self.num_groups
                for client_id in cc_clients:
                    self.groups[group_id].append(client_id)
                    self.client_groups[client_id] = group_id

        # Output grouping information
        logger.info("Client grouping completed:")
        for group_id, clients in enumerate(self.groups):
            logger.info(
                f"Group #{group_id + 1}: {len(clients)} clients - {clients}")

        self.is_grouped = True

    def _load_saved_groups(self):
        """Load saved grouping information and data distribution information"""
        peer_communities_path = 'peer_communities.txt'
        data_info_path = 'data_info.txt'

        # First load data distribution information
        if os.path.exists(data_info_path):
            try:
                with open(data_info_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Find JSON data section
                    json_start = content.find(
                        '\nRaw Data (JSON format):\n') + len(
                            '\nRaw Data (JSON format):\n')
                    json_data = json.loads(content[json_start:])

                    # Update client data distribution information
                    self.client_data_dist = {}
                    for client_id, dist in enumerate(json_data['data_info']):
                        self.client_data_dist[str(client_id)] = {
                            str(label): count
                            for label, count in enumerate(dist)
                        }

                    # Calculate global data distribution
                    self._calculate_global_distribution()

            except Exception as e:
                logger.warning(
                    f"Error loading data distribution information: {str(e)}")
                return False
        else:
            logger.warning(
                f"Data distribution information file {data_info_path} does not exist"
            )
            return False

        # Then load grouping information
        if os.path.exists(peer_communities_path):
            try:
                with open(peer_communities_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Find JSON data section
                    json_start = content.find(
                        '\nRaw Data (JSON format):\n') + len(
                            '\nRaw Data (JSON format):\n')
                    json_data = json.loads(content[json_start:])
                    communities = json_data['peer_communities']

                    # Update grouping information
                    self.groups = communities
                    self.client_groups = {}
                    for group_id, clients in enumerate(communities):
                        for client_id in clients:
                            self.client_groups[client_id] = group_id

                    self.is_grouped = True
                    logger.info(f"Loaded grouping information from file:")
                    for group_id, clients in enumerate(self.groups):
                        logger.info(
                            f"Community #{group_id + 1}: {len(clients)} clients - {clients}"
                        )

                    # Build CCs
                    self._form_client_communities()

                    return True
            except Exception as e:
                logger.warning(f"Error loading grouping information: {str(e)}")
                return False
        else:
            logger.warning(
                f"Grouping information file {peer_communities_path} does not exist"
            )
            return False

    def _calculate_global_distribution(self):
        """Calculate global data distribution"""
        if not self.client_data_dist:
            return

        # Initialize global distribution
        total_samples = 0
        self.global_data_dist = {}

        # Accumulate data from all clients
        for client_dist in self.client_data_dist.values():
            for label, count in client_dist.items():
                if label not in self.global_data_dist:
                    self.global_data_dist[label] = 0
                self.global_data_dist[label] += count
                total_samples += count

        # Convert to percentages
        for label in self.global_data_dist:
            self.global_data_dist[label] /= total_samples

        logger.info("Global data distribution calculation completed:")
        for label, percentage in sorted(self.global_data_dist.items()):
            logger.info(f"Class {label}: {percentage*100:.2f}%")

    def _solve_cc_composition(self):
        """Solve CC construction linear programming problem with sample size consideration"""
        if self.pc_features is None or self.target_distribution is None:
            logger.error("Missing required feature data")
            return None

        K = len(self.peer_communities)  # Number of peer communities
        B = np.array([len(pc) for pc in self.peer_communities])  # PC sizes

        logger.info(f"Starting to solve linear programming problem:")
        logger.info(f"Number of peer communities K: {K}")
        logger.info(f"PC sizes B: {B}")

        try:
            # 1. 随机预采样每个PC中的客户端
            L_rnd = 1  # 每个PC随机预采样的客户端数量
            random_selected = []
            remaining_clients = []

            # 记录每个客户端的样本量
            client_samples = {}
            for k, pc in enumerate(self.peer_communities):
                for client_id in pc:
                    if str(client_id) in self.client_data_dist:
                        # 计算该客户端的总样本量
                        client_samples[client_id] = sum(
                            self.client_data_dist[str(client_id)].values())
                    else:
                        # 如果找不到分布信息，使用默认值1
                        client_samples[client_id] = 1

            # 记录有效的PC索引
            valid_pc_indices = []
            for k, pc in enumerate(self.peer_communities):
                if len(pc) > L_rnd:
                    selected = random.sample(list(pc), L_rnd)
                    random_selected.extend(selected)
                    remaining = [c for c in pc if c not in selected]
                    remaining_clients.append(remaining)
                    if remaining:  # 只记录非空的PC
                        valid_pc_indices.append(k)
                else:
                    random_selected.extend(pc)
                    remaining_clients.append([])

            logger.info(
                f"Randomly pre-selected {len(random_selected)} clients")
            logger.info(
                f"Number of valid PCs after pre-selection: {len(valid_pc_indices)}"
            )

            if not valid_pc_indices:
                logger.warning("No valid PCs remaining after pre-selection")
                return None

            # 2. 计算随机预采样客户端的特征和样本量
            random_features = np.zeros(self.num_classes)
            random_samples = 0
            for client_id in random_selected:
                client_dist = self._get_distribution([client_id])
                client_feat = np.array([
                    client_dist.get(str(i), 0) for i in range(self.num_classes)
                ])
                random_features += client_feat
                random_samples += client_samples[client_id]

            # 3. 计算剩余客户端的特征矩阵和样本量
            remaining_features = []
            remaining_samples = []

            # 只处理非空的PC
            for k in valid_pc_indices:
                pc = remaining_clients[k]
                if pc:  # Double check
                    pc_dist = self._get_distribution(pc)
                    pc_feat = [
                        pc_dist.get(str(i), 0) for i in range(self.num_classes)
                    ]
                    remaining_features.append(pc_feat)
                    remaining_samples.append(sum(client_samples[c]
                                                 for c in pc))

            if not remaining_features:
                logger.warning("No remaining features to process")
                return None

            remaining_features = np.array(remaining_features)
            remaining_samples = np.array(remaining_samples)

            # 4. 计算目标分布
            target = self.target_distribution - (
                random_features /
                np.sum(random_features) if np.sum(random_features) > 0 else 0)
            target = np.maximum(target, 0)
            if np.sum(target) > 0:
                target = target / np.sum(target)

            # 5. Calculate binary decomposition for each valid PC
            R = len(valid_pc_indices)  # 使用有效PC的数量

            # Build new feature matrix Q and sample weights
            Q = np.zeros((R, self.num_classes))
            sample_weights = np.zeros(R)
            e_coefficients = []

            for idx, k in enumerate(valid_pc_indices):
                Q[idx] = remaining_features[idx]
                sample_weights[idx] = remaining_samples[idx]
                e_coefficients.append((k, 1))  # 使用原始PC索引

            # 6. Build linear programming problem
            prob = LpProblem("CC_Construction", LpMinimize)

            # Decision variables
            x = [LpVariable(f"x_{i}", 0, 1, cat='Binary') for i in range(R)]
            y = {}
            for i in range(R):
                for j in range(R):
                    y[i, j] = LpVariable(f"y_{i}_{j}", 0, 1, cat='Binary')
            z = [LpVariable(f"z_{k}", 0, 1, cat='Binary') for k in range(K)]

            # 计算样本量权重
            total_samples = np.sum(sample_weights)
            normalized_weights = sample_weights / total_samples if total_samples > 0 else np.ones(
                R) / R

            # Objective function with sample weights
            Q_Qt = Q @ Q.T
            P_tar_Qt = target @ Q.T
            epsilon = 0.01

            objective = (lpSum(normalized_weights[i] * Q_Qt[i, j] * y[i, j]
                               for i in range(R) for j in range(R)) -
                         2 * lpSum(normalized_weights[i] * P_tar_Qt[i] * x[i]
                                   for i in range(R)) -
                         epsilon * lpSum(z[k] for k in range(K)))
            prob += objective

            # Linearization constraints
            for i in range(R):
                for j in range(R):
                    prob += y[i, j] <= x[i]
                    prob += y[i, j] <= x[j]
                    prob += y[i, j] >= x[i] + x[j] - 1

            # 样本量约束
            min_samples = total_samples * 0.2  # 至少选择20%的样本
            max_samples = total_samples * 0.5  # 最多选择50%的样本
            prob += lpSum(sample_weights[i] * x[i]
                          for i in range(R)) >= min_samples
            prob += lpSum(sample_weights[i] * x[i]
                          for i in range(R)) <= max_samples

            # 每个PC的选择约束
            max_per_pc = 2
            for k in range(K):
                if k in valid_pc_indices:
                    idx = valid_pc_indices.index(k)
                    prob += x[idx] <= max_per_pc
                    prob += x[idx] >= z[k]

            # Solve
            prob.solve()

            if LpStatus[prob.status] == 'Optimal':
                h = np.zeros(K)
                for i in range(R):
                    k = e_coefficients[i][0]  # 获取原始PC索引
                    h[k] = value(x[i])

                # 加入预采样的客户端
                for k in range(K):
                    h[k] += L_rnd if len(
                        self.peer_communities[k]) > L_rnd else len(
                            self.peer_communities[k])

                self.h_vector = h
                logger.info(f"Linear programming solved successfully:")
                logger.info(f"Objective value: {value(prob.objective)}")
                logger.info(f"h vector: {self.h_vector}")

                # 输出选中客户端的样本量信息
                selected_samples = sum(sample_weights[i] * value(x[i])
                                       for i in range(R)) + random_samples
                logger.info(
                    f"Total selected samples: {selected_samples} ({selected_samples/total_samples*100:.2f}% of total)"
                )

                return self.h_vector
            else:
                logger.error(
                    f"Linear programming failed: {LpStatus[prob.status]}")
                return None

        except Exception as e:
            logger.error(f"Error solving linear programming: {str(e)}")
            return None

    def _form_client_communities(self):
        """Build CC based on solving results (每个CC都独立采样5个客户端，尽量不重叠)"""
        # Solve linear programming problem
        h = self._solve_cc_composition()
        if h is None:
            logger.error("Cannot build CC: Linear programming failed")
            return False

        self.h_vector = h
        logger.info(f"CC template vector h: {h}")

        # Reset CC信息
        self.ccs = [[] for _ in range(self.num_ccs)]
        self.client_ccs = {}

        # 统计每个PC的可用客户端
        pc_available_clients = [list(pc) for pc in self.peer_communities]
        # 记录已被选过的客户端，尽量避免重叠
        used_clients = set()

        for cc_idx in range(self.num_ccs):
            cc_clients = []
            for pc_idx, num in enumerate(h):
                num = int(round(num))
                if num == 0:
                    continue
                # 优先从未被选过的客户端中采样
                available = [
                    c for c in pc_available_clients[pc_idx]
                    if c not in used_clients
                ]
                if len(available) < num:
                    # 不够就允许重叠，从所有PC客户端中采样
                    available = list(pc_available_clients[pc_idx])
                if len(available) >= num:
                    selected = random.sample(available, num)
                else:
                    # 兜底：如果PC本身人数都不够num，全部选上
                    selected = list(available)
                cc_clients.extend(selected)
                used_clients.update(selected)
            self.ccs[cc_idx] = cc_clients
            for client_id in cc_clients:
                self.client_ccs[client_id] = cc_idx

        # 记录CC信息
        self.is_cc_formed = True
        logger.info(
            "Client Communities construction completed (each CC independently samples 5 clients):"
        )
        for cc_idx, clients in enumerate(self.ccs):
            cc_dist = self._get_distribution(clients)
            logger.info(
                f"CC #{cc_idx + 1}: {len(clients)} clients - {clients}")
            logger.info(f"CC #{cc_idx + 1} data distribution:")
            for label, percentage in sorted(cc_dist.items()):
                logger.info(f"  Class {label}: {percentage*100:.2f}%")
            # Calculate difference from target distribution
            cc_dist_array = np.array(
                [cc_dist.get(str(i), 0) for i in range(self.num_classes)])
            diff = np.linalg.norm(cc_dist_array - self.target_distribution)
            logger.info(
                f"  L2 difference from target distribution: {diff:.4f}")
        return True

    def _register_default_handlers(self):
        """Register message processing functions"""
        super()._register_default_handlers()
        self.register_handlers('model_para',
                               self.callback_funcs_for_model_para,
                               ['model_para', 'evaluate', 'finish'])
        self.register_handlers('join_in', self.callback_funcs_for_join_in)

    def callback_funcs_for_join_in(self, message: Message):
        """Handle client joining"""
        if message.sender not in self.comm_manager.neighbors:
            self.comm_manager.neighbors[message.sender] = message.content
        return super().callback_funcs_for_join_in(message)

    def _sample_clients(self, group_id):
        """Select clients based on sampling strategy
        
        Args:
            group_id: Group ID
            
        Returns:
            sampled_clients: List of sampled clients
        """
        group_clients = self.groups[group_id]
        if not group_clients:
            return []

        if self.sampler == 'uniform':  # Uniform sampling
            return group_clients  # Return all clients in the group

        elif self.sampler == 'gbp-cs':  # Gradient-Based Priority with Client Selection
            # Calculate weights for each client
            weights = []
            for client_id in group_clients:
                if client_id not in self.client_weights:
                    self.client_weights[client_id] = 1.0
                weights.append(self.client_weights[client_id])

            # Normalize weights
            weights = np.array(weights)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = np.ones_like(weights) / len(weights)

            # Sample clients based on weights, return all clients
            return group_clients

        else:  # Default return all clients
            return group_clients

    def _update_client_weights(self, round_idx, client_id, performance):
        """Update client sampling weights
        
        Args:
            round_idx: Current round
            client_id: Client ID
            performance: Client performance metric
        """
        if self.sampler != 'gbp-cs':
            return

        # Update client score
        if client_id not in self.client_scores:
            self.client_scores[client_id] = []
        self.client_scores[client_id].append(performance)

        # Calculate new weights
        if len(self.client_scores[client_id]) > 1:
            # Calculate performance improvement
            prev_score = self.client_scores[client_id][-2]
            curr_score = self.client_scores[client_id][-1]
            improvement = curr_score - prev_score

            # Update weights
            self.client_weights[client_id] = max(
                0.1,  # Minimum weight
                self.client_weights.get(client_id, 1.0) + improvement)

    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1,
                             filter_unseen_clients=True):
        """Broadcast model parameters to clients"""
        if self.state >= self._cfg.federate.total_round_num:
            self.terminate(msg_type='finish')
            return

        # If it's an evaluation request, send directly to all clients
        if msg_type == 'evaluate':
            receiver = list(self.comm_manager.neighbors.keys())
            for rcv_idx in receiver:
                # 如果有个性化切片，使用客户端的个性化模型进行评估
                content = None
                if hasattr(self, "personalized_slices"
                           ) and rcv_idx in self.personalized_slices:
                    content = self.personalized_slices[rcv_idx]
                else:
                    content = None

                self.comm_manager.send(
                    Message(msg_type=msg_type,
                            sender=self.ID,
                            receiver=[rcv_idx],
                            state=self.state,
                            content=content))
            return

            # 首先，如果存在个性化切片且有上一轮的客户端，先发送结果给上一轮的客户端
        if hasattr(self,
                   "personalized_slices") and self.prev_sample_client_ids:
            for cid in self.prev_sample_client_ids:
                if cid in self.personalized_slices:
                    logger.debug(f"Send personalized slice to C{cid}")
                    # 使用新消息类型update_model，只更新模型不触发训练
                    self.comm_manager.send(
                        Message(msg_type="update_model",
                                sender=self.ID,
                                receiver=[cid],
                                state=self.state,
                                content=self.personalized_slices[cid]))
            self.prev_sample_client_ids = []

        # 确保所有客户端状态正确设置为"空闲"
        all_client_ids = list(self.comm_manager.neighbors.keys())
        self.sampler.change_state(all_client_ids, 'idle')

        # 然后，采样新的客户端进行下一轮训练
        if filter_unseen_clients:
            # 适配父类方法的用法
            self.sampler.change_state(self.unseen_clients_id, 'unseen')

        # Generate CCs for this round
        self.current_ccs = self._generate_ccs_for_round(self.state)
        if not any(self.current_ccs):
            logger.warning(
                f"Failed to generate CCs for round {self.state}, skipping this round"
            )
            self.state += 1
            if self.state < self._cfg.federate.total_round_num:
                self.broadcast_model_para(msg_type='model_para')
            return

        # Broadcast model parameters to all clients in all CCs
        sampled_clients = []
        for cc_idx, cc in enumerate(self.current_ccs):
            if cc:
                sampled_clients.extend(cc)
                logger.info(f'CC #{cc_idx + 1}: {len(cc)} clients - {cc}')

        logger.info(
            f'Server: Broadcasting model parameters to {len(sampled_clients)} clients...'
        )
        self.sampler.change_state(sampled_clients, 'working')
        self.sample_client_ids = sampled_clients

        for client_id in sampled_clients:
            if self.state == 0 or not hasattr(self, "personalized_slices"):
                content = self.model.state_dict()
                self.comm_manager.send(
                    Message(msg_type="model_para",
                            sender=self.ID,
                            receiver=[client_id],
                            state=self.state,
                            content=content))
            else:
                content = None
                if client_id in self.personalized_slices:
                    content = self.personalized_slices[client_id]
                self.comm_manager.send(
                    Message(msg_type="trigger_train",
                            sender=self.ID,
                            receiver=[client_id],
                            state=self.state,
                            content=content))

        logger.info('Server: Finished model broadcasting')

    def _select_clients_for_cc(self, pc_clients, target_size):
        """Select clients from PC to form CC based on sample counts
        
        Args:
            pc_clients: List of clients in PC
            target_size: Number of clients to select
            
        Returns:
            selected_clients: List of selected clients
        """
        if not pc_clients:
            return []

        logger.info(f"Selecting {target_size} clients from PC")
        logger.info(f"Available clients: {pc_clients}")

        # Get target sample counts based on target distribution
        total_target_samples = sum(
            self._get_raw_distribution(pc_clients).values())
        target_samples = {
            str(i): total_target_samples * ratio
            for i, ratio in enumerate(self.target_distribution)
        }

        # Calculate per-client sample counts
        client_samples = {}
        for client_id in pc_clients:
            if str(client_id) in self.client_data_dist:
                client_samples[client_id] = self.client_data_dist[str(
                    client_id)]

        # Add random noise to client selection
        noise_scale = 0.1  # Scale of random noise
        client_scores = {}
        for client_id in pc_clients:
            if client_id in client_samples:
                # Calculate base score based on distribution similarity
                dist = client_samples[client_id]
                total = sum(dist.values())
                normalized_dist = {k: v / total for k, v in dist.items()}

                # Calculate difference from target
                diff = 0
                for label in range(self.num_classes):
                    target = self.target_distribution[label]
                    current = normalized_dist.get(str(label), 0)
                    diff += abs(current - target)

                # Add random noise
                noise = np.random.normal(0, noise_scale)
                client_scores[client_id] = -diff + noise

        # Sort clients by score and select top-k
        sorted_clients = sorted(client_scores.items(),
                                key=lambda x: x[1],
                                reverse=True)
        selected_clients = [
            client_id for client_id, _ in sorted_clients[:target_size]
        ]

        # Shuffle selected clients to add more randomness
        random.shuffle(selected_clients)

        logger.info(f"Selected clients: {selected_clients}")
        return selected_clients

    def _generate_ccs_for_round(self, round_idx):
        """Dynamically generate multiple CCs for current round
        每个CC都独立根据h向量采样5个客户端，允许重叠
        """
        # Set random seed based on round index to ensure different but reproducible results
        np.random.seed(round_idx)
        random.seed(round_idx)

        # Add small random perturbation to target distribution
        noise_scale = 0.05  # Scale of random noise
        perturbed_target = self.target_distribution + np.random.normal(
            0, noise_scale, size=self.num_classes)
        perturbed_target = np.clip(perturbed_target, 0.01,
                                   1.0)  # Ensure positive values
        perturbed_target = perturbed_target / perturbed_target.sum(
        )  # Normalize

        # Temporarily replace target distribution
        original_target = self.target_distribution
        self.target_distribution = perturbed_target

        # Solve CC composition to get h vector
        h = self._solve_cc_composition()

        # Restore original target distribution
        self.target_distribution = original_target

        if h is None:
            logger.error("Cannot generate CCs: Linear programming failed")
            return [[] for _ in range(self.num_ccs)]

        logger.info(f"Round {round_idx}: CC template vector h: {h}")
        logger.info(f"Perturbed target distribution: {perturbed_target}")

        # 每个CC都独立采样5个客户端
        ccs = [[] for _ in range(self.num_ccs)]
        pc_available_clients = [list(pc) for pc in self.peer_communities]
        for cc_idx in range(self.num_ccs):
            cc_clients = []
            for pc_idx, num in enumerate(h):
                num = int(round(num))
                if num == 0:
                    continue
                available = list(pc_available_clients[pc_idx])
                if len(available) >= num:
                    selected = random.sample(available, num)
                else:
                    selected = list(available)
                cc_clients.extend(selected)
            ccs[cc_idx] = cc_clients

        # 日志输出
        logger.info(
            f"Generated {self.num_ccs} CCs for round {round_idx} (each CC independently samples 5 clients):"
        )
        for cc_idx, cc in enumerate(ccs):
            logger.info(f"CC #{cc_idx + 1}: {len(cc)} clients - {cc}")
            if cc:
                cc_dist = self._get_distribution(cc)
                logger.info(f"CC #{cc_idx + 1} data distribution:")
                for label, percentage in sorted(cc_dist.items()):
                    logger.info(f"  Class {label}: {percentage*100:.2f}%")
                cc_dist_array = np.array(
                    [cc_dist.get(str(i), 0) for i in range(self.num_classes)])
                diff = np.linalg.norm(cc_dist_array - original_target)
                logger.info(
                    f"  L2 difference from target distribution: {diff:.4f}")
        return ccs

    def callback_funcs_for_model_para(self, message: Message):
        """Handle model parameters from clients"""
        round, sender, content = message.state, message.sender, message.content

        # Update state
        if round not in self.msg_buffer['train']:
            self.msg_buffer['train'][round] = dict()

        # Save client uploaded model
        self.msg_buffer['train'][round][sender] = content
        logger.info(
            f'Server: Received model parameters from client {sender} for round {round}'
        )

        # Check if received all responses from all CCs
        if not hasattr(self, 'current_ccs'):
            logger.warning("No current CCs found, skipping aggregation")
            return True

        # 统计每个CC的每个客户端都收到才聚合
        expected_total = 0
        received_total = 0
        for cc in self.current_ccs:
            for client_id in cc:
                expected_total += 1
                if client_id in self.msg_buffer['train'][round]:
                    received_total += 1

        if received_total < expected_total:
            # 还没收齐，等待
            return True

        # 收齐后聚合
        logger.info(
            'Server: All CCs have received sufficient responses, starting aggregation'
        )

        # First step: Intra-CC aggregation
        cc_models = {}
        for cc_idx, cc in enumerate(self.current_ccs):
            if not cc:  # Skip empty CCs
                continue
            # Prepare model for intra-CC clients
            cc_feedback = []
            for client_id in cc:
                if client_id in self.msg_buffer['train'][round]:
                    content = self.msg_buffer['train'][round][client_id]
                    if isinstance(content, tuple) and len(content) == 2:
                        sample_size, model_para = content
                    else:
                        sample_size = 1
                        model_para = content
                    client_info = {
                        'client_id': client_id,
                        'sample_size': sample_size,
                        'model_para': model_para
                    }
                    cc_feedback.append(client_info)
            if cc_feedback:
                logger.info(
                    f'Server: Performing intra-CC aggregation for CC {cc_idx} with {len(cc_feedback)} clients'
                )
                result = self.aggregator.aggregate_intra_group(cc_feedback)
                cc_models[cc_idx] = {
                    'model': result,
                    'size': sum(client['sample_size']
                                for client in cc_feedback)
                }

        # Second step: Inter-CC aggregation using FedSAK
        if cc_models:
            logger.info(
                f'Server: Performing inter-CC aggregation with {len(cc_models)} CCs using FedSAK'
            )
            cc_feedback = []
            for cc_idx, cc_info in cc_models.items():
                cc_feedback.append({
                    'client_id': f'cc_{cc_idx}',
                    'sample_size': cc_info['size'],
                    'model_para': cc_info['model']
                })

            # 使用FedSAK聚合方法进行组间聚合
            result = self.aggregator.aggregate_inter_group(cc_feedback)

            # 保存聚合结果作为个性化切片，下一轮使用
            if "model_para_all" in result:
                self.personalized_slices = result["model_para_all"]

            # 保存当前客户端列表为上一轮客户端列表，以便下次发送结果
            self.prev_sample_client_ids = self.sample_client_ids.copy()

            if self.agg_lmbda == 0:
                receiver = list(self.comm_manager.neighbors.keys())
                avg_model = list(result['model_para_all'].values())[0]
                for rcv_idx in receiver:
                    self.personalized_slices[rcv_idx] = avg_model
                self.prev_sample_client_ids = receiver.copy()
                print("avg_receiver:", self.prev_sample_client_ids)

            # 清理缓存
            self.msg_buffer['train'][round].clear()
            self.state += 1

            # 检查是否需要执行评估
            if (self.state % self._cfg.eval.freq == 0
                    and self.state != self._cfg.federate.total_round_num):
                logger.info(
                    f'Server: Starting evaluation at round {self.state}')
                self.eval()

            # 检查是否继续训练
            if self.state < self._cfg.federate.total_round_num:
                logger.info('----------- Starting training ' +
                            f'(Round #{self.state}) -------------')
                self.broadcast_model_para(msg_type='model_para')
            else:
                logger.info(
                    'Server: Training is finished! Starting final evaluation.')
                self.eval()

        return True

    def eval(self):
        """Perform evaluation on server side
        
        When cfg.federate.make_global_eval=True, server will perform global evaluation.
        Otherwise, evaluation task will be distributed to clients to execute.
        """
        if self._cfg.federate.make_global_eval:
            # Default, evaluate each internal model one by one
            for i in range(self.model_num):
                trainer = self.trainers[i]
                # Execute evaluation on server side
                metrics = {}
                for split in self._cfg.eval.split:
                    eval_metrics = trainer.evaluate(
                        target_data_split_name=split)
                    metrics.update(**eval_metrics)

                # Format evaluation results
                formatted_eval_res = self._monitor.format_eval_res(
                    metrics,
                    rnd=self.state,
                    role='Server #',
                    forms=self._cfg.eval.report,
                    return_raw=self._cfg.federate.make_global_eval)

                # Update best results
                self._monitor.update_best_result(
                    self.best_results,
                    formatted_eval_res['Results_raw'],
                    results_type="server_global_eval")

                # Merge historical results
                self.history_results = merge_dict_of_results(
                    self.history_results, formatted_eval_res)

                # Save formatted results
                self._monitor.save_formatted_results(formatted_eval_res)
                logger.info(formatted_eval_res)

            # Check and save model
            self.check_and_save()
        else:
            # Execute evaluation on clients
            logger.info('Server: Broadcasting model for client evaluation')
            self.broadcast_model_para(msg_type='evaluate',
                                      filter_unseen_clients=False)

    def _calculate_features(self):
        """Calculate feature matrix P and target distribution"""
        if not self.peer_communities or not hasattr(
                self, 'num_classes') or self.num_classes is None:
            logger.error("Missing PC information or number of classes")
            return False

        # Calculate average sample counts per client for each PC
        self.pc_features = np.zeros(
            (len(self.peer_communities), self.num_classes))

        # Calculate average sample counts per client for each PC
        for k, pc in enumerate(self.peer_communities):
            if not pc:  # Skip empty PC
                continue

            pc_dist = self._get_raw_distribution(pc)  # Get raw sample counts
            num_clients = len(pc)  # Number of clients in this PC

            # Calculate average samples per client
            for i in range(self.num_classes):
                self.pc_features[k][i] = pc_dist.get(str(i), 0) / num_clients

        # Convert target distribution from dict to array
        target_array = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            target_array[i] = self.target_distribution.get(str(i), 0.0)
        self.target_distribution = target_array

        logger.info("Feature calculation completed:")
        logger.info(f"Feature matrix P shape: {self.pc_features.shape}")
        logger.info("Average sample counts per client for each PC:")
        for k in range(len(self.peer_communities)):
            if self.peer_communities[k]:  # Only show non-empty PCs
                logger.info(
                    f"PC #{k + 1} ({len(self.peer_communities[k])} clients): {self.pc_features[k]}"
                )
        logger.info(f"Target distribution: {self.target_distribution}")
        return True

    def _get_raw_distribution(self, client_ids):
        """Calculate raw sample counts for given client set
        
        Args:
            client_ids: List of client IDs
            
        Returns:
            distribution: Data distribution dictionary, key is class, value is raw sample count
        """
        if not client_ids:
            return {}

        distribution = {}

        # Accumulate raw sample counts from all clients
        for client_id in client_ids:
            if str(client_id) not in self.client_data_dist:
                continue

            client_dist = self.client_data_dist[str(client_id)]
            for label, count in client_dist.items():
                if label not in distribution:
                    distribution[label] = 0
                distribution[label] += count

        return distribution

    def _get_distribution(self, client_ids):
        """Calculate normalized distribution for given client set
        
        Args:
            client_ids: List of client IDs
            
        Returns:
            distribution: Data distribution dictionary, key is class, value is normalized ratio
        """
        if not client_ids:
            return {}

        # Get raw counts first
        raw_dist = self._get_raw_distribution(client_ids)
        total_samples = sum(raw_dist.values())

        # Convert to normalized distribution
        distribution = {}
        if total_samples > 0:
            for label, count in raw_dist.items():
                distribution[label] = count / total_samples

        return distribution


class FedGSClient(Client):
    """FedGS client implementation"""
    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cpu',
                 strategy=None,
                 is_unseen_client=False,
                 *args,
                 **kwargs):
        super().__init__(ID, server_id, state, config, data, model, device,
                         strategy, is_unseen_client, *args, **kwargs)

    def _register_default_handlers(self):
        """覆写默认消息处理函数注册"""
        # 继承父类的注册
        super()._register_default_handlers()

        # 覆盖model_para处理器
        self.register_handlers("model_para",
                               self.callback_funcs_for_model_para,
                               ['model_para', 'ss_model_para'])

        # 注册新的消息处理器
        self.register_handlers("update_model",
                               self.callback_funcs_for_update_model,
                               ['model_para', 'ss_model_para'])

        self.register_handlers("trigger_train",
                               self.callback_funcs_for_trigger_train,
                               ['model_para', 'ss_model_para'])

    def callback_funcs_for_model_para(self, message: Message):
        """Handle model parameters update message"""
        round, sender, content = message.state, message.sender, message.content

        cur = self.model.state_dict()
        cur.update(content)
        self.model.load_state_dict(cur, strict=False)

        self.state = round

        # Execute local training
        sample_size, model_para, results = self.trainer.train()

        # Output training results
        formatted_results = self._monitor.format_eval_res(
            results,
            rnd=self.state,
            role='Client #{}'.format(self.ID),
            return_raw=True)
        logger.info(formatted_results)
        model_para = self.trainer.get_model_para()

        # Send training results to server
        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=(sample_size, model_para)))

    def callback_funcs_for_update_model(self, message: Message):
        """处理来自服务器的模型更新消息（只更新模型，不触发训练）"""
        round, sender, payload = message.state, message.sender, message.content

        if payload is not None:
            # 更新共享层
            cur = self.model.state_dict()
            cur.update(payload)
            self.model.load_state_dict(cur, strict=False)

        # 更新轮次状态
        self.state = round

        # 不触发训练
        logger.debug(f"Client #{self.ID}: Model updated without training")

    def callback_funcs_for_trigger_train(self, message: Message):
        """处理来自服务器的训练触发消息"""
        round, sender, payload = message.state, message.sender, message.content

        # 如果服务器发送了模型参数，则更新模型
        if payload is not None:
            cur = self.model.state_dict()
            cur.update(payload)
            self.model.load_state_dict(cur, strict=False)

        # 更新轮次状态
        self.state = round

        # 执行本地训练
        sample_size, _, results = self.trainer.train()

        # 记录训练结果到日志
        logger.info(
            self._monitor.format_eval_res(results,
                                          rnd=self.state,
                                          role='Client #{}'.format(self.ID),
                                          return_raw=True))

        # 获取模型参数并发送
        model_para = self.trainer.get_model_para()
        # print(model_para.keys())

        self.comm_manager.send(
            Message(msg_type="model_para",
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=(sample_size, model_para)))


# Register worker according to framework requirements
def call_fedgs_worker(worker_type):
    if worker_type == 'fedgs':
        worker_builder = {'client': FedGSClient, 'server': FedGSServer}
        return worker_builder
    return None


register_worker('fedgs', call_fedgs_worker)
