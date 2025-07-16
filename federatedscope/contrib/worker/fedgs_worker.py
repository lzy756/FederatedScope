import logging
import copy
import numpy as np
import torch
import random
import os
import json
import math
from pulp import *  # For linear programming solver
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    print("Gurobi not available, will use PuLP as fallback")

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
        self.num_groups = config.fedgs.num_groups  # K: Number of peer communities (clusters)

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

        # Cluster-based selection (removed CC concept)
        self.selected_clients_per_cluster = {}  # {cluster_idx: [client_list]}
        self.h_vector = None  # Template vector for cluster-based selection (solved once at initialization)

        # Template solving parameters
        self.L = config.fedgs.get("L", 5)  # Additional clients to select per round
        self.n_batch_size = config.dataloader.get("batch_size", 32)  # Batch size per client
        self.A_matrix = None  # Feature matrix A
        self.M_vector = None  # Target vector M
        self.B_vector = None  # Upper bounds for each cluster
        self.template_solved = False  # Flag to indicate if template is solved

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

        # Solve template once at initialization
        if not self._solve_template_once():
            logger.warning(
                "Template solving failed, will use fallback strategy")
            return False

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

                # Template will be solved in _solve_template_once()
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

                    # Template will be solved in _solve_template_once()

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

        # Select clients from each cluster using pre-solved template
        self.selected_clients_per_cluster = self._select_clients_from_clusters(self.state)
        if not any(self.selected_clients_per_cluster.values()):
            logger.warning(
                f"Failed to select clients for round {self.state}, skipping this round"
            )
            self.state += 1
            if self.state < self._cfg.federate.total_round_num:
                self.broadcast_model_para(msg_type='model_para')
            return

        # Broadcast model parameters to all selected clients in all clusters
        sampled_clients = []
        for cluster_idx, cluster_clients in self.selected_clients_per_cluster.items():
            if cluster_clients:
                sampled_clients.extend(cluster_clients)
                logger.info(f'Cluster #{cluster_idx + 1}: {len(cluster_clients)} clients - {cluster_clients}')

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

        # Check if received all responses from all selected clients
        if not hasattr(self, 'selected_clients_per_cluster'):
            logger.warning("No selected clients found, skipping aggregation")
            return True

        # Count expected and received responses from all selected clients
        expected_total = 0
        received_total = 0
        for cluster_clients in self.selected_clients_per_cluster.values():
            for client_id in cluster_clients:
                expected_total += 1
                if client_id in self.msg_buffer['train'][round]:
                    received_total += 1

        if received_total < expected_total:
            # Still waiting for responses
            return True

        # All responses received, start two-layer aggregation
        logger.info(
            'Server: All selected clients have responded, starting two-layer aggregation'
        )

        # First step: Intra-cluster aggregation using FedAvg
        cluster_models = {}
        for cluster_idx, cluster_clients in self.selected_clients_per_cluster.items():
            if not cluster_clients:  # Skip empty clusters
                continue

            # Prepare model parameters for intra-cluster clients
            cluster_feedback = []
            for client_id in cluster_clients:
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
                    cluster_feedback.append(client_info)

            if cluster_feedback:
                logger.info(
                    f'Server: Performing intra-cluster FedAvg aggregation for Cluster {cluster_idx} with {len(cluster_feedback)} clients'
                )
                # Use FedAvg for intra-cluster aggregation
                result = self.aggregator.aggregate_intra_group(cluster_feedback)
                cluster_models[cluster_idx] = {
                    'model': result,
                    'size': sum(client['sample_size'] for client in cluster_feedback)
                }

        # Second step: Inter-cluster aggregation using FedSAK
        if cluster_models:
            logger.info(
                f'Server: Performing inter-cluster FedSAK aggregation with {len(cluster_models)} clusters'
            )
            cluster_feedback = []
            for cluster_idx, cluster_info in cluster_models.items():
                cluster_feedback.append({
                    'client_id': f'cluster_{cluster_idx}',
                    'sample_size': cluster_info['size'],
                    'model_para': cluster_info['model']
                })

            # Use FedSAK aggregation method for inter-cluster aggregation
            result = self.aggregator.aggregate_inter_group(cluster_feedback)

            # 保存聚合结果作为个性化切片，下一轮使用
            if "model_para_all" in result:
                self.personalized_slices = result["model_para_all"]
            logger.info(f'type of personalized_slices: {type(self.personalized_slices)}')
            logger.info(f'personalized_slices length: {len(self.personalized_slices)}')
            # 输出personalized_slices字典id
            logger.info(f'personalized_slices keys: {self.personalized_slices.keys()}')
            # 现在和cluster数量相同，将相应的cluster的复制到对应选择

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

    def _solve_template_once(self):
        """Solve template h vector once at initialization using Binary Split method"""
        if not self.peer_communities or not hasattr(self, 'num_classes') or self.num_classes is None:
            logger.error("Missing PC information or number of classes")
            return False

        if self.template_solved:
            logger.info("Template already solved, skipping...")
            return True

        logger.info("Starting template solving using Binary Split method...")

        # Step 1: Build matrix A and vector M
        if not self._build_template_matrices():
            return False

        # Step 2: Solve using Binary Split
        h_solution = self._solve_binary_split()
        if h_solution is None:
            logger.error("Binary Split solving failed")
            return False

        self.h_vector = h_solution
        self.template_solved = True

        logger.info(f"Template solved successfully: h = {self.h_vector}")
        return True

    def _build_template_matrices(self):
        """Build matrix A and vector M for template solving"""
        try:
            K = len(self.peer_communities)  # Number of clusters
            F = self.num_classes  # Number of features (classes)

            # Build matrix A: A[f, k] = n^{C_k} * P^{C_k}[f]
            # where n^{C_k} is average samples per client in cluster k
            # and P^{C_k}[f] is the probability of class f in cluster k
            self.A_matrix = np.zeros((F, K))
            self.B_vector = np.zeros(K)  # Upper bounds (cluster sizes)

            for k, pc in enumerate(self.peer_communities):
                if not pc:  # Skip empty clusters
                    continue

                self.B_vector[k] = len(pc)  # Cluster size as upper bound

                # Calculate cluster distribution
                pc_raw_dist = self._get_raw_distribution(pc)
                total_samples = sum(pc_raw_dist.values()) if pc_raw_dist else 1

                # Average samples per client in this cluster
                n_ck = total_samples / len(pc) if len(pc) > 0 else 0

                # Probability distribution for this cluster
                for f in range(F):
                    class_samples = pc_raw_dist.get(str(f), 0)
                    p_ck_f = class_samples / total_samples if total_samples > 0 else 0
                    self.A_matrix[f, k] = n_ck * p_ck_f

            # Build vector M: M = n(L+K) * P^{tar}
            total_clients_selected = self.L + K  # L additional + K (at least one per cluster)
            self.M_vector = self.n_batch_size * total_clients_selected * self.target_distribution

            logger.info(f"Matrix A shape: {self.A_matrix.shape}")
            logger.info(f"Vector M shape: {self.M_vector.shape}")
            logger.info(f"Upper bounds B: {self.B_vector}")
            logger.info(f"Target total clients: {total_clients_selected}")

            return True

        except Exception as e:
            logger.error(f"Error building template matrices: {str(e)}")
            return False

    def _solve_binary_split(self):
        """Solve the constrained least squares problem using Binary Split method"""
        try:
            K = len(self.peer_communities)  # Number of variables

            # Step 1: Binary decomposition
            G = []  # Number of binary bits for each variable
            total_binary_vars = 0

            for k in range(K):
                B_k = int(self.B_vector[k])
                G_k = math.ceil(math.log2(B_k + 1)) if B_k > 0 else 1
                G.append(G_k)
                total_binary_vars += G_k

            logger.info(f"Binary decomposition: G = {G}, total binary vars = {total_binary_vars}")

            # Step 2: Build transformation matrix T
            T_matrix = self._build_transformation_matrix(G)

            # Step 3: Solve using optimization solver
            if GUROBI_AVAILABLE:
                return self._solve_with_gurobi(T_matrix, G)
            else:
                return self._solve_with_pulp(T_matrix, G)

        except Exception as e:
            logger.error(f"Error in Binary Split solving: {str(e)}")
            return None

    def _build_transformation_matrix(self, G):
        """Build transformation matrix T for binary split"""
        K = len(G)
        total_binary_vars = sum(G)
        T_matrix = np.zeros((K, total_binary_vars))

        col_idx = 0
        for i in range(K):
            for j in range(G[i]):
                T_matrix[i, col_idx] = 2**j
                col_idx += 1

        return T_matrix

    def _solve_with_gurobi(self, T_matrix, G):
        """Solve using Gurobi optimizer"""
        try:
            K = len(self.peer_communities)
            total_binary_vars = sum(G)

            # Create model
            model = gp.Model("BinarySplit")
            model.setParam('OutputFlag', 0)  # Suppress output

            # Create binary variables
            b_vars = model.addVars(total_binary_vars, vtype=GRB.BINARY, name="b")

            # Objective: minimize ||A*T*b - M||_2^2
            AT = self.A_matrix @ T_matrix  # Shape: (F, total_binary_vars)

            # Quadratic objective
            obj = gp.QuadExpr()
            for f in range(self.num_classes):
                expr = gp.LinExpr()
                for j in range(total_binary_vars):
                    expr += AT[f, j] * b_vars[j]
                expr -= self.M_vector[f]
                obj += expr * expr

            model.setObjective(obj, GRB.MINIMIZE)

            # Constraint: sum(T*b) = L
            constraint_expr = gp.LinExpr()
            for i in range(K):
                for j in range(total_binary_vars):
                    constraint_expr += T_matrix[i, j] * b_vars[j]
            model.addConstr(constraint_expr == self.L, "sum_constraint")

            # Solve
            model.optimize()

            if model.status == GRB.OPTIMAL:
                # Extract solution
                b_solution = np.array([b_vars[j].x for j in range(total_binary_vars)])
                h_solution = T_matrix @ b_solution

                # Add the mandatory 1 client per cluster
                h_solution = h_solution + 1

                logger.info(f"Gurobi solved successfully with objective value: {model.objVal}")
                return h_solution
            else:
                logger.error(f"Gurobi optimization failed with status: {model.status}")
                return None

        except Exception as e:
            logger.error(f"Error in Gurobi solving: {str(e)}")
            return None

    def _solve_with_pulp(self, T_matrix, G):
        """Solve using PuLP as fallback (approximation)"""
        try:
            K = len(self.peer_communities)
            total_binary_vars = sum(G)

            logger.info("Using PuLP solver (approximation method)")

            # Create problem
            prob = LpProblem("BinarySplit_Approx", LpMinimize)

            # Create binary variables
            b_vars = [LpVariable(f"b_{j}", cat='Binary') for j in range(total_binary_vars)]

            # Objective: minimize ||A*T*b - M||_2^2 (approximated as L1 norm)
            AT = self.A_matrix @ T_matrix

            # Create auxiliary variables for absolute values
            aux_vars = [LpVariable(f"aux_{f}", lowBound=0) for f in range(self.num_classes)]

            # Objective: minimize sum of auxiliary variables
            prob += lpSum(aux_vars)

            # Constraints for auxiliary variables: aux_f >= |AT[f,:]*b - M[f]|
            for f in range(self.num_classes):
                expr = lpSum(AT[f, j] * b_vars[j] for j in range(total_binary_vars)) - self.M_vector[f]
                prob += aux_vars[f] >= expr
                prob += aux_vars[f] >= -expr

            # Constraint: sum(T*b) = L
            constraint_expr = lpSum(T_matrix[i, j] * b_vars[j]
                                  for i in range(K) for j in range(total_binary_vars))
            prob += constraint_expr == self.L

            # Solve
            prob.solve()

            if LpStatus[prob.status] == 'Optimal':
                # Extract solution
                b_solution = np.array([value(b_vars[j]) for j in range(total_binary_vars)])
                h_solution = T_matrix @ b_solution

                # Add the mandatory 1 client per cluster
                h_solution = h_solution + 1

                logger.info(f"PuLP solved successfully with objective value: {value(prob.objective)}")
                return h_solution
            else:
                logger.error(f"PuLP optimization failed with status: {LpStatus[prob.status]}")
                return None

        except Exception as e:
            logger.error(f"Error in PuLP solving: {str(e)}")
            return None

    def _select_clients_from_clusters(self, round_idx):
        """Select clients from each cluster using pre-solved template h vector"""
        if not self.template_solved or self.h_vector is None:
            logger.error("Template not solved, cannot select clients")
            return {}

        # Set random seed for reproducible but different results per round
        np.random.seed(round_idx + 42)  # Add offset to avoid collision with other random seeds
        random.seed(round_idx + 42)

        logger.info(f"Round {round_idx}: Selecting clients from clusters using template h = {self.h_vector}")

        # Select clients from each cluster based on the template
        selected_clients_per_cluster = {}

        for k, pc in enumerate(self.peer_communities):
            if not pc:  # Skip empty clusters
                selected_clients_per_cluster[k] = []
                continue

            num_to_select = int(round(self.h_vector[k]))
            if num_to_select <= 0:
                selected_clients_per_cluster[k] = []
                continue

            # Ensure we don't select more than available
            num_to_select = min(num_to_select, len(pc))

            # Random selection from this cluster
            if num_to_select == len(pc):
                selected = list(pc)
            else:
                selected = random.sample(list(pc), num_to_select)

            selected_clients_per_cluster[k] = selected

        # Log the results
        logger.info(f"Selected clients from {len(self.peer_communities)} clusters for round {round_idx}:")
        total_selected = 0
        all_selected_clients = []

        for k, clients in selected_clients_per_cluster.items():
            logger.info(f"Cluster #{k + 1}: {len(clients)} clients - {clients}")
            total_selected += len(clients)
            all_selected_clients.extend(clients)

            if clients:  # Log distribution if cluster has selected clients
                cluster_dist = self._get_distribution(clients)
                logger.info(f"Cluster #{k + 1} data distribution:")
                for label, percentage in sorted(cluster_dist.items()):
                    logger.info(f"  Class {label}: {percentage*100:.2f}%")

        # Calculate overall distribution and L2 difference from target
        if all_selected_clients:
            overall_dist = self._get_distribution(all_selected_clients)
            logger.info(f"Overall selected clients data distribution:")
            for label, percentage in sorted(overall_dist.items()):
                logger.info(f"  Class {label}: {percentage*100:.2f}%")

            # Calculate L2 difference from target for overall distribution
            overall_dist_array = np.array([overall_dist.get(str(i), 0) for i in range(self.num_classes)])
            overall_diff = np.linalg.norm(overall_dist_array - self.target_distribution)
            logger.info(f"Overall L2 difference from target distribution: {overall_diff:.4f}")

        logger.info(f"Total clients selected across all clusters: {total_selected}")
        logger.info(f"Expected total (L + K): {self.L + len(self.peer_communities)}")

        return selected_clients_per_cluster

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
