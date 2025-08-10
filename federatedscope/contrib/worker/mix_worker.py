import logging
import copy
import numpy as np
import random
import math
from pulp import *  # For linear programming solver
from federatedscope.core.auxiliaries.utils import (
    merge_dict_of_results,
    calculate_time_cost,
)

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
from federatedscope.register import register_worker
from federatedscope.contrib.data.utils import load_cluster_results

logger = logging.getLogger(__name__)


MODEL_UPDATE_TYPE = "model_update_type"
MODEL_PARA_TYPE = "model_param_type"

from collections import OrderedDict


class MIXServer(Server):
    """
    MIX server implementation with key features:
    - Uses predefined client grouping
    - Implements intra-group and inter-group aggregation
    - Uses MIXAggregator for model aggregation
    """

    def __init__(
        self,
        ID=-1,
        state=0,
        config=None,
        data=None,
        model=None,
        client_num=5,
        total_round_num=10,
        device="cpu",
        strategy=None,
        unseen_clients_id=None,
        **kwargs,
    ):
        super().__init__(
            ID,
            state,
            config,
            data,
            model,
            client_num,
            total_round_num,
            device,
            strategy,
            unseen_clients_id,
            **kwargs,
        )

        # FedSAK specific parameters
        self.agg_lmbda = config.aggregator.get("lambda_", 1e-3)  # FedSAK的lambda参数
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
        self.S = config.federate.get("sample_client_num", 25)
        # Total clients to select per round
        self.n_batch_size = config.dataloader.get("batch_size", 32)
        # Batch size per client
        self.A_matrix = None  # Feature matrix A
        self.M_vector = None  # Target vector M
        self.B_vector = None  # Upper bounds for each cluster
        self.template_solved = False  # Flag to indicate if template is solved

        self._load_and_process_communities()

        # Adjust S if necessary: if S < K, set S = K + 5
        K = len(self.peer_communities) if self.peer_communities else 0
        if self.S < K:
            old_S = self.S
            self.S = K
            logger.info(f"Adjusted S from {old_S} to {self.S} (K={K}, S must be >= K)")

        # compute the template vector h
        if not self._initialize_communities():
            raise ValueError("Failed to initialize communities")

    def _initialize_communities(self):
        """Initialize communities and related data"""

        # Calculate feature matrix P and target distribution
        if not self._calculate_features():
            return False

        # Solve template once at initialization
        if not self._solve_template_once():
            logger.warning("Template solving failed, will use fallback strategy")
            return False

        return True

    def _load_and_process_communities(self):
        """Load PC information from file and process data distribution"""
        # peer_communities_path = "peer_communities.txt"
        # data_info_path = "data_info.txt"
        # test_data_path = "test_data_info.txt"
        data_root = self._cfg.data.root
        train_dists, test_dists, self.peer_communities = load_cluster_results(
            f"{data_root}/peer_communities.json"
        )
        # Load test set distribution first (this will be our target)

        test_dist = {}
        total_samples = 0

        # Sum up all samples across clients
        for client_dist in test_dists:
            for label, count in enumerate(client_dist):
                if str(label) not in test_dist:
                    test_dist[str(label)] = 0
                test_dist[str(label)] += count
                total_samples += count

        # Set number of classes
        self.num_classes = len(test_dists[0])

        # Normalize to get distribution
        if total_samples > 0:
            for label in test_dist:
                test_dist[label] /= total_samples

        # Store test distribution as target
        self.target_distribution = test_dist

        # Load training data distribution

        self.client_data_dist = {}
        # Client start from 1, so enumerate starts from 1
        for client_id, dist in enumerate(train_dists, start=1):
            self.client_data_dist[str(client_id)] = {
                str(label): count for label, count in enumerate(dist)
            }

        logger.info(f"Using test set distribution as target: {test_dist}")
        logger.info(
            f"Loaded training data distribution for {len(self.client_data_dist)} clients"
        )
        logger.info(f"Loaded {len(self.peer_communities)} peer communities")

        # 需要对peer_communities进行处理，每个用户的索引+1，因为客户端ID从1开始
        if self.peer_communities:
            for i, pc in enumerate(self.peer_communities):
                self.peer_communities[i] = [cid + 1 for cid in pc]

    def _register_default_handlers(self):
        """Register message processing functions"""
        super()._register_default_handlers()
        self.register_handlers(
            "model_para",
            self.callback_funcs_for_model_para,
            ["model_para", "evaluate", "finish"],
        )
        self.register_handlers("join_in", self.callback_funcs_for_join_in)

    def callback_funcs_for_join_in(self, message: Message):
        """Handle client joining"""
        if message.sender not in self.comm_manager.neighbors:
            self.comm_manager.neighbors[message.sender] = message.content
        return super().callback_funcs_for_join_in(message)

    def broadcast_model_para(
        self, msg_type="model_para", sample_client_num=-1, filter_unseen_clients=True
    ):
        """覆写广播模型参数方法"""
        # if self.state >= self._cfg.federate.total_round_num:
        #     self.terminate(msg_type="finish")
        #     return

        # 如果是评估请求，直接发送给所有客户端
        if msg_type == "evaluate":
            receiver = list(self.comm_manager.neighbors.keys())
            for rcv_idx in receiver:
                # 如果有个性化切片，使用客户端的个性化模型进行评估
                content = None
                if rcv_idx in self.personalized_slices:
                    content = self.personalized_slices[rcv_idx]
                self.comm_manager.send(
                    Message(
                        msg_type=msg_type,
                        sender=self.ID,
                        receiver=[rcv_idx],
                        state=self.state,
                        content=content,
                    )
                )
            return

        # 处理两种情况：
        # 1. 向上一轮参与的客户端发送更新后的个性化切片
        # 2. 采样新客户端启动新一轮训练

        # 首先，如果存在个性化切片且有上一轮的客户端，先发送结果给上一轮的客户端
        if self.prev_sample_client_ids:
            for cid in self.prev_sample_client_ids:
                if cid in self.personalized_slices:
                    logger.debug(f"Send personalized slice to C{cid}")
                    # 使用新消息类型update_model，只更新模型不触发训练
                    self.comm_manager.send(
                        Message(
                            msg_type="update_model",
                            sender=self.ID,
                            receiver=[cid],
                            state=self.state,
                            content=self.personalized_slices[cid],
                        )
                    )

            # 清空上一轮客户端列表(已经发送完成)
            self.prev_sample_client_ids = []

        # 获取需要广播的客户端
        if self.S > 0:
            self.selected_clients_per_cluster = self._select_clients_from_clusters(
                self.state
            )
            receiver = []
            for _, client_list in self.selected_clients_per_cluster.items():
                receiver.extend(client_list)
            # 保存当前轮次选择的客户端ID
            self.sample_client_ids = receiver
        else:
            # 广播给所有客户端
            receiver = list(self.comm_manager.neighbors.keys())
            if msg_type == "model_para":
                self.sample_client_ids = receiver

        # 广播开始训练的消息 - 不需要发送模型参数，客户端将使用自己当前的模型
        for rcv_idx in receiver:
            # 对于首轮训练或没有个性化切片的情况，需要发送初始模型
            if self.state == 0 or not hasattr(self, "personalized_slices"):
                content = self.model.state_dict()
                self.comm_manager.send(
                    Message(
                        msg_type="model_para",
                        sender=self.ID,
                        receiver=[rcv_idx],
                        state=self.state,
                        content=content,
                    )
                )
            else:
                # 如果客户端有个性化切片，使用trigger_train消息类型触发训练
                # 如果没有个性化切片(新加入的客户端)，发送空内容，客户端会使用现有模型
                content = None
                if rcv_idx in self.personalized_slices:
                    content = self.personalized_slices[rcv_idx]
                self.comm_manager.send(
                    Message(
                        msg_type="trigger_train",
                        sender=self.ID,
                        receiver=[rcv_idx],
                        state=self.state,
                        content=content,
                    )
                )

    def callback_funcs_for_model_para(self, message: Message):
        """Handle model parameters from clients"""
        round, sender, content = message.state, message.sender, message.content

        # Update state
        if round not in self.msg_buffer["train"]:
            self.msg_buffer["train"][round] = dict()

        # Save client uploaded model
        self.msg_buffer["train"][round][sender] = content
        # logger.info(
        #     f"Server: Received model parameters from client {sender} for round {round}"
        # )

        # Count expected and received responses from all selected clients
        if len(self.msg_buffer["train"][round]) < len(self.sample_client_ids):
            return False

        client_feedback = []
        for cid in self.sample_client_ids:
            client_info = {
                "client_id": cid,
                "model_para": self.msg_buffer["train"][round][cid][1],
            }
            # logger.info(
            #     f"model_para from client {cid}: {type(client_info['model_para'])}"
            # )
            client_feedback.append(client_info)

        # 使用聚合器进行聚合
        agg_info = {"client_feedback": client_feedback}
        # logger.info(f"aggregator: {self.aggregator.__class__.__name__}")
        result = self.aggregator.aggregate(agg_info)

        # 保存聚合结果作为个性化切片，下一轮使用
        if "model_para_all" in result:
            self.personalized_slices = result["model_para_all"]
            # logger.info(
            #     f"Server: Aggregated personalized slices for round {round}, "
            #     f"sampled clients: {self.sample_client_ids}"
            #     f", {self.personalized_slices.keys()} keys"
            #     f", {type(list(self.personalized_slices.values())[0])} values"
            # )

        # 保存当前客户端列表为上一轮客户端列表，以便下次发送结果
        self.prev_sample_client_ids = self.sample_client_ids.copy()

        # 清理缓存
        self.msg_buffer["train"][round].clear()
        self.state += 1

        # 检查是否需要执行评估
        if (
            self.state % self._cfg.eval.freq == 0
            and self.state != self._cfg.federate.total_round_num
        ):
            logger.info(f"Server: Starting evaluation at round {self.state}")
            self.eval()

        # 检查是否继续训练
        if self.state < self._cfg.federate.total_round_num:
            # 启动新一轮训练
            logger.info(
                "----------- Starting a new training round"
                + f" (Round #{self.state}) -------------"
            )
            self.broadcast_model_para(
                msg_type="model_para",
                sample_client_num=self._cfg.federate.sample_client_num,
            )
        else:
            # 最终评估
            logger.info("Server: Training is finished! Starting evaluation.")
            self.eval()

        return True

    def _calculate_features(self):
        """Calculate feature matrix P and target distribution"""
        if (
            not self.peer_communities
            or not hasattr(self, "num_classes")
            or self.num_classes is None
        ):
            logger.error("Missing PC information or number of classes")
            return False

        # Calculate average sample counts per client for each PC
        self.pc_features = np.zeros((len(self.peer_communities), self.num_classes))

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

        # 优化下面输出数字的格式
        for k in range(len(self.peer_communities)):
            if self.peer_communities[k]:  # Only show non-empty PCs
                logger.info(
                    f"PC #{k + 1} ({len(self.peer_communities[k])} clients): {self.pc_features[k]}"
                )
        logger.info(f"Target distribution: {self.target_distribution}")
        return True

    def _solve_template_once(self):
        """Solve template h vector once at initialization using Binary Split method"""
        if (
            not self.peer_communities
            or not hasattr(self, "num_classes")
            or self.num_classes is None
        ):
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

        if not self._validate_solution(h_solution):
            logger.error("Solution validation failed")
            return False

        # 在原有代码的模板求解完成后添加

        self.h_vector = h_solution
        self.template_solved = True

        self.evaluate_template_solution()

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
                # n_ck = total_samples / len(pc) if len(pc) > 0 else 0

                # Probability distribution for this cluster
                for f in range(F):
                    class_samples = pc_raw_dist.get(str(f), 0)
                    p_ck_f = class_samples / total_samples if total_samples > 0 else 0
                    # self.A_matrix[f, k] = n_ck * p_ck_f
                    self.A_matrix[f, k] = self.n_batch_size * p_ck_f
                    # 使用n_batch_size作为缩放因子

            # Build vector M: M = n*S * P^{tar}
            # S is the total number of clients to select per round
            total_clients_selected = self.S
            self.M_vector = (
                self.n_batch_size * total_clients_selected * self.target_distribution
            )

            logger.info(f"Matrix A shape: {self.A_matrix.shape}")
            logger.info(f"Vector M shape: {self.M_vector.shape}")
            logger.info(f"Upper bounds B: {self.B_vector}")
            logger.info(f"Total clusters K: {K}")
            logger.info(f"Total clients to select S: {self.S}")
            logger.info(f"Target total clients: {total_clients_selected}")
            logger.info(f"Additional clients beyond mandatory (S-K): {self.S - K}")

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
                G_k = math.floor(math.log2(B_k)) + 1 if B_k > 0 else 1
                G.append(G_k)
                total_binary_vars += G_k

            logger.info(
                f"Binary decomposition: G = {G}, total binary vars = {total_binary_vars}"
            )

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
            model.setParam("OutputFlag", 0)  # Suppress output

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

            # Constraint 1: sum(T*b) = S - K (additional clients beyond mandatory 1 per cluster)
            additional_clients = self.S - K
            constraint_expr = gp.LinExpr()
            for i in range(K):
                for j in range(total_binary_vars):
                    constraint_expr += T_matrix[i, j] * b_vars[j]
            model.addConstr(constraint_expr == additional_clients, "sum_constraint")

            # Constraint 2: Each cluster's selection doesn't exceed its size
            col_idx = 0
            for i in range(K):
                if self.peer_communities[i]:  # Only constrain non-empty clusters
                    cluster_expr = gp.LinExpr()
                    for j in range(G[i]):
                        cluster_expr += T_matrix[i, col_idx] * b_vars[col_idx]
                        col_idx += 1
                    # z_i = T*b for cluster i, h_i = z_i + 1
                    # Constraint: z_i <= B_i - 1
                    model.addConstr(
                        cluster_expr <= self.B_vector[i] - 1, f"cluster_{i}_max"
                    )
                else:
                    # Skip empty clusters (no variables)
                    col_idx += G[i]

            # Solve
            model.optimize()

            if model.status == GRB.OPTIMAL:
                # Extract solution
                b_solution = np.array([b_vars[j].x for j in range(total_binary_vars)])
                z_solution = T_matrix @ b_solution  # Additional clients per cluster

                # Add the mandatory 1 client per cluster
                h_solution = z_solution + 1

                # Set empty clusters to 0
                for i in range(K):
                    if not self.peer_communities[i]:
                        h_solution[i] = 0

                logger.info(
                    f"Gurobi solved successfully with objective value: {model.objVal}"
                )
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
            b_vars = [
                LpVariable(f"b_{j}", cat="Binary") for j in range(total_binary_vars)
            ]

            # Objective: minimize ||A*T*b - M||_2^2 (approximated as L1 norm)
            AT = self.A_matrix @ T_matrix

            # Create auxiliary variables for absolute values
            aux_vars = [
                LpVariable(f"aux_{f}", lowBound=0) for f in range(self.num_classes)
            ]

            # Objective: minimize sum of auxiliary variables
            prob += lpSum(aux_vars)

            # Constraints for auxiliary variables: aux_f >= |AT[f,:]*b - M[f]|
            for f in range(self.num_classes):
                expr = (
                    lpSum(AT[f, j] * b_vars[j] for j in range(total_binary_vars))
                    - self.M_vector[f]
                )
                prob += aux_vars[f] >= expr
                prob += aux_vars[f] >= -expr

            # Constraint 1: sum(T*b) = S - K (additional clients beyond mandatory 1 per cluster)
            additional_clients = self.S - K
            constraint_expr = lpSum(
                T_matrix[i, j] * b_vars[j]
                for i in range(K)
                for j in range(total_binary_vars)
            )
            prob += constraint_expr == additional_clients

            # Constraint 2: Each cluster's selection doesn't exceed its size
            col_idx = 0
            for i in range(K):
                if self.peer_communities[i]:  # Only constrain non-empty clusters
                    cluster_expr = lpSum(
                        T_matrix[i, col_idx + j] * b_vars[col_idx + j]
                        for j in range(G[i])
                    )
                    # z_i = T*b for cluster i, h_i = z_i + 1
                    # Constraint: z_i <= B_i - 1
                    prob += cluster_expr <= self.B_vector[i] - 1
                    col_idx += G[i]
                else:
                    # Skip empty clusters
                    col_idx += G[i]

            # Solve
            prob.solve()

            if LpStatus[prob.status] == "Optimal":
                # Extract solution
                b_solution = np.array(
                    [value(b_vars[j]) for j in range(total_binary_vars)]
                )
                z_solution = T_matrix @ b_solution

                # Add the mandatory 1 client per cluster
                h_solution = z_solution + 1

                # Set empty clusters to 0
                for i in range(K):
                    if not self.peer_communities[i]:
                        h_solution[i] = 0

                logger.info(
                    f"PuLP solved successfully with objective value: {value(prob.objective)}"
                )
                return h_solution
            else:
                logger.error(
                    f"PuLP optimization failed with status: {LpStatus[prob.status]}"
                )
                return None

        except Exception as e:
            logger.error(f"Error in PuLP solving: {str(e)}")
            return None

    def _validate_solution(self, h_solution):
        for k in range(len(self.peer_communities)):
            if self.peer_communities[k]:
                if h_solution[k] < 1:
                    return False
                if h_solution[k] > self.B_vector[k]:
                    return False
        return True

    def evaluate_template_solution(self):
        """
        评估模板解决方案与目标之间的差异

        Returns:
            dict: 包含多种差值指标的字典
        """
        if not hasattr(self, "h_vector") or self.h_vector is None:
            logger.error("Solution not calculated yet")
            return

        try:
            # 计算预测的特征向量
            M_hat = self.A_matrix @ self.h_vector

            # 计算关键指标
            diff_absolute = M_hat - self.M_vector
            diff_relative = np.divide(
                diff_absolute,
                self.M_vector,
                out=np.zeros_like(diff_absolute),
                where=self.M_vector != 0,
            )
            max_abs_diff = np.max(np.abs(diff_absolute))
            max_rel_diff = np.max(np.abs(diff_relative))
            l2_norm = np.linalg.norm(diff_absolute)
            l1_norm = np.sum(np.abs(diff_absolute))

            # 计算KL散度（作为概率分布比较）
            epsilon = 1e-10  # 避免log(0)
            target_dist_normalized = self.target_distribution / np.sum(
                self.target_distribution
            )
            actual_dist = M_hat / np.sum(M_hat)
            kl_divergence = np.sum(
                target_dist_normalized
                * np.log(target_dist_normalized / (actual_dist + epsilon) + epsilon)
            )

            # 计算相对特征匹配分数
            matched_features = np.sum(np.abs(diff_relative) < 0.05)  # 5%以内误差
            feature_score = matched_features / self.num_classes

            # 记录关键指标
            logger.info(
                f"Feature match score: {feature_score:.1%} ({matched_features}/{self.num_classes} features matched within 5% error)"
            )
            logger.info(f"Max absolute difference: {max_abs_diff:.4f}")
            logger.info(f"L2 norm of differences: {l2_norm:.4f}")
            logger.info(f"KL divergence: {kl_divergence:.4f}")
            logger.info(f"Predicted features: {M_hat}")
            logger.info(f"Target features: {self.M_vector}")
            logger.info(f"Absolute difference: {diff_absolute}")
            logger.info(f"Relative difference: {diff_relative}")

        except Exception as e:
            logger.error(f"Failed to evaluate solution: {str(e)}")
            return

    def _select_clients_from_clusters(self, round_idx):
        """Select clients from each cluster using pre-solved template h vector"""
        if not self.template_solved or self.h_vector is None:
            logger.error("Template not solved, cannot select clients")
            return {}

        # Set random seed for reproducible but different results per round
        seed = self._cfg.get("seed", 42) + round_idx
        np.random.seed(seed)
        random.seed(seed)

        logger.info(
            f"Round {round_idx}: Selecting clients from clusters using template h = {self.h_vector}"
        )

        # Prepare container for final selections
        selected_clients_per_cluster = {}
        total_selected = 0

        for k, pc in enumerate(self.peer_communities):
            # Get number of clients to select from this cluster
            num_to_select = int(round(self.h_vector[k]))

            if num_to_select <= 0:
                selected_clients_per_cluster[k] = []
                continue

            # Check if cluster has enough clients
            if len(pc) < num_to_select:
                logger.error(
                    f"Cluster {k} has only {len(pc)} clients, "
                    f"but {num_to_select} were requested"
                )
                raise ValueError(f"Not enough clients in cluster {k}")

            # Select the required number of clients
            if num_to_select == len(pc):
                selected = list(pc)  # Select all
            else:
                selected = random.sample(list(pc), num_to_select)

            selected_clients_per_cluster[k] = selected
            total_selected += num_to_select

        # Log the results
        logger.info(
            f"Selected clients from {len(self.peer_communities)} clusters for round {round_idx}:"
        )

        for k, clients in selected_clients_per_cluster.items():
            logger.info(f"Cluster #{k}: {len(clients)} clients - {clients}")

        # Verify total selection count
        expected_total = self.S
        logger.info(
            f"Total clients selected: {total_selected} (expected: {expected_total})"
        )

        if total_selected != expected_total:
            logger.error(
                f"Selection mismatch: selected {total_selected}, expected {expected_total}"
            )
            raise ValueError("Total selected clients does not match expected total")

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


class MIXClient(Client):
    def __init__(
        self,
        ID=-1,
        server_id=None,
        state=-1,
        config=None,
        data=None,
        model=None,
        device="cpu",
        strategy=None,
        is_unseen_client=False,
        *args,
        **kwargs,
    ):
        """修改构造函数以匹配Client父类的参数列表"""
        super().__init__(
            ID,
            server_id,
            state,
            config,
            data,
            model,
            device,
            strategy,
            is_unseen_client,
            *args,
            **kwargs,
        )

    def _register_default_handlers(self):
        """覆写默认消息处理函数注册"""
        # 继承父类的注册
        super()._register_default_handlers()

        # 覆盖model_para处理器
        self.register_handlers(
            "model_para",
            self.callback_funcs_for_model_para,
            ["model_para", "ss_model_para"],
        )

        # 注册新的消息处理器
        self.register_handlers(
            "update_model",
            self.callback_funcs_for_update_model,
            ["model_para", "ss_model_para"],
        )

        self.register_handlers(
            "trigger_train",
            self.callback_funcs_for_trigger_train,
            ["model_para", "ss_model_para"],
        )

    def callback_funcs_for_model_para(self, message: Message):
        """处理来自服务器的个性化模型切片"""
        round, sender, payload = message.state, message.sender, message.content

        # 1) 只更新共享层
        cur = self.model.state_dict()
        cur.update(payload)
        self.model.load_state_dict(cur, strict=False)

        # 更新轮次状态
        self.state = round

        # 2) 本地训练 - 使用标准接口，让自定义trainer处理共享参数
        sample_size, _, results = self.trainer.train()

        # 记录训练结果到日志
        logger.info(
            self._monitor.format_eval_res(
                results,
                rnd=self.state,
                role="Client #{}".format(self.ID),
                return_raw=True,
            )
        )

        # 3) 获取模型参数 - 自定义trainer的_param_filter会自动筛选共享层
        model_para = copy.deepcopy(self.trainer.get_model_para())

        # 4) 发送给服务器 - 使用标准通信接口
        self.comm_manager.send(
            Message(
                msg_type="model_para",
                sender=self.ID,
                receiver=[sender],
                state=self.state,
                content=(sample_size, model_para),
            )
        )

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
            self._monitor.format_eval_res(
                results,
                rnd=self.state,
                role="Client #{}".format(self.ID),
                return_raw=True,
            )
        )

        # 获取模型参数并发送
        model_para = copy.deepcopy(self.trainer.get_model_para())

        self.comm_manager.send(
            Message(
                msg_type="model_para",
                sender=self.ID,
                receiver=[sender],
                state=self.state,
                content=(sample_size, model_para),
            )
        )

    def callback_funcs_for_evaluate(self, message: Message):
        """
        The handling function for receiving the request of evaluating

        Arguments:
            message: The received message
        """
        sender, timestamp = message.sender, message.timestamp
        self.state = message.state
        if message.content is not None:
            self.trainer.update(
                message.content, strict=self._cfg.federate.share_local_model
            )
        if self.early_stopper.early_stopped and self._cfg.federate.method in [
            "local",
            "global",
        ]:
            metrics = list(self.best_results.values())[0]
        else:
            metrics = {}
            if self._cfg.finetune.before_eval:
                self.trainer.finetune()
            for split in self._cfg.eval.split:
                # TODO: The time cost of evaluation is not considered here
                eval_metrics = self.trainer.evaluate(target_data_split_name=split)

                if self._cfg.federate.mode == "distributed":
                    logger.info(
                        self._monitor.format_eval_res(
                            eval_metrics,
                            rnd=self.state,
                            role="Client #{}".format(self.ID),
                            return_raw=True,
                        )
                    )

                metrics.update(**eval_metrics)

            formatted_eval_res = self._monitor.format_eval_res(
                metrics,
                rnd=self.state,
                role="Client #{}".format(self.ID),
                forms=["raw"],
                return_raw=True,
            )

            logger.info(formatted_eval_res)

            self._monitor.update_best_result(
                self.best_results,
                formatted_eval_res["Results_raw"],
                results_type=f"client #{self.ID}",
            )
            self.history_results = merge_dict_of_results(
                self.history_results, formatted_eval_res["Results_raw"]
            )
            self.early_stopper.track_and_check(
                self.history_results[self._cfg.eval.best_res_update_round_wise_key]
            )

        self.comm_manager.send(
            Message(
                msg_type="metrics",
                sender=self.ID,
                receiver=[sender],
                state=self.state,
                timestamp=timestamp,
                content=metrics,
            )
        )


# 按照框架要求注册worker
def call_mix_worker(worker_type):
    if worker_type == "mix":
        worker_builder = {"client": MIXClient, "server": MIXServer}
        return worker_builder
    return None


register_worker("mix", call_mix_worker)
