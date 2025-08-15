import logging
import copy
import numpy as np
import random
import math
from pulp import *  # For linear programming solver
from federatedscope.core.auxiliaries.utils import merge_dict_of_results

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


class FDSEServer(Server):
    """
    FDSE server implementation with key features:
    - Uses predefined client grouping
    - Implements intra-group and inter-group aggregation
    - Uses FDSEAggregator for model aggregation
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
        self.agg_lmbda = config.aggregator.get("lambda_", 1e-3)
        self.share_patterns = config.fedsak.share_patterns

        # Client selection tracking
        self.sample_client_ids = []
        self.prev_sample_client_ids = []

        # Data distribution related
        self.peer_communities = None
        self.client_features = {}
        self.pc_features = None
        self.target_distribution = None
        self.num_classes = None

        # Cluster-based selection
        self.selected_clients_per_cluster = {}
        self.h_vector = None

        # Template solving parameters
        self.S = config.federate.get("sample_client_num", 25)
        self.n_batch_size = config.dataloader.get("batch_size", 32)
        self.A_matrix = None
        self.M_vector = None
        self.B_vector = None
        self.template_solved = False

        self._load_and_process_communities()

        # Adjust S if necessary
        K = len(self.peer_communities) if self.peer_communities else 0
        if self.S < K:
            old_S = self.S
            self.S = K
            logger.info(f"Adjusted S from {old_S} to {self.S} (K={K}, S must be >= K)")

        # Initialize communities
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
        data_root = self._cfg.data.root
        train_dists, test_dists, self.peer_communities = load_cluster_results(
            f"{data_root}/peer_communities.json"
        )

        # Load test set distribution as target
        test_dist = {}
        total_samples = 0

        for client_dist in test_dists:
            for label, count in enumerate(client_dist):
                if str(label) not in test_dist:
                    test_dist[str(label)] = 0
                test_dist[str(label)] += count
                total_samples += count

        self.num_classes = len(test_dists[0])

        # Normalize to get distribution
        if total_samples > 0:
            for label in test_dist:
                test_dist[label] /= total_samples

        self.target_distribution = test_dist

        # Load training data distribution
        self.client_data_dist = {}
        for client_id, dist in enumerate(train_dists, start=1):
            self.client_data_dist[str(client_id)] = {
                str(label): count for label, count in enumerate(dist)
            }

        logger.info(f"Using test set distribution as target: {test_dist}")
        logger.info(
            f"Loaded training data distribution for {len(self.client_data_dist)} clients"
        )
        logger.info(f"Loaded {len(self.peer_communities)} peer communities")

        # Adjust client indices (from 0-based to 1-based)
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
        """Broadcast model parameters with optimized client selection"""
        # Handle evaluation requests
        if msg_type == "evaluate":
            receiver = list(self.comm_manager.neighbors.keys())
            for rcv_idx in receiver:
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

        # Send updates to previous round participants
        if self.prev_sample_client_ids:
            for cid in self.prev_sample_client_ids:
                if cid in self.personalized_slices:
                    logger.debug(f"Send personalized slice to C{cid}")
                    self.comm_manager.send(
                        Message(
                            msg_type="update_model",
                            sender=self.ID,
                            receiver=[cid],
                            state=self.state,
                            content=self.personalized_slices[cid],
                        )
                    )
            self.prev_sample_client_ids = []

        # Select clients for new round
        if self.S > 0:
            self.selected_clients_per_cluster = self._select_clients_from_clusters(
                self.state
            )
            receiver = []
            for _, client_list in self.selected_clients_per_cluster.items():
                receiver.extend(client_list)
            self.sample_client_ids = receiver
        else:
            receiver = list(self.comm_manager.neighbors.keys())
            if msg_type == "model_para":
                self.sample_client_ids = receiver

        # Broadcast training messages
        for rcv_idx in receiver:
            if self.state == 0 or not hasattr(self, "personalized_slices"):
                content = self.model.state_dict()
                msg_type_to_send = "model_para"
            else:
                content = None
                if rcv_idx in self.personalized_slices:
                    content = self.personalized_slices[rcv_idx]
                msg_type_to_send = "trigger_train"

            self.comm_manager.send(
                Message(
                    msg_type=msg_type_to_send,
                    sender=self.ID,
                    receiver=[rcv_idx],
                    state=self.state,
                    content=content,
                )
            )

    def callback_funcs_for_model_para(self, message: Message):
        """Handle model parameters from clients"""
        round, sender, content = message.state, message.sender, message.content

        # Update message buffer
        if round not in self.msg_buffer["train"]:
            self.msg_buffer["train"][round] = dict()

        self.msg_buffer["train"][round][sender] = content

        # Check if all expected clients have responded
        if len(self.msg_buffer["train"][round]) < len(self.sample_client_ids):
            return False

        # Collect client feedback
        client_feedback = []
        for cid in self.sample_client_ids:
            client_info = {
                "client_id": cid,
                "model_para": self.msg_buffer["train"][round][cid],
            }
            client_feedback.append(client_info)

        # Aggregate using the aggregator
        agg_info = {"client_feedback": client_feedback}
        result = self.aggregator.aggregate(agg_info)

        # Save personalized slices for next round
        if "model_para_all" in result:
            self.personalized_slices = result["model_para_all"]

        # Update client tracking
        self.prev_sample_client_ids = self.sample_client_ids.copy()

        # Clean up and advance state
        self.msg_buffer["train"][round].clear()
        self.state += 1

        # Handle evaluation and next round
        if (
            self.state % self._cfg.eval.freq == 0
            and self.state != self._cfg.federate.total_round_num
        ):
            logger.info(f"Server: Starting evaluation at round {self.state}")
            self.eval()

        if self.state < self._cfg.federate.total_round_num:
            logger.info(
                f"----------- Starting a new training round (Round #{self.state}) -------------"
            )
            self.broadcast_model_para(
                msg_type="model_para",
                sample_client_num=self._cfg.federate.sample_client_num,
            )
        else:
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

        for k, pc in enumerate(self.peer_communities):
            if not pc:  # Skip empty PC
                continue

            pc_dist = self._get_raw_distribution(pc)
            num_clients = len(pc)

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
            if self.peer_communities[k]:
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

        # Build matrix A and vector M
        if not self._build_template_matrices():
            return False

        # Solve using Binary Split
        h_solution = self._solve_binary_split()
        if h_solution is None:
            logger.error("Binary Split solving failed")
            return False

        if not self._validate_solution(h_solution):
            logger.error("Solution validation failed")
            return False

        self.h_vector = h_solution
        self.template_solved = True
        self.evaluate_template_solution()

        logger.info(f"Template solved successfully: h = {self.h_vector}")
        return True

    def _build_template_matrices(self):
        """Build matrix A and vector M for template solving"""
        try:
            K = len(self.peer_communities)
            F = self.num_classes

            # Build matrix A and upper bounds B
            self.A_matrix = np.zeros((F, K))
            self.B_vector = np.zeros(K)

            for k, pc in enumerate(self.peer_communities):
                if not pc:  # Skip empty clusters
                    continue

                self.B_vector[k] = len(pc)

                # Calculate cluster distribution
                pc_raw_dist = self._get_raw_distribution(pc)
                total_samples = sum(pc_raw_dist.values()) if pc_raw_dist else 1

                # Build A matrix
                for f in range(F):
                    class_samples = pc_raw_dist.get(str(f), 0)
                    p_ck_f = class_samples / total_samples if total_samples > 0 else 0
                    self.A_matrix[f, k] = self.n_batch_size * p_ck_f

            # Build vector M
            total_clients_selected = self.S
            self.M_vector = (
                self.n_batch_size * total_clients_selected * self.target_distribution
            )

            logger.info(f"Matrix A shape: {self.A_matrix.shape}")
            logger.info(f"Vector M shape: {self.M_vector.shape}")
            logger.info(f"Upper bounds B: {self.B_vector}")
            logger.info(f"Total clusters K: {K}")
            logger.info(f"Total clients to select S: {self.S}")

            return True

        except Exception as e:
            logger.error(f"Error building template matrices: {str(e)}")
            return False

    def _solve_binary_split(self):
        """Solve the constrained least squares problem using Binary Split method"""
        try:
            K = len(self.peer_communities)

            # Binary decomposition
            G = []
            total_binary_vars = 0

            for k in range(K):
                B_k = int(self.B_vector[k])
                G_k = math.floor(math.log2(B_k)) + 1 if B_k > 0 else 1
                G.append(G_k)
                total_binary_vars += G_k

            logger.info(
                f"Binary decomposition: G = {G}, total binary vars = {total_binary_vars}"
            )

            # Build transformation matrix T
            T_matrix = self._build_transformation_matrix(G)

            # Solve using optimization solver
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
            AT = self.A_matrix @ T_matrix

            # Quadratic objective
            obj = gp.QuadExpr()
            for f in range(self.num_classes):
                expr = gp.LinExpr()
                for j in range(total_binary_vars):
                    expr += AT[f, j] * b_vars[j]
                expr -= self.M_vector[f]
                obj += expr * expr

            model.setObjective(obj, GRB.MINIMIZE)

            # Constraint 1: sum(T*b) = S - K
            additional_clients = self.S - K
            constraint_expr = gp.LinExpr()
            for i in range(K):
                for j in range(total_binary_vars):
                    constraint_expr += T_matrix[i, j] * b_vars[j]
            model.addConstr(constraint_expr == additional_clients, "sum_constraint")

            # Constraint 2: Each cluster's selection doesn't exceed its size
            col_idx = 0
            for i in range(K):
                if self.peer_communities[i]:
                    cluster_expr = gp.LinExpr()
                    for j in range(G[i]):
                        cluster_expr += T_matrix[i, col_idx] * b_vars[col_idx]
                        col_idx += 1
                    model.addConstr(
                        cluster_expr <= self.B_vector[i] - 1, f"cluster_{i}_max"
                    )
                else:
                    col_idx += G[i]

            # Solve
            model.optimize()

            if model.status == GRB.OPTIMAL:
                # Extract solution
                b_solution = np.array([b_vars[j].x for j in range(total_binary_vars)])
                z_solution = T_matrix @ b_solution

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

            # Constraints for auxiliary variables
            for f in range(self.num_classes):
                expr = (
                    lpSum(AT[f, j] * b_vars[j] for j in range(total_binary_vars))
                    - self.M_vector[f]
                )
                prob += aux_vars[f] >= expr
                prob += aux_vars[f] >= -expr

            # Constraint 1: sum(T*b) = S - K
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
                if self.peer_communities[i]:
                    cluster_expr = lpSum(
                        T_matrix[i, col_idx + j] * b_vars[col_idx + j]
                        for j in range(G[i])
                    )
                    prob += cluster_expr <= self.B_vector[i] - 1
                    col_idx += G[i]
                else:
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
        """Evaluate template solution against target"""
        if not hasattr(self, "h_vector") or self.h_vector is None:
            logger.error("Solution not calculated yet")
            return

        try:
            # Calculate predicted feature vector
            M_hat = self.A_matrix @ self.h_vector

            # Calculate metrics
            diff_absolute = M_hat - self.M_vector
            diff_relative = np.divide(
                diff_absolute,
                self.M_vector,
                out=np.zeros_like(diff_absolute),
                where=self.M_vector != 0,
            )
            max_abs_diff = np.max(np.abs(diff_absolute))
            l2_norm = np.linalg.norm(diff_absolute)

            # Calculate KL divergence
            epsilon = 1e-10
            target_dist_normalized = self.target_distribution / np.sum(
                self.target_distribution
            )
            actual_dist = M_hat / np.sum(M_hat)
            kl_divergence = np.sum(
                target_dist_normalized
                * np.log(target_dist_normalized / (actual_dist + epsilon) + epsilon)
            )

            # Calculate feature matching score
            matched_features = np.sum(
                np.abs(diff_relative) < 0.05
            )  # 5% error tolerance
            feature_score = matched_features / self.num_classes

            # Log results
            logger.info(
                f"Feature match score: {feature_score:.1%} ({matched_features}/{self.num_classes} features matched within 5% error)"
            )
            logger.info(f"Max absolute difference: {max_abs_diff:.4f}")
            logger.info(f"L2 norm of differences: {l2_norm:.4f}")
            logger.info(f"KL divergence: {kl_divergence:.4f}")
            logger.info(f"Predicted features: {M_hat}")
            logger.info(f"Target features: {self.M_vector}")

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

        selected_clients_per_cluster = {}
        total_selected = 0

        for k, pc in enumerate(self.peer_communities):
            num_to_select = int(round(self.h_vector[k]))

            if num_to_select <= 0:
                selected_clients_per_cluster[k] = []
                continue

            # Check if cluster has enough clients
            if len(pc) < num_to_select:
                logger.error(
                    f"Cluster {k} has only {len(pc)} clients, but {num_to_select} were requested"
                )
                raise ValueError(f"Not enough clients in cluster {k}")

            # Select clients
            if num_to_select == len(pc):
                selected = list(pc)
            else:
                selected = random.sample(list(pc), num_to_select)

            selected_clients_per_cluster[k] = selected
            total_selected += num_to_select

        # Log results
        logger.info(
            f"Selected clients from {len(self.peer_communities)} clusters for round {round_idx}:"
        )
        for k, clients in selected_clients_per_cluster.items():
            logger.info(f"Cluster #{k}: {len(clients)} clients - {clients}")

        # Verify total selection count
        logger.info(f"Total clients selected: {total_selected} (expected: {self.S})")
        if total_selected != self.S:
            logger.error(
                f"Selection mismatch: selected {total_selected}, expected {self.S}"
            )
            raise ValueError("Total selected clients does not match expected total")

        return selected_clients_per_cluster

    def _get_raw_distribution(self, client_ids):
        """Calculate raw sample counts for given client set"""
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


class FDSEClient(Client):
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
        """Initialize FDSE client with proper parameter handling"""
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
        """Register message handlers"""
        super()._register_default_handlers()

        # Override model_para handler
        self.register_handlers(
            "model_para",
            self.callback_funcs_for_model_para,
            ["model_para", "ss_model_para"],
        )

        # Register new message handlers
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
        """Handle personalized model slices from server"""
        round, sender, payload = message.state, message.sender, message.content

        # Update shared layers only
        cur = self.model.state_dict()
        cur.update(payload)
        self.model.load_state_dict(cur, strict=False)

        # Update round state
        self.state = round

        # Local training
        sample_size, _, results = self.trainer.train()

        # Log training results
        logger.info(
            self._monitor.format_eval_res(
                results,
                rnd=self.state,
                role="Client #{}".format(self.ID),
                return_raw=True,
            )
        )

        # Get model parameters (filtered by custom trainer)
        model_para = copy.deepcopy(self.trainer.get_model_para())

        # Send to server
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
        """Handle model update without triggering training"""
        round, sender, payload = message.state, message.sender, message.content

        if payload is not None:
            # Update shared layers
            cur = self.model.state_dict()
            cur.update(payload)
            self.model.load_state_dict(cur, strict=False)

        # Update round state
        self.state = round
        logger.debug(f"Client #{self.ID}: Model updated without training")

    def callback_funcs_for_trigger_train(self, message: Message):
        """Handle training trigger from server"""
        round, sender, payload = message.state, message.sender, message.content

        # Update model if server sent parameters
        if payload is not None:
            cur = self.model.state_dict()
            cur.update(payload)
            self.model.load_state_dict(cur, strict=False)

        # Update round state
        self.state = round

        # Execute local training
        sample_size, _, results = self.trainer.train()

        # Log training results
        logger.info(
            self._monitor.format_eval_res(
                results,
                rnd=self.state,
                role="Client #{}".format(self.ID),
                return_raw=True,
            )
        )

        # Get and send model parameters
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
        """Handle evaluation request from server"""
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


# Register worker according to framework requirements
def call_fdse_worker(worker_type):
    if worker_type == "fdse":
        worker_builder = {"client": FDSEClient, "server": FDSEServer}
        return worker_builder
    return None


register_worker("fdse", call_fdse_worker)
