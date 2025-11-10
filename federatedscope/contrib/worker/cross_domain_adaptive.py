"""Workers for the cross-domain adaptive (CDA) algorithm."""
import logging
from typing import Dict, List

import numpy as np
import torch

from federatedscope.contrib.utils.nnls import nnls_solve
from federatedscope.contrib.worker.ondemfl import OnDemFLClient
from federatedscope.core.message import Message
from federatedscope.core.workers.client_FedLSA import FedLSAClient
from federatedscope.core.workers.server_FedLSA import FedLSAServer
from federatedscope.register import register_worker

logger = logging.getLogger(__name__)


class CrossDomainAdaptiveClient(FedLSAClient, OnDemFLClient):
    """Client that supports both FedLSA anchors and On-Demand predictors."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CrossDomainAdaptiveServer(FedLSAServer):
    """Server that aligns semantic anchors with On-Demand client selection."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self._cfg.ondemfl
        self._ondem_enabled = cfg.enable
        self._pretrain_rounds = cfg.pretrain_rounds
        self._subset_size = cfg.subset_size
        self._min_ratio = cfg.min_ratio
        self._nnls_max_iter = cfg.nnls_max_iter
        self._nnls_tol = cfg.nnls_tol
        self._weight_scheme = cfg.weight_scheme.lower()
        self._target_distribution = np.asarray(cfg.target_distribution,
                                               dtype=np.float64)
        self._clients: Dict[int, CrossDomainAdaptiveClient] = dict()
        self._ondem_active = False
        self._need_setup_after_stage1 = False
        self._ratio_map: Dict[int, float] = dict()
        self._sampling_clients: List[int] = []

        if self._ondem_enabled:
            expected_total = cfg.pretrain_rounds + cfg.ondemand_rounds
            if expected_total > 0 and self.total_round_num != expected_total:
                logger.warning('total_round_num (%s) != pretrain (%s) + '
                               'ondemand (%s).', self.total_round_num,
                               cfg.pretrain_rounds, cfg.ondemand_rounds)

    def link_clients(self, clients):
        self._clients = {
            cid: client for cid, client in clients.items()
            if isinstance(client, CrossDomainAdaptiveClient)
        }

    @staticmethod
    def _sanitize_distribution(dist):
        if dist is None:
            return None
        if hasattr(dist, 'detach'):
            dist = dist.detach()
        if hasattr(dist, 'cpu'):
            dist = dist.cpu()
        try:
            arr = dist.numpy()
        except AttributeError:
            arr = np.asarray(dist)
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=np.float64).flatten()
        if arr.size == 0:
            return None
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, 0.0, None)
        total = arr.sum()
        if total <= 0:
            return None
        return arr / total

    def _aggregate_client_distribution(self, attr_name: str):
        if not self._ratio_map:
            return None
        agg = None
        total_weight = 0.0
        for cid, weight in self._ratio_map.items():
            client = self._clients.get(cid)
            if client is None:
                continue
            dist = self._sanitize_distribution(
                getattr(client, attr_name, None))
            if dist is None:
                continue
            if agg is None:
                agg = np.zeros_like(dist, dtype=np.float64)
            agg += weight * dist
            total_weight += weight
        if agg is None or total_weight <= 0:
            return None
        return agg / total_weight

    def _log_subset_mae(self, context: str):
        if not self._ratio_map or self._target_distribution.size == 0:
            return
        target = self._sanitize_distribution(self._target_distribution)
        if target is None:
            return
        pred_dist = self._aggregate_client_distribution(
            'estimated_distribution')
        true_dist = self._aggregate_client_distribution('true_distribution')
        pred_mae = (float(np.mean(np.abs(pred_dist - target)))
                    if pred_dist is not None else None)
        true_mae = (float(np.mean(np.abs(true_dist - target)))
                    if true_dist is not None else None)
        logger.info(
            '[OnDemand][%s] MAE(pred,target)=%s | MAE(true,target)=%s',
            context,
            'N/A' if pred_mae is None else f'{pred_mae:.6f}',
            'N/A' if true_mae is None else f'{true_mae:.6f}')

    def _perform_federated_aggregation(self):
        if self._ondem_enabled and self._ondem_active and \
                not self._cfg.federate.use_ss:
            buffer = self.msg_buffer['train'][self.state]
            adjusted = dict()
            for client_id, content in buffer.items():
                ratio = self._ratio_map.get(client_id, 0.0)
                sample_size, model_para = content
                if self._weight_scheme == 'ratio':
                    adjusted_sample = ratio
                elif self._weight_scheme == 'size':
                    adjusted_sample = sample_size
                else:
                    adjusted_sample = sample_size * ratio
                adjusted[client_id] = (adjusted_sample, model_para)
            self.msg_buffer['train'][self.state] = adjusted
        aggregated_num = super()._perform_federated_aggregation()
        if self._ondem_enabled and not self._ondem_active and \
                self.state == self._pretrain_rounds:
            self._need_setup_after_stage1 = True
        return aggregated_num

    def _start_new_training_round(self, aggregated_num=0):
        if self._ondem_enabled and self._need_setup_after_stage1:
            self._prepare_on_demand_phase()
            self._need_setup_after_stage1 = False
        if self._ondem_enabled and self._ondem_active and \
                self.mode == 'standalone':
            self._log_subset_mae(f'round#{self.state}')
            self.train_semantic_anchors()
            self.broadcast_model_para(msg_type='model_para',
                                      sample_client_num=self.sample_client_num)
        else:
            super()._start_new_training_round(aggregated_num)

    def _prepare_on_demand_phase(self):
        if not self._clients:
            logger.warning('Cross-domain adaptive server cannot access '
                           'clients; fallback to FedLSA-only training.')
            return
        if len(self._target_distribution) == 0:
            client = next(iter(self._clients.values()))
            predictor = getattr(client, 'estimated_distribution', None)
            if predictor is None:
                logger.error('No distribution predictions collected; '
                             'cannot infer target dimension.')
                return
            self._target_distribution = np.ones_like(predictor) / len(predictor)
        preds = []
        ids = []
        for client_id, client in self._clients.items():
            if client.estimated_distribution is None:
                continue
            preds.append(client.estimated_distribution)
            ids.append(client_id)
        if not preds:
            logger.error('No valid distribution predictions collected; '
                         'abort On-Demand phase.')
            return
        num_classes = preds[0].shape[0]
        if self._target_distribution.shape[0] != num_classes:
            raise ValueError('Target distribution length mismatch.')
        pred_matrix = np.stack(preds, axis=0)
        D = pred_matrix.T
        target = self._target_distribution.astype(np.float64)
        target = target / target.sum()
        ratios = nnls_solve(D,
                            target,
                            max_iter=self._nnls_max_iter,
                            tol=self._nnls_tol)
        ratios = np.asarray(ratios, dtype=np.float64)
        ratios[ratios < self._min_ratio] = 0.0
        if ratios.sum() == 0:
            ratios = np.ones_like(ratios) / len(ratios)
        else:
            ratios = ratios / ratios.sum()
        target_size = len(ids)
        if self._subset_size and self._subset_size > 0:
            target_size = min(target_size, self._subset_size)
        target_size = max(1, target_size)
        positive_idx = np.where(ratios > self._min_ratio)[0]
        selected_idx = positive_idx.tolist()
        selected_scores = ratios[positive_idx].tolist()
        if len(selected_idx) < target_size:
            remaining_idx = [idx for idx in range(len(ids))
                             if idx not in selected_idx]
            slots_needed = target_size - len(selected_idx)
            if remaining_idx and slots_needed > 0:
                distances = np.linalg.norm(pred_matrix[remaining_idx] - target,
                                           axis=1)
                order = np.argsort(distances)[:slots_needed]
                fallback_idx = [remaining_idx[i] for i in order]
                fallback_scores = 1.0 / np.maximum(distances[order], 1e-12)
                selected_idx.extend(fallback_idx)
                selected_scores.extend(fallback_scores.tolist())
        if not selected_idx:
            logger.error('Unable to assemble On-Demand subset.')
            return
        selected_scores = np.asarray(selected_scores, dtype=np.float64)
        if selected_scores.sum() <= 0:
            selected_scores = np.ones(len(selected_idx), dtype=np.float64)
        selected_scores = selected_scores / selected_scores.sum()
        paired = [(ids[idx], selected_scores[i])
                  for i, idx in enumerate(selected_idx)]
        paired.sort(key=lambda x: x[1], reverse=True)
        if len(paired) > target_size:
            paired = paired[:target_size]
        total_weight = sum(weight for _, weight in paired)
        if total_weight > 0:
            paired = [(cid, weight / total_weight) for cid, weight in paired]
        self._ratio_map = {cid: float(r) for cid, r in paired if r > 0}
        self._sampling_clients = list(self._ratio_map.keys())
        if not self._sampling_clients:
            logger.error('Selected subset is empty; skip On-Demand stage.')
            return
        weights_full = np.zeros(len(ids), dtype=np.float64)
        for idx, cid in enumerate(ids):
            weights_full[idx] = self._ratio_map.get(cid, 0.0)
        approx_dist = weights_full @ pred_matrix
        approx_sum = approx_dist.sum()
        if approx_sum > 0:
            approx_dist = approx_dist / approx_sum
        else:
            approx_dist = None
        logger.info('[Subset Select] subset=%s | approx target=%s',
                    self._sampling_clients, approx_dist)
        self._log_subset_mae('prepare')
        if len(self._sampling_clients) < self.sample_client_num:
            self.sample_client_num = len(self._sampling_clients)
            try:
                self._cfg.defrost()
                self._cfg.federate.sample_client_num = self.sample_client_num
                self._cfg.freeze()
            except AttributeError:
                self._cfg.federate.sample_client_num = self.sample_client_num
        for client in self._clients.values():
            client.set_phase('ondemand')
        self._ondem_active = True

    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1,
                             filter_unseen_clients=True):
        if self._ondem_enabled and self._ondem_active and \
                msg_type == 'model_para' and sample_client_num > 0 and \
                self.mode == 'standalone':
            if filter_unseen_clients:
                self.sampler.change_state(self.unseen_clients_id, 'unseen')
            sample_client_num = min(sample_client_num,
                                    len(self._sampling_clients))
            if sample_client_num <= 0:
                logger.warning('No clients available for On-Demand broadcast.')
                return
            receiver = self._sample_ondemand_clients(sample_client_num)
            if self._noise_injector is not None:
                for model_idx_i in range(len(self.models)):
                    num_sample_clients = [
                        v["num_sample"] for v in self.join_in_info.values()
                    ]
                    self._noise_injector(self._cfg, num_sample_clients,
                                         self.models[model_idx_i])
            skip_broadcast = self._cfg.federate.method in ['local', 'global']
            if self.model_num > 1:
                model_para = [{} if skip_broadcast else model.state_dict()
                              for model in self.models]
            else:
                model_para = {} if skip_broadcast else self.models[0].state_dict()
            if msg_type == 'model_para' and not skip_broadcast and \
                    self._cfg.quantization.method == 'uniform':
                from federatedscope.core.compression import \
                    symmetric_uniform_quantization
                nbits = self._cfg.quantization.nbits
                if self.model_num > 1:
                    model_para = [
                        symmetric_uniform_quantization(x, nbits)
                        for x in model_para
                    ]
                else:
                    model_para = symmetric_uniform_quantization(model_para,
                                                                nbits)
            rnd = self.state if msg_type != 'evaluate' else self.state - 1
            self.comm_manager.send(
                Message(msg_type=msg_type,
                        sender=self.ID,
                        receiver=receiver,
                        state=min(rnd, self.total_round_num),
                        timestamp=self.cur_timestamp,
                        content=model_para))
            if self.semantic_anchors is not None:
                self.broadcast_semantic_anchors(receiver=receiver)
            if self._cfg.federate.online_aggr:
                for idx in range(self.model_num):
                    self.aggregators[idx].reset()
            if filter_unseen_clients:
                self.sampler.change_state(self.unseen_clients_id, 'seen')
            return
        super().broadcast_model_para(
            msg_type=msg_type,
            sample_client_num=sample_client_num,
            filter_unseen_clients=filter_unseen_clients)

    def _sample_ondemand_clients(self, size: int) -> List[int]:
        size = min(size, len(self._sampling_clients))
        if size <= 0:
            return []
        if size >= len(self._sampling_clients):
            chosen = self._sampling_clients
        else:
            idle_mask = self.sampler.client_state.astype(bool)
            available = [cid for cid in self._sampling_clients if idle_mask[cid]]
            if len(available) < size:
                available = self._sampling_clients
                size = min(size, len(available))
            probs = np.array([self._ratio_map[cid] for cid in available],
                             dtype=np.float64)
            probs = probs / probs.sum()
            chosen = np.random.choice(available,
                                      size=size,
                                      replace=False,
                                      p=probs).tolist()
        self.sampler.change_state(chosen, 'working')
        return chosen


def call_cross_domain_adaptive_worker(method: str):
    if method.lower() == 'cross_domain_adaptive':
        return {
            'client': CrossDomainAdaptiveClient,
            'server': CrossDomainAdaptiveServer,
        }
    return None


register_worker('cross_domain_adaptive', call_cross_domain_adaptive_worker)
