import copy
import logging
from typing import Dict, List, Optional

import numpy as np

from federatedscope.core.message import Message
from federatedscope.core.workers import Client, Server
from federatedscope.register import register_worker

from federatedscope.contrib.utils.nnls import nnls_solve

logger = logging.getLogger(__name__)


class OnDemFLClient(Client):
    """
    Client implementation that caches the distribution estimates produced by
    the OnDemFL trainer.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.estimated_distribution: Optional[np.ndarray] = None
        self.true_distribution: Optional[np.ndarray] = None

    def set_phase(self, phase: str):
        if hasattr(self.trainer, 'set_phase'):
            self.trainer.set_phase(phase)

    def callback_funcs_for_model_para(self, message: Message):
        super().callback_funcs_for_model_para(message)
        self._refresh_distribution_cache()

    def _refresh_distribution_cache(self):
        if hasattr(self.trainer, 'ctx'):
            pred = getattr(self.trainer.ctx, 'ondem_pred_distribution', None)
            if pred is not None:
                self.estimated_distribution = pred.numpy()
            true = getattr(self.trainer.ctx, 'ondem_true_distribution', None)
            if true is not None:
                self.true_distribution = true.numpy()

    @property
    def num_train_samples(self) -> int:
        train_data = self.trainer.ctx.train_data
        if hasattr(train_data, '__len__'):
            return len(train_data)
        if hasattr(self.data, 'num_train_data'):
            return self.data.num_train_data
        return 0


class OnDemFLServer(Server):
    """
    Server implementation for the On-Demand Federated Learning algorithm.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._clients: Dict[int, OnDemFLClient] = dict()
        self._ondem_active = False
        self._need_setup_after_stage1 = False
        self._ratio_map: Dict[int, float] = dict()
        self._sampling_clients: List[int] = []
        cfg = self._cfg.ondemfl
        self._enabled = cfg.enable
        self._pretrain_rounds = cfg.pretrain_rounds
        self._subset_size = cfg.subset_size
        self._min_ratio = cfg.min_ratio
        self._nnls_max_iter = cfg.nnls_max_iter
        self._nnls_tol = cfg.nnls_tol
        self._weight_scheme = cfg.weight_scheme.lower()
        self._target_distribution = np.asarray(cfg.target_distribution,
                                               dtype=np.float64)
        if self._enabled:
            expected_total = cfg.pretrain_rounds + cfg.ondemand_rounds
            if expected_total > 0 and \
                    self.total_round_num != expected_total:
                logger.warning(
                    'Configured total_round_num (%s) does not match the sum of '
                    'pretrain_rounds (%s) and ondemand_rounds (%s).',
                    self.total_round_num, cfg.pretrain_rounds,
                    cfg.ondemand_rounds)

    def link_clients(self, clients: Dict[int, Client]):
        self._clients = {cid: client for cid, client in clients.items()}

    def _perform_federated_aggregation(self):
        if self._enabled and self._ondem_active and \
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

        if self._enabled and not self._ondem_active and \
                self.state == self._pretrain_rounds:
            self._need_setup_after_stage1 = True

        return aggregated_num

    def _start_new_training_round(self, aggregated_num=0):
        if self._enabled and self._need_setup_after_stage1:
            self._prepare_on_demand_phase()
            self._need_setup_after_stage1 = False

        if self._enabled and self._ondem_active and self.mode == 'standalone':
            self.broadcast_model_para(msg_type='model_para',
                                      sample_client_num=self.sample_client_num)
        else:
            super()._start_new_training_round(aggregated_num)

    def _prepare_on_demand_phase(self):
        if not self._clients:
            logger.warning('OnDemFLServer cannot access client objects; '
                           'fallback to normal training.')
            return

        if len(self._target_distribution) == 0:
            predictor = next(iter(
                self._clients.values())).estimated_distribution
            if predictor is None:
                logger.error('No distribution predictions available to infer '
                             'target dimension.')
                return
            self._target_distribution = np.ones_like(predictor) / len(
                predictor)

        preds = []
        ids = []
        for client_id, client in self._clients.items():
            if not isinstance(client, OnDemFLClient):
                continue
            if client.estimated_distribution is None:
                logger.warning(
                    'Client #%s did not upload distribution '
                    'prediction; skip from OnDem subset.', client_id)
                continue
            preds.append(client.estimated_distribution)
            ids.append(client_id)

        if not preds:
            logger.error('No valid distribution predictions collected; '
                         'cannot activate On-Demand stage.')
            return

        num_classes = preds[0].shape[0]
        if self._target_distribution.shape[0] != num_classes:
            raise ValueError(
                f'Target distribution length ({len(self._target_distribution)})'
                f' mismatches class count ({num_classes}).')

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
            remaining_idx = [
                idx for idx in range(len(ids)) if idx not in selected_idx
            ]
            slots_needed = target_size - len(selected_idx)
            if not selected_idx and remaining_idx:
                logger.warning(
                    'NNLS returned an all-zero or insufficient solution; '
                    'fall back to distance-based selection for the On-Demand stage.'
                )
            if remaining_idx and slots_needed > 0:
                distances = np.linalg.norm(pred_matrix[remaining_idx] - target,
                                           axis=1)
                order = np.argsort(distances)[:slots_needed]
                fallback_idx = [remaining_idx[i] for i in order]
                fallback_scores = 1.0 / (np.maximum(distances[order], 1e-12))
                if selected_idx:
                    logger.info(
                        'NNLS selected %s clients (< target %s); '
                        'adding %s closest clients based on predictor distance '
                        'to the target distribution.', len(selected_idx),
                        target_size, len(fallback_idx))
                selected_idx.extend(fallback_idx)
                selected_scores.extend(fallback_scores.tolist())

        if not selected_idx:
            logger.error('Unable to assemble an On-Demand subset after NNLS '
                         'and fallback selection.')
            return

        selected_idx_scores = np.asarray(selected_scores, dtype=np.float64)
        if selected_idx_scores.sum() <= 0:
            selected_idx_scores = np.ones(len(selected_idx), dtype=np.float64)
        selected_idx_scores = selected_idx_scores / selected_idx_scores.sum()

        paired = [(ids[idx], selected_idx_scores[i])
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
            logger.error('On-Demand client subset is empty; abort stage 2.')
            return

        weights_full = np.zeros(len(ids), dtype=np.float64)
        for idx, cid in enumerate(ids):
            weights_full[idx] = self._ratio_map.get(cid, 0.0)

        approx_dist = weights_full @ pred_matrix
        approx_sum = approx_dist.sum()
        if approx_sum > 0:
            approx_dist = approx_dist / approx_sum
        mae = float(np.mean(np.abs(approx_dist - target)))

        logger.info(
            '[Subset Select] selected subset is %s, subset size is %s and '
            'weight is %s', np.asarray(self._sampling_clients),
            len(self._sampling_clients), weights_full)
        logger.info('[Subset Select] approximated target distribution is %s',
                    approx_dist)
        logger.info(
            '[Subset Select] MAE (approximated target distribution and target '
            'distribution is %s', mae)

        if len(self._sampling_clients) < self.sample_client_num:
            self.sample_client_num = len(self._sampling_clients)
            try:
                self._cfg.defrost()
                self._cfg.federate.sample_client_num = self.sample_client_num
                self._cfg.freeze()
            except AttributeError:
                self._cfg.federate.sample_client_num = self.sample_client_num
            logger.info('Adjust sample_client_num to %s for On-Demand stage.',
                        self.sample_client_num)

        for client in self._clients.values():
            if isinstance(client, OnDemFLClient):
                client.set_phase('ondemand')

        self._ondem_active = True
        logger.info('On-Demand stage activated with clients %s and ratios %s',
                    self._sampling_clients,
                    [self._ratio_map[cid] for cid in self._sampling_clients])

    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1,
                             filter_unseen_clients=True):
        if self._enabled and self._ondem_active and msg_type == 'model_para' \
                and sample_client_num > 0 and self.mode == 'standalone':
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

            skip_broadcast = self._cfg.federate.method in ["local", "global"]
            if self.model_num > 1:
                model_para = [{} if skip_broadcast else model.state_dict()
                              for model in self.models]
            else:
                model_para = {} if skip_broadcast else self.models[
                    0].state_dict()

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
                    model_para = symmetric_uniform_quantization(
                        model_para, nbits)

            rnd = self.state if msg_type != 'evaluate' else self.state - 1
            self.comm_manager.send(
                Message(msg_type=msg_type,
                        sender=self.ID,
                        receiver=receiver,
                        state=min(rnd, self.total_round_num),
                        timestamp=self.cur_timestamp,
                        content=model_para))
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
            available = [
                cid for cid in self._sampling_clients if idle_mask[cid]
            ]
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


def call_ondemfl_worker(method: str):
    if method.lower() == 'ondemfl':
        return {
            'client': OnDemFLClient,
            'server': OnDemFLServer,
        }
    return None


register_worker('ondemfl', call_ondemfl_worker)
