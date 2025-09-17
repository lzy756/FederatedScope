from federatedscope.core.workers.server import Server
from federatedscope.core.workers.client import Client
from federatedscope.core.message import Message
from federatedscope.register import register_worker
import logging
import os
import torch

logger = logging.getLogger(__name__)


class FedSAKServer(Server):
    """
    继承默认 Server，主要特点:
      • broadcast_model_para: 不构造全局模型，而是广播每客户端个性化模型
      • callback_funcs_for_model_para: 使用FedSAKAggregator进行聚合
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

        self.agg_lmbda = config.aggregator.get("lambda_", 1e-3)
        # 保存共享层配置，用于调试
        self.share_patterns = config.fedsak.share_patterns

        # 初始化客户端选择样本ID列表（当前轮次的）
        self.sample_client_ids = []

        # 新增：记录上一轮参与训练的客户端ID列表
        self.prev_sample_client_ids = []

    def _register_default_handlers(self):
        """覆写默认消息处理函数注册"""
        super()._register_default_handlers()
        # 覆盖model_para处理器
        self.register_handlers('model_para',
                               self.callback_funcs_for_model_para,
                               ['model_para', 'evaluate', 'finish'])

    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1,
                             filter_unseen_clients=True):
        """覆写广播模型参数方法"""
        # 与基类保持一致：仅在训练广播时超过总轮次才终止；评估广播需要放行
        if msg_type != 'evaluate' and \
                self.state >= self._cfg.federate.total_round_num:
            self.terminate(msg_type='finish')
            return

        # 如果是评估请求，直接发送给所有客户端
        if msg_type == 'evaluate':
            receiver = list(self.comm_manager.neighbors.keys())
            for rcv_idx in receiver:
                # 如果有个性化切片，使用客户端的个性化模型进行评估
                content = None
                if hasattr(self, "personalized_slices") and \
                        rcv_idx in self.personalized_slices:
                    content = self.personalized_slices[rcv_idx]

                # 与基类一致：评估发生在一个训练轮结束后，使用 state-1 记录评测轮次
                eval_state = max(self.state - 1, 0)
                self.comm_manager.send(
                    Message(msg_type=msg_type,
                            sender=self.ID,
                            receiver=[rcv_idx],
                            state=eval_state,
                            content=content))
            return

        # 处理两种情况：
        # 1. 向上一轮参与的客户端发送更新后的个性化切片
        # 2. 采样新客户端启动新一轮训练

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

            # 清空上一轮客户端列表(已经发送完成)
            self.prev_sample_client_ids = []

        # 确保所有客户端状态正确设置为"空闲"
        all_client_ids = list(self.comm_manager.neighbors.keys())
        self.sampler.change_state(all_client_ids, 'idle')

        # 然后，采样新的客户端进行下一轮训练
        if filter_unseen_clients:
            # 适配父类方法的用法
            self.sampler.change_state(self.unseen_clients_id, 'unseen')

        # 获取需要广播的客户端
        if sample_client_num > 0:
            receiver = self.sampler.sample(size=sample_client_num)
            # 保存当前轮次选择的客户端ID
            self.sample_client_ids = receiver
        else:
            # 广播给所有客户端
            receiver = list(self.comm_manager.neighbors.keys())
            if msg_type == 'model_para':
                self.sampler.change_state(receiver, 'working')
                self.sample_client_ids = receiver

        # 广播开始训练的消息 - 不需要发送模型参数，客户端将使用自己当前的模型
        for rcv_idx in receiver:
            # 对于首轮训练或没有个性化切片的情况，需要发送初始模型
            if self.state == 0 or not hasattr(self, "personalized_slices"):
                content = self.model.state_dict()
                self.comm_manager.send(
                    Message(msg_type="model_para",
                            sender=self.ID,
                            receiver=[rcv_idx],
                            state=self.state,
                            content=content))
            else:
                # 如果客户端有个性化切片，使用trigger_train消息类型触发训练
                # 如果没有个性化切片(新加入的客户端)，发送空内容，客户端会使用现有模型
                content = None
                if rcv_idx in self.personalized_slices:
                    content = self.personalized_slices[rcv_idx]
                self.comm_manager.send(
                    Message(msg_type="trigger_train",
                            sender=self.ID,
                            receiver=[rcv_idx],
                            state=self.state,
                            content=content))

    def callback_funcs_for_model_para(self, message: Message):
        """处理来自客户端的模型参数"""
        round, sender, content = message.state, message.sender, message.content

        # 更新状态，使用正确的轮次管理
        if round not in self.msg_buffer['train']:
            self.msg_buffer['train'][round] = dict()

        # 保存客户端上传的模型
        self.msg_buffer['train'][round][sender] = content

        # 等待所有选中客户端返回
        if len(self.msg_buffer['train'][round]) < len(self.sample_client_ids):
            return False

        # 准备聚合信息
        client_feedback = []
        for cid in self.sample_client_ids:
            client_info = {
                'client_id': cid,
                'model_para': self.msg_buffer['train'][round][cid]
            }
            client_feedback.append(client_info)

        # 使用聚合器进行聚合
        agg_info = {'client_feedback': client_feedback}
        result = self.aggregator.aggregate(agg_info)

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

        # 清空缓冲 & 进入下一轮
        self.msg_buffer['train'][round].clear()
        self.state += 1

        # 检查是否需要进行评估
        if (self.state % self._cfg.eval.freq == 0
                and self.state != self._cfg.federate.total_round_num):
            logger.info('Server: Starting evaluation at the end of round '
                        f'{self.state - 1}')
            # 使用原始eval方法，它会调用broadcast_model_para
            self.eval()

        if self.state < self._cfg.federate.total_round_num:
            # 启动新一轮训练
            logger.info('----------- Starting a new training round' +
                        f' (Round #{self.state}) -------------')
            self.broadcast_model_para(
                msg_type='model_para',
                sample_client_num=self._cfg.federate.sample_client_num)
        else:
            # 最终评估
            logger.info('Server: Training is finished! Starting evaluation.')
            self.eval()

        return True


class FedSAKClient(Client):
    """
    FedSAK客户端，主要特点:
      – 收到服务器回包只替换共享层
      – 然后执行本地训练
      – 上传参数时依赖FedSAKTrainer的_param_filter自动过滤共享层
    """
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
        """修改构造函数以匹配Client父类的参数列表"""
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
            self._monitor.format_eval_res(results,
                                          rnd=self.state,
                                          role='Client #{}'.format(self.ID),
                                          return_raw=True))

        # 2.5) 保存本地模型
        self._save_local_model()

        # 3) 获取模型参数 - 自定义trainer的_param_filter会自动筛选共享层
        model_para = self.trainer.get_model_para()

        # 4) 发送给服务器 - 使用标准通信接口
        self.comm_manager.send(
            Message(msg_type="model_para",
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

        # 保存本地模型
        self._save_local_model()

        # 获取模型参数并发送
        model_para = self.trainer.get_model_para()
        # print(model_para.keys())

        self.comm_manager.send(
            Message(msg_type="model_para",
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=(sample_size, model_para)))

    # ------- 工具方法：保存客户端本地模型 -------
    def _save_local_model(self):
        """将当前客户端完整本地模型保存到 {save_to}/client/client_model_{ID}.pt

        说明：
        - 仅在配置项 cfg.federate.save_to 非空时启用；
        - 直接使用 torch.save，保持与框架一致的 checkpoint 结构：
          {'cur_round': round, 'model': state_dict}
        - 采用覆盖式保存，保持最新快照；
        """
        try:
            save_root = getattr(self._cfg.federate, 'save_to', '')
            if not save_root:
                return
            save_dir = os.path.join(save_root, 'client')
            os.makedirs(save_dir, exist_ok=True)
            ckpt_path = os.path.join(save_dir, f'client_model_{self.ID}.pt')
            # 直接保存当前本地完整模型
            ckpt = {
                'cur_round': self.state,
                'model': self.model.state_dict(),
            }
            torch.save(ckpt, ckpt_path)
            logger.debug(f"Client #{self.ID}: model saved to {ckpt_path} "
                         f"(round {self.state})")
        except Exception as e:
            logger.warning(
                f"Client #{self.ID}: failed to save model, error: {e}")


# 按照框架要求注册worker
def call_fedsak_worker(worker_type):
    if worker_type == 'fedsak':
        worker_builder = {'client': FedSAKClient, 'server': FedSAKServer}
        return worker_builder
    return None


register_worker('fedsak', call_fedsak_worker)
