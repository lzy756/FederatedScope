"""
FedLSA (联邦可学习语义锚点) Client实现

核心功能:
1. 接收服务器学习的语义锚点
2. 将锚点传递给Trainer用于本地训练
3. 上传训练后的模型参数给服务器
"""
import logging
from federatedscope.core.workers import Client
from federatedscope.core.message import Message

logger = logging.getLogger(__name__)


class FedLSAClient(Client):
    """
    FedLSA方法的客户端实现

    客户端职责:
    1. 接收来自服务器的语义锚点 (msg_type='semantic_anchors')
    2. 将锚点更新到本地trainer
    3. 使用锚点指导进行本地训练 (超球面对比学习)
    4. 上传训练后的模型参数

    通信流程:
    Server → Client: 锚点 A + 全局模型 Φ_glo
    Client训练: 优化 L_HC = L_CE + λ·L_COM
    Client → Server: 本地模型 Φ_m
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
        super(FedLSAClient, self).__init__(
            ID=ID,
            server_id=server_id,
            state=state,
            config=config,
            data=data,
            model=model,
            device=device,
            strategy=strategy,
            is_unseen_client=is_unseen_client,
            *args,
            **kwargs
        )

        # 注册语义锚点处理器
        # 当收到'semantic_anchors'类型的消息时，调用callback_funcs_for_semantic_anchors
        self.register_handlers('semantic_anchors',
                              self.callback_funcs_for_semantic_anchors,
                              ['model_para'])

    def callback_funcs_for_semantic_anchors(self, message: Message):
        """
        处理从服务器接收的语义锚点

        处理流程:
        1. 从消息中提取锚点张量 [C, D]
        2. 调用trainer的update_semantic_anchors方法更新锚点
        3. Trainer在后续训练中使用锚点计算紧凑性损失 L_COM

        锚点的作用:
        - 指导客户端在超球面空间进行对比学习
        - 实现类内紧凑性 (将样本拉向对应的锚点)
        - 锚点由服务器学习，独立于客户端域

        Args:
            message: 包含语义锚点的消息对象
                    message.content: 锚点张量 [C, D]
        """
        content = message.content  # 语义锚点张量

        if content is not None:
            # 更新trainer中的语义锚点
            if hasattr(self.trainer, 'update_semantic_anchors'):
                self.trainer.update_semantic_anchors(content)
                logger.debug(f"客户端 #{self.ID} 接收到语义锚点，形状: {content.shape}")
            else:
                logger.warning(f"客户端 #{self.ID} 的trainer不支持语义锚点更新")


def call_fedlsa_worker(method):
    """
    FedLSA worker注册的辅助函数

    Args:
        method: 联邦学习方法名称

    Returns:
        包含'client'和'server'类的字典，如果方法不匹配则返回None
    """
    if method.lower() == 'fedlsa':
        from federatedscope.core.workers.server_FedLSA import FedLSAServer
        return {
            'client': FedLSAClient,
            'server': FedLSAServer
        }
    else:
        return None


# 注册FedLSA workers到FederatedScope框架
from federatedscope.register import register_worker
register_worker('fedlsa', call_fedlsa_worker)



