from federatedscope.register import register_regularizer, logger

try:
    from torch.nn import Module
    import torch
except ImportError:
    Module = object
    torch = None


REGULARIZER_NAME = "proximal_regularizer"


class ProximalRegularizer(Module):
    """Returns the norm of the specific weight update.

        Arguments:
            p (int): The order of norm.
            tensor_before: The original matrix or vector
            tensor_after: The updated matrix or vector

        Returns:
            Tensor: the norm of the given udpate.
    """
    def __init__(self):
        super(ProximalRegularizer, self).__init__()

    def forward(self, ctx, p=2):
        norm = 0.
        for w_init, w in zip(ctx.weight_init, ctx.model.parameters()):
            norm += torch.pow(torch.norm(w - w_init, p), p)
        proximal_loss = norm * 1. / float(p)
        # 添加调试日志，输出 proximal loss 数值
        # logger.info(f"[FedProx Debug] Proximal loss: {proximal_loss.item()}")
        return proximal_loss


def call_proximal_regularizer(type):
    if type == REGULARIZER_NAME:
        regularizer = ProximalRegularizer
        return regularizer


register_regularizer(REGULARIZER_NAME, call_proximal_regularizer)
