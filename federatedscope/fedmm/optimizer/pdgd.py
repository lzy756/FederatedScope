import torch
from torch.optim import Optimizer


class PDOptimizer(Optimizer):
    """Primal-dual gradient descent optimizer used in FedMM."""

    def __init__(self, params, lr=0.01, mu=2.0):
        defaults = dict(lr=lr, mu=mu)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            mu = group['mu']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                vstar = state.get('vstar')
                if vstar is None:
                    vstar = torch.zeros_like(p.data)
                    state['vstar'] = vstar
                dual = state.get('dual')
                if dual is None:
                    dual = torch.zeros_like(p.data)
                    state['dual'] = dual
                update = p.grad + mu * (p.data - vstar) + dual
                p.data.add_(-lr * update)
        return loss

    def set_reference(self, reference_state, name_map):
        for group in self.param_groups:
            for p in group['params']:
                name = name_map[id(p)]
                tensor = reference_state[name]
                self.state[p]['vstar'] = tensor.detach().clone().to(p.device)

    def set_dual(self, dual_state, name_map):
        for group in self.param_groups:
            for p in group['params']:
                name = name_map[id(p)]
                tensor = dual_state.get(name)
                if tensor is None:
                    tensor = torch.zeros_like(p.data)
                self.state[p]['dual'] = tensor.detach().clone().to(p.device)
