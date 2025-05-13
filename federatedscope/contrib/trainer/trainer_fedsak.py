# federatedscope/contrib/trainer/trainer_fedsak.py
from typing import List
import types
import logging
from federatedscope.register import register_trainer
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer

logger = logging.getLogger(__name__)


# ---------- 公共工具 ----------
def _share_patterns(cfg) -> List[str]:
    return getattr(getattr(cfg, "fedsak", None), "share_patterns", [
        "embeddings.position", "encoder.layer.9", "encoder.layer.10",
        "encoder.layer.11", "classifier"
    ])


def _param_filter(self, state, _=None):
    pats = self.ctx.fedsak_share_patterns
    return {k: v for k, v in state.items() if any(p in k for p in pats)}


# ---------- 真正的 Trainer 类 ----------
class FedSAKTrainer(GeneralTorchTrainer):
    """
    继承官方 GeneralTorchTrainer，仅改 2 处：
      • on_fit_start 时保存需要同步的参数名
      • 覆写 _param_filter 让通信只处理这些参数
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 1) 保存共享层匹配串
        self.ctx.fedsak_share_patterns = _share_patterns(self.cfg)
        logger.info(
            f"[FedSAK] share patterns = {self.ctx.fedsak_share_patterns}")

        # 2) Monkey‑patch 参数过滤
        self._param_filter = types.MethodType(_param_filter, self)


# ---------- 向框架注册 ----------
def fedsak_factory(trainer_type):
    return FedSAKTrainer if trainer_type.lower() == "fedsak" else None


register_trainer("fedsak", fedsak_factory)
