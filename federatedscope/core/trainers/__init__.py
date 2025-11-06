from federatedscope.core.trainers.base_trainer import BaseTrainer
from federatedscope.core.trainers.trainer import Trainer
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
from federatedscope.core.trainers.tf_trainer import GeneralTFTrainer
from federatedscope.core.trainers.trainer_multi_model import \
    GeneralMultiModelTrainer
from federatedscope.core.trainers.trainer_pFedMe import wrap_pFedMeTrainer
from federatedscope.core.trainers.trainer_Ditto import wrap_DittoTrainer
from federatedscope.core.trainers.trainer_FedEM import FedEMTrainer
from federatedscope.core.trainers.trainer_FedRep import wrap_FedRepTrainer
from federatedscope.core.trainers.trainer_simple_tuning import \
    wrap_Simple_tuning_Trainer
from federatedscope.core.trainers.context import Context
from federatedscope.core.trainers.trainer_fedprox import wrap_fedprox_trainer
from federatedscope.core.trainers.trainer_nbafl import wrap_nbafl_trainer, \
    wrap_nbafl_server

# Import FedLSA trainer to trigger registration
try:
    from federatedscope.core.trainers.trainer_FedLSA import FedLSATrainer
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(
        f'FedLSA trainer not available: {e}'
    )

# Import FedProto trainer to trigger registration
try:
    from federatedscope.core.trainers.trainer_FedProto import FedProtoTrainer
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(
        f'FedProto trainer not available: {e}'
    )

__all__ = [
    'Trainer', 'Context', 'GeneralTorchTrainer', 'GeneralMultiModelTrainer',
    'wrap_pFedMeTrainer', 'wrap_DittoTrainer', 'FedEMTrainer',
    'wrap_fedprox_trainer', 'wrap_nbafl_trainer', 'wrap_nbafl_server',
    'wrap_Simple_tuning_Trainer', 'wrap_FedRepTrainer', 'BaseTrainer',
    'GeneralTFTrainer', 'FedLSATrainer', 'FedProtoTrainer'
]




