"""
FedProto Client Worker Implementation

Handles client-side operations for FedProto:
1. Receive global prototypes from server
2. Train local model with prototype-based loss
3. Compute and send local prototypes to server
4. Send updated model parameters
"""

import logging
from federatedscope.core.workers import Client
from federatedscope.core.message import Message

logger = logging.getLogger(__name__)


class FedProtoClient(Client):
    """
    FedProto Client Implementation

    Extends the standard client to:
    1. Receive and store global prototypes from server
    2. Update trainer with global prototypes
    3. Extract local prototypes after training
    4. Send both model parameters and prototypes to server
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
                 *args,
                 **kwargs):
        super(FedProtoClient, self).__init__(
            ID=ID,
            server_id=server_id,
            state=state,
            config=config,
            data=data,
            model=model,
            device=device,
            strategy=strategy,
            *args,
            **kwargs
        )

        # Register callback for global prototypes
        self.register_handlers('global_prototypes',
                              self.callback_funcs_for_global_prototypes)

        logger.info(f"FedProtoClient {self.ID} initialized")

    def callback_funcs_for_global_prototypes(self, message: Message):
        """
        Callback to receive global prototypes from server

        Args:
            message: Message containing global prototypes
        """
        global_prototypes = message.content
        logger.info(f"Client {self.ID} received global prototypes: shape={global_prototypes.shape}")

        # Update trainer with global prototypes
        if hasattr(self.trainer, 'update_global_prototypes'):
            self.trainer.update_global_prototypes(global_prototypes)
            logger.debug(f"Client {self.ID} updated trainer with global prototypes")
        else:
            logger.warning(f"Client {self.ID} trainer does not support global prototypes")

    def callback_funcs_for_model_para(self, message: Message):
        """
        Override to handle model parameter updates

        This is called when server sends updated model parameters.
        Standard behavior, no changes needed.
        """
        # Call parent implementation
        super().callback_funcs_for_model_para(message)

    def callback_funcs_for_finish(self, message: Message):
        """
        Callback for training completion

        After local training:
        1. Extract local prototypes from trainer
        2. Package model parameters and prototypes
        3. Send to server
        """
        # Standard finish handling
        if message.content is not None:
            # This means we should update our model
            self.trainer.update(message.content, strict=self._cfg.federate.share_local_model)

        # Start local training
        sample_size, model_para, results = self.trainer.train()

        # Extract local prototypes from trainer
        local_prototypes = None
        if hasattr(self.trainer, 'get_local_prototypes'):
            local_prototypes = self.trainer.get_local_prototypes()
            if local_prototypes is not None:
                logger.info(f"Client {self.ID} computed local prototypes: "
                           f"shape={local_prototypes.shape}")
            else:
                logger.warning(f"Client {self.ID} failed to compute local prototypes")

        # Package content with both model and prototypes
        if local_prototypes is not None:
            content = {
                'model_para': model_para,
                'prototypes': local_prototypes.cpu(),
                'sample_count': sample_size
            }
        else:
            # Fallback: send only model parameters
            content = model_para
            logger.warning(f"Client {self.ID} sending model without prototypes")

        # Create message to send back to server
        self.comm_manager.send(
            Message(msg_type='model_para',
                   sender=self.ID,
                   receiver=[self.server_id],
                   state=self.state,
                   content=content)
        )

        logger.info(f"Client {self.ID} completed training round {self.state}, "
                   f"sent model and prototypes to server")
