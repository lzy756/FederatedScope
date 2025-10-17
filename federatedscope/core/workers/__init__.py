from federatedscope.core.workers.base_worker import Worker
from federatedscope.core.workers.base_server import BaseServer
from federatedscope.core.workers.base_client import BaseClient
from federatedscope.core.workers.server import Server
from federatedscope.core.workers.client import Client

# Import FedLSA workers to trigger registration
try:
    from federatedscope.core.workers.client_FedLSA import FedLSAClient
    from federatedscope.core.workers.server_FedLSA import FedLSAServer
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(
        f'FedLSA workers not available: {e}'
    )

__all__ = ['Worker', 'BaseServer', 'BaseClient', 'Server', 'Client',
           'FedLSAClient', 'FedLSAServer']


