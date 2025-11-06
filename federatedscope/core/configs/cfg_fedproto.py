from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_fedproto_cfg(cfg):
    """
    Add FedProto-specific configuration options

    FedProto (Federated Prototypical Learning) learns class prototypes
    in the embedding space for personalized federated learning.

    Reference:
    FedProto: Federated Prototype Learning across Heterogeneous Clients
    """

    # Create FedProto configuration node
    cfg.fedproto = CN()

    # Enable FedProto method
    cfg.fedproto.use = False

    # Loss weight for prototype distance loss
    cfg.fedproto.proto_weight = 1.0

    # Embedding dimension for prototype learning
    cfg.fedproto.embedding_dim = 512

    # Whether to use a separate projection layer for embeddings
    cfg.fedproto.use_projector = False

    # Projector configuration (if use_projector is True)
    cfg.fedproto.projector_hidden_dim = 256

    # Distance metric: 'euclidean' or 'cosine'
    cfg.fedproto.distance_metric = 'euclidean'

    # Temperature for softmax when using cosine distance
    cfg.fedproto.temperature = 0.5

    # Server-side prototype aggregation method: 'mean' or 'weighted_mean'
    cfg.fedproto.aggregation_method = 'mean'

    # Number of local epochs for prototype update
    cfg.fedproto.local_proto_epochs = 1

    # Whether to normalize prototypes
    cfg.fedproto.normalize_prototypes = False

    # Whether to freeze backbone during prototype learning
    cfg.fedproto.freeze_backbone = False


# Register the configuration extension
register_config("fedproto", extend_fedproto_cfg)

