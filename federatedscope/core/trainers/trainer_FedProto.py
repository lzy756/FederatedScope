"""
FedProto Trainer Implementation

FedProto (Federated Prototypical Learning) is a personalized federated learning
method that learns class prototypes in the embedding space.

Key Features:
- Learns local prototypes for each class based on embeddings
- Aggregates prototypes on server to form global prototypes
- Uses prototype distances for classification and regularization

Reference:
FedProto: Federated Prototype Learning across Heterogeneous Clients
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.trainers.context import CtxVar
from federatedscope.register import register_trainer
import logging

logger = logging.getLogger(__name__)


class FedProtoTrainer(GeneralTorchTrainer):
    """
    FedProto Trainer for federated prototype learning

    This trainer extends GeneralTorchTrainer to support prototype-based
    personalized federated learning.
    """

    def __init__(self, model, data, device, config, only_for_eval=False, monitor=None):
        """
        Initialize FedProto trainer

        Args:
            model: The neural network model
            data: Training and evaluation data
            device: Computation device (cpu/cuda)
            config: Configuration object
            only_for_eval: Whether this trainer is only for evaluation
            monitor: Monitor for logging and tracking
        """
        super().__init__(model, data, device, config, only_for_eval, monitor)

        # FedProto-specific parameters
        self.proto_weight = config.fedproto.proto_weight
        self.embedding_dim = config.fedproto.embedding_dim
        self.use_projector = config.fedproto.use_projector
        self.distance_metric = config.fedproto.distance_metric
        self.temperature = config.fedproto.temperature
        self.normalize_prototypes = config.fedproto.normalize_prototypes
        self.freeze_backbone = config.fedproto.freeze_backbone

        # Get number of classes
        if hasattr(config.model, 'num_classes'):
            self.num_classes = config.model.num_classes
        else:
            # Try to infer from data
            if hasattr(data, 'num_classes'):
                self.num_classes = data['num_classes']
            else:
                self.num_classes = 10  # Default value
                logger.warning(f"Could not determine num_classes, using default: {self.num_classes}")

        # Initialize local prototypes (will be computed during training)
        self.local_prototypes = None

        # Global prototypes received from server
        self.global_prototypes = None

        # Build projector if needed
        if self.use_projector:
            self.projector = self._build_projector().to(device)
            if self.ctx.cfg.federate.share_local_model:
                # If sharing local model, projector params will be in get_model_para()
                pass
        else:
            self.projector = None

        logger.info(f"FedProto Trainer initialized with {self.num_classes} classes")
        logger.info(f"Proto weight: {self.proto_weight}, Distance metric: {self.distance_metric}")

    def _build_projector(self):
        """
        Build projection network for embeddings

        Returns:
            nn.Module: Projection network
        """
        hidden_dim = self.ctx.cfg.fedproto.projector_hidden_dim
        projector = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.embedding_dim)
        )
        logger.info(f"Built projector: {self.embedding_dim} -> {hidden_dim} -> {self.embedding_dim}")
        return projector

    def _extract_embeddings(self, model, x):
        """
        Extract embeddings from the model before the classification layer

        Args:
            model: Neural network model
            x: Input data

        Returns:
            torch.Tensor: Embeddings
        """
        # Try to extract embeddings based on model structure
        # Common pattern: model has a feature extractor + classifier

        # Priority 1: Use get_embedding() method if available (e.g., FedLSACNN)
        if hasattr(model, 'get_embedding'):
            embeddings = model.get_embedding(x)
        elif hasattr(model, 'features') and hasattr(model, 'classifier'):
            # Model with separate features and classifier (e.g., VGG-style)
            embeddings = model.features(x)
            embeddings = embeddings.view(embeddings.size(0), -1)
        elif hasattr(model, 'forward_embedding'):
            # Model with explicit embedding extraction method
            embeddings = model.forward_embedding(x)
        else:
            # Try to hook into the model to extract intermediate features
            # This is a general approach but may need customization
            embeddings = None

            def hook_fn(module, input, output):
                nonlocal embeddings
                embeddings = output

            # Try to find the last layer before classifier
            modules = list(model.children())
            if len(modules) > 0:
                # Register hook on the second-to-last module
                if len(modules) >= 2:
                    handle = modules[-2].register_forward_hook(hook_fn)
                    _ = model(x)
                    handle.remove()

                    if embeddings is not None:
                        # Flatten if needed
                        embeddings = embeddings.view(embeddings.size(0), -1)
                else:
                    # Fallback: use model output as embedding
                    logger.warning("Could not extract embeddings, using model output")
                    embeddings = model(x)

        if embeddings is None:
            raise RuntimeError("Failed to extract embeddings from model")

        # Apply projector if available
        if self.projector is not None:
            embeddings = self.projector(embeddings)

        return embeddings

    def _compute_prototypes(self, embeddings, labels):
        """
        Compute class prototypes from embeddings and labels

        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim)
            labels: Tensor of shape (batch_size,)

        Returns:
            torch.Tensor: Prototypes of shape (num_classes, embedding_dim)
        """
        prototypes = []
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                class_embeddings = embeddings[mask]
                prototype = class_embeddings.mean(dim=0)
                if self.normalize_prototypes:
                    prototype = F.normalize(prototype, p=2, dim=0)
                prototypes.append(prototype)
            else:
                # No samples for this class, use zero vector
                prototypes.append(torch.zeros(embeddings.size(1), device=embeddings.device))

        return torch.stack(prototypes)

    def _compute_distance(self, embeddings, prototypes):
        """
        Compute distance between embeddings and prototypes

        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim)
            prototypes: Tensor of shape (num_classes, embedding_dim)

        Returns:
            torch.Tensor: Distance matrix of shape (batch_size, num_classes)
        """
        if self.distance_metric == 'euclidean':
            # Euclidean distance: ||e - p||^2
            distances = torch.cdist(embeddings, prototypes, p=2)
        elif self.distance_metric == 'cosine':
            # Cosine similarity: dot(e, p) / (||e|| * ||p||)
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            prototypes_norm = F.normalize(prototypes, p=2, dim=1)
            # Similarity: higher is better
            similarity = torch.mm(embeddings_norm, prototypes_norm.t())
            # Convert to distance: lower is better
            distances = 1 - similarity
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        return distances

    def _compute_proto_loss(self, embeddings, labels, prototypes):
        """
        Compute prototype-based loss

        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim)
            labels: Tensor of shape (batch_size,)
            prototypes: Tensor of shape (num_classes, embedding_dim)

        Returns:
            torch.Tensor: Prototype loss (scalar)
        """
        if prototypes is None:
            return torch.tensor(0.0, device=embeddings.device)

        # Compute distances to all prototypes
        distances = self._compute_distance(embeddings, prototypes)

        # Contrastive loss: minimize distance to correct class, maximize to others
        batch_size = embeddings.size(0)

        # Distance to correct class prototype
        correct_distances = distances[torch.arange(batch_size), labels]

        # Use contrastive approach or triplet-like loss
        if self.distance_metric == 'euclidean':
            # L2 distance: minimize distance to correct class
            loss = correct_distances.mean()
        else:
            # For cosine, use temperature-scaled cross-entropy
            # Convert distances back to logits
            logits = -distances / self.temperature
            loss = F.cross_entropy(logits, labels)

        return loss

    def _hook_on_fit_start(self, ctx):
        """
        Clean up before training starts to prevent memory accumulation
        """
        # Clear any leftover embeddings from previous rounds
        if hasattr(ctx, 'embeddings_list'):
            del ctx.embeddings_list
        if hasattr(ctx, 'labels_list'):
            del ctx.labels_list

        # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _hook_on_batch_forward(self, ctx):
        """
        Override forward pass hook to incorporate prototype-based learning

        This is the key method where FedProto's custom logic is implemented.
        """
        x, label = [_.to(ctx.device) for _ in ctx.data_batch]

        # Extract embeddings
        embeddings = self._extract_embeddings(ctx.model, x)

        # Get predictions from full model
        pred = ctx.model(x)

        # Compute standard classification loss
        if len(label.size()) == 0:
            label = label.unsqueeze(0)
        loss_ce = ctx.criterion(pred, label)

        # Compute prototype-based loss during training
        proto_loss = torch.tensor(0.0, device=ctx.device)
        if ctx.cur_mode in [MODE.TRAIN] and self.global_prototypes is not None:
            proto_loss = self._compute_proto_loss(embeddings, label, self.global_prototypes)

        # Total loss
        total_loss = loss_ce + self.proto_weight * proto_loss

        # Store in context
        ctx.loss_batch = CtxVar(total_loss, LIFECYCLE.BATCH)
        ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(label), LIFECYCLE.BATCH)

        # Store embeddings for prototype computation
        if ctx.cur_mode == MODE.TRAIN:
            if not hasattr(ctx, 'embeddings_list'):
                ctx.embeddings_list = []
                ctx.labels_list = []
            # Move to CPU immediately to avoid GPU memory accumulation
            ctx.embeddings_list.append(embeddings.detach().cpu())
            ctx.labels_list.append(label.detach().cpu())

    def _hook_on_fit_end(self, ctx):
        """
        Compute local prototypes at the end of training
        """
        super()._hook_on_fit_end(ctx)

        # Compute local prototypes from collected embeddings
        if hasattr(ctx, 'embeddings_list') and len(ctx.embeddings_list) > 0:
            # Move data back to device for computation
            all_embeddings = torch.cat(ctx.embeddings_list, dim=0).to(ctx.device)
            all_labels = torch.cat(ctx.labels_list, dim=0).to(ctx.device)
            self.local_prototypes = self._compute_prototypes(all_embeddings, all_labels)

            logger.info(f"Computed local prototypes: shape={self.local_prototypes.shape}")

            # Properly clear the lists and free memory
            del ctx.embeddings_list
            del ctx.labels_list
            del all_embeddings
            del all_labels

            # Force garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def update_global_prototypes(self, global_prototypes):
        """
        Update global prototypes received from server

        Args:
            global_prototypes: Tensor of shape (num_classes, embedding_dim)
        """
        # Clear old prototypes first to free memory
        if self.global_prototypes is not None:
            del self.global_prototypes
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if global_prototypes is not None:
            self.global_prototypes = global_prototypes.to(self.ctx.device)
            logger.info(f"Updated global prototypes: shape={self.global_prototypes.shape}")
        else:
            self.global_prototypes = None
            logger.info("Cleared global prototypes")

    def get_local_prototypes(self):
        """
        Get local prototypes for aggregation

        Returns:
            torch.Tensor or None: Local prototypes
        """
        return self.local_prototypes

    def get_model_para(self):
        """
        Get model parameters including projector if shared

        Returns:
            dict: State dict with model and optionally projector parameters
        """
        # Get base model parameters
        model_para = super().get_model_para()

        # Add projector parameters if sharing is enabled
        if self.use_projector and self.projector is not None:
            if self.ctx.cfg.federate.share_local_model:
                # Include projector in shared parameters
                projector_para = {f'projector.{k}': v.cpu()
                                 for k, v in self.projector.state_dict().items()}
                model_para.update(projector_para)

        return model_para

    def update(self, model_parameters, strict=False):
        """
        Update model parameters including projector

        Args:
            model_parameters: Dictionary of parameters
            strict: Whether to strictly enforce parameter matching
        """
        # Separate projector parameters from model parameters
        if self.use_projector and self.projector is not None:
            projector_para = {}
            model_para = {}

            for key, value in model_parameters.items():
                if key.startswith('projector.'):
                    # Remove 'projector.' prefix
                    new_key = key[len('projector.'):]
                    projector_para[new_key] = value
                else:
                    model_para[key] = value

            # Update model
            if model_para:
                super().update(model_para, strict=strict)

            # Update projector
            if projector_para:
                self.projector.load_state_dict(projector_para, strict=strict)
                logger.debug("Updated projector parameters")
        else:
            # No projector, update model only
            super().update(model_parameters, strict=strict)


def call_fedproto_trainer(trainer_type):
    """
    Return FedProto trainer class if type matches

    Args:
        trainer_type: String specifying trainer type

    Returns:
        FedProtoTrainer class or None
    """
    if trainer_type.lower() == 'fedproto_trainer':
        return FedProtoTrainer
    return None


# Register the trainer
register_trainer('fedproto_trainer', call_fedproto_trainer)



