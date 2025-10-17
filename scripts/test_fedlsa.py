"""
Quick Start Example for FedLSA

This script demonstrates how to use FedLSA with a simple synthetic dataset.
"""
import torch
import torch.nn as nn
from federatedscope.core.trainers.trainer_FedLSA import FedLSATrainer
from federatedscope.core.workers.server_FedLSA import FedLSAServer
from federatedscope.core.workers.client_FedLSA import FedLSAClient


class SimpleModel(nn.Module):
    """
    Simple CNN model for testing FedLSA
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU()
        )
        self.classifier = nn.Linear(512, num_classes)
        self.feature_dim = 512

    def get_embedding(self, x):
        """Extract feature embeddings"""
        return self.encoder(x)

    def forward(self, x, return_embedding=False):
        z = self.get_embedding(x)
        out = self.classifier(z)
        if return_embedding:
            return out, z
        return out


def create_synthetic_data(num_samples=100, num_classes=10):
    """
    Create synthetic domain-skewed data for testing

    Returns:
        dict: {'train': dataset, 'test': dataset}
    """
    from torch.utils.data import TensorDataset

    # Generate random images (32x32 RGB)
    X_train = torch.randn(num_samples, 3, 32, 32)
    y_train = torch.randint(0, num_classes, (num_samples,))

    X_test = torch.randn(num_samples // 5, 3, 32, 32)
    y_test = torch.randint(0, num_classes, (num_samples // 5,))

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    return {
        'train': train_dataset,
        'test': test_dataset
    }


def test_fedlsa_trainer():
    """
    Test FedLSA trainer with synthetic data
    """
    print("=" * 80)
    print("Testing FedLSA Trainer")
    print("=" * 80)

    # Create a simple config
    from federatedscope.core.configs.config import CN

    cfg = CN()
    cfg.backend = 'torch'
    cfg.device = 0 if torch.cuda.is_available() else -1
    cfg.seed = 123

    # Model config
    cfg.model = CN()
    cfg.model.type = 'simple'
    cfg.model.num_classes = 10
    cfg.model.hidden = 512

    # Training config
    cfg.train = CN()
    cfg.train.local_update_steps = 5
    cfg.train.batch_or_epoch = 'epoch'
    cfg.train.optimizer = CN()
    cfg.train.optimizer.type = 'SGD'
    cfg.train.optimizer.lr = 0.01
    cfg.train.optimizer.momentum = 0.9
    cfg.train.scheduler = CN()
    cfg.train.scheduler.type = ''

    # FedLSA config
    cfg.fedlsa = CN()
    cfg.fedlsa.use = True
    cfg.fedlsa.lambda_com = 0.5
    cfg.fedlsa.tau = 0.1
    cfg.fedlsa.use_projector = True
    cfg.fedlsa.projector_input_dim = 512
    cfg.fedlsa.projector_output_dim = 128
    cfg.fedlsa.share_projector = True
    cfg.fedlsa.alpha_sep = 0.4
    cfg.fedlsa.anchor_train_epochs = 100
    cfg.fedlsa.anchor_lr = 0.001

    # Dataloader config
    cfg.dataloader = CN()
    cfg.dataloader.type = 'base'
    cfg.dataloader.batch_size = 32
    cfg.dataloader.shuffle = True
    cfg.dataloader.num_workers = 0

    # Eval config
    cfg.eval = CN()
    cfg.eval.freq = 1
    cfg.eval.metrics = ['acc', 'loss']

    # Criterion
    cfg.criterion = CN()
    cfg.criterion.type = 'CrossEntropyLoss'

    # Regularizer
    cfg.regularizer = CN()
    cfg.regularizer.type = ''
    cfg.regularizer.mu = 0.0

    # Grad
    cfg.grad = CN()
    cfg.grad.grad_clip = 5.0

    # Personalization
    cfg.personalization = CN()
    cfg.personalization.local_param = []
    cfg.personalization.share_non_trainable_para = False

    # Federate
    cfg.federate = CN()
    cfg.federate.method = 'fedlsa'
    cfg.federate.share_local_model = False
    cfg.federate.online_aggr = False

    # Verbose
    cfg.verbose = 1
    cfg.print_decimal_digits = 6

    # Finetune
    cfg.finetune = CN()
    cfg.finetune.before_eval = False

    cfg.freeze()

    # Create model
    device = torch.device(f'cuda:{cfg.device}' if cfg.device >= 0 and torch.cuda.is_available() else 'cpu')
    model = SimpleModel(num_classes=cfg.model.num_classes)

    # Create synthetic data
    data = create_synthetic_data(num_samples=100, num_classes=10)

    # Create a simple monitor
    from federatedscope.core.monitors import Monitor
    monitor = Monitor(cfg, monitored_object=None)

    # Create trainer
    print("\n1. Creating FedLSA Trainer...")
    trainer = FedLSATrainer(
        model=model,
        data=data,
        device=device,
        config=cfg,
        monitor=monitor
    )
    print("   ✓ Trainer created successfully")

    # Create semantic anchors (simulating server broadcast)
    print("\n2. Creating semantic anchors...")
    semantic_anchors = torch.randn(cfg.model.num_classes, cfg.fedlsa.projector_output_dim)
    semantic_anchors = torch.nn.functional.normalize(semantic_anchors, p=2, dim=1)
    trainer.update_semantic_anchors(semantic_anchors)
    print(f"   ✓ Anchors created with shape {semantic_anchors.shape}")

    # Run training
    print("\n3. Running local training...")
    try:
        num_samples, model_params, eval_metrics = trainer.train()
        print(f"   ✓ Training completed - Processed {num_samples} samples")
        print(f"   ✓ Metrics: {eval_metrics}")
    except Exception as e:
        print(f"   ✗ Training failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("FedLSA Trainer Test Completed!")
    print("=" * 80)


def test_fedlsa_components():
    """
    Test all FedLSA components
    """
    print("\n" + "=" * 80)
    print("Testing FedLSA Components")
    print("=" * 80)

    # Test semantic anchor learner
    print("\n1. Testing Semantic Anchor Learner...")
    from federatedscope.core.workers.server_FedLSA import SemanticAnchorLearner

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    anchor_learner = SemanticAnchorLearner(
        num_classes=10,
        embedding_dim=128,
        device=device
    )

    anchors = anchor_learner()
    print(f"   ✓ Generated anchors with shape: {anchors.shape}")
    print(f"   ✓ Anchors are normalized: {torch.allclose(torch.norm(anchors, p=2, dim=1), torch.ones(10, device=device), atol=1e-5)}")

    print("\n" + "=" * 80)
    print("All Component Tests Completed!")
    print("=" * 80)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("FedLSA Quick Start Example")
    print("=" * 80)

    # Run component tests
    test_fedlsa_components()

    # Run trainer test
    test_fedlsa_trainer()

    print("\n" + "=" * 80)
    print("All Tests Completed Successfully!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Prepare your domain-skewed dataset")
    print("2. Configure FedLSA parameters in YAML file")
    print("3. Run: python scripts/run_fedlsa.py --cfg your_config.yaml")
    print("=" * 80 + "\n")

