import os
import logging
import torch
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision import transforms

from federatedscope.core.data.base_translator import BaseDataTranslator
from federatedscope.register import register_data

logger = logging.getLogger(__name__)


class MNISTM(VisionDataset):
    """MNIST-M Dataset for FederatedScope.

    This dataset loads MNIST-M data from pre-downloaded .pt files.
    """

    training_file = "mnist_m_train.pt"
    test_file = "mnist_m_test.pt"
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    def __init__(self, root, train=True, transform=None, target_transform=None):
        """Initialize MNIST-M dataset.

        Args:
            root (str): Root directory where data is stored
            train (bool): If True, load training data; otherwise load test data
            transform: Transform to apply to images
            target_transform: Transform to apply to targets
        """
        super(MNISTM, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train

        # Check if data files exist
        if not self._check_exists():
            raise RuntimeError(
                f"Dataset not found in {self.data_folder}. "
                f"Please ensure {self.training_file} and {self.test_file} are present."
            )

        # Load data
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        data_path = os.path.join(self.data_folder, data_file)
        logger.info(f"Loading MNIST-M data from: {data_path}")

        self.data, self.targets = torch.load(data_path)

        # Convert targets to long tensor if needed
        if not isinstance(self.targets, torch.LongTensor):
            self.targets = self.targets.long()

        logger.info(f"Loaded MNIST-M {'train' if train else 'test'} data: "
                   f"{len(self.data)} samples, {len(torch.unique(self.targets))} classes")

    def __getitem__(self, index):
        """Get item by index.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class
        """
        img, target = self.data[index], int(self.targets[index])

        # Convert tensor to PIL Image
        # MNIST-M images are RGB, so we need to handle the tensor format properly
        if isinstance(img, torch.Tensor):
            # If tensor is in CHW format, convert to HWC for PIL
            if img.dim() == 3 and img.shape[0] == 3:
                img = img.permute(1, 2, 0)
            img = img.numpy()

        # Ensure the image is in the right format for PIL
        if img.dtype != 'uint8':
            img = (img * 255).astype('uint8')

        img = Image.fromarray(img, mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        return len(self.data)

    @property
    def data_folder(self):
        """Return the folder containing the data files."""
        return os.path.join(self.root, 'mnist_m')

    @property
    def class_to_idx(self):
        """Return mapping from class names to indices."""
        return {cls: i for i, cls in enumerate(self.classes)}

    def _check_exists(self):
        """Check if data files exist."""
        train_path = os.path.join(self.data_folder, self.training_file)
        test_path = os.path.join(self.data_folder, self.test_file)
        return os.path.exists(train_path) and os.path.exists(test_path)

    def extra_repr(self):
        """Return extra representation string."""
        return f"Split: {'Train' if self.train else 'Test'}"


def load_mnist_m_dataset(config, client_cfgs=None):
    """Load MNIST-M dataset and convert to federated format.

    Args:
        config: Configuration object
        client_cfgs: Client-specific configurations (optional)

    Returns:
        tuple: (datadict, config) containing federated data and updated config
    """
    # Get data root from config
    data_root = config.data.root if hasattr(config.data, 'root') else 'data'

    logger.info(f"Loading MNIST-M dataset from: {data_root}")

    # Define transforms
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),  # 统一缩放到32x32
            # 将灰度图复制为3通道（如果是RGB则无变化）
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),  # 转为Tensor并归一化到[0,1]
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),  # 标准化到[-1,1]
        ]
    )

    try:
        # Load training and test datasets
        train_dataset = MNISTM(root=data_root, train=True, transform=transform)
        test_dataset = MNISTM(root=data_root, train=False, transform=transform)

        logger.info(f"Successfully loaded MNIST-M dataset:")
        logger.info(f"  Training samples: {len(train_dataset)}")
        logger.info(f"  Test samples: {len(test_dataset)}")
        logger.info(f"  Number of classes: {len(train_dataset.classes)}")

    except Exception as e:
        logger.error(f"Failed to load MNIST-M dataset: {e}")
        raise

    # Use BaseDataTranslator to convert to federated format
    translator = BaseDataTranslator(config, client_cfgs)

    # Create dataset tuple (train, val, test) - using test for both val and test
    dataset_tuple = (train_dataset, test_dataset, test_dataset)

    # Apply translator to create federated data
    federated_data = translator(dataset_tuple)

    logger.info(f"Successfully created federated MNIST-M dataset with {len(federated_data)} clients")

    return federated_data, config


def call_mnist_m_data(config, client_cfgs=None):
    """Entry point for MNIST-M dataset registration.

    Args:
        config: Configuration object
        client_cfgs: Client-specific configurations (optional)

    Returns:
        tuple: (datadict, config) if dataset type matches, None otherwise
    """
    if config.data.type.lower() == "mnist_m":
        logger.info("Loading MNIST-M dataset...")
        return load_mnist_m_dataset(config, client_cfgs)
    return None


# Register the MNIST-M dataset
register_data("mnist_m", call_mnist_m_data)