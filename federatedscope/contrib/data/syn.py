import os
import logging
from torchvision import transforms
from torchvision.datasets import ImageFolder
from federatedscope.core.data import BaseDataTranslator
from federatedscope.register import register_data

logger = logging.getLogger(__name__)


def load_syn_dataset(config, client_cfgs=None):
    """
    Load SYN (synthetic) dataset for federated learning.

    This function loads the synthetic digit dataset that is commonly used
    in cross-domain federated learning experiments. The dataset should be
    organized in ImageFolder format with train and val subdirectories.

    Args:
        config: Configuration object containing data settings
        client_cfgs: Client-specific configurations (optional)

    Returns:
        tuple: (datadict, config) where datadict contains the federated data
    """
    data_root = config.data.root
    syn_root = os.path.join(data_root, 'SYN')

    # Check if SYN dataset exists
    train_path = os.path.join(syn_root, 'imgs_train')
    val_path = os.path.join(syn_root, 'imgs_valid')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"SYN training data not found at {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"SYN validation data not found at {val_path}")

    logger.info(f"Loading SYN dataset from {syn_root}")

    # Define transforms similar to RethinkFL implementation
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Load datasets using ImageFolder
    train_dataset = ImageFolder(root=train_path, transform=transform)
    test_dataset = ImageFolder(root=val_path, transform=test_transform)

    logger.info(f"Full SYN dataset loaded: train={len(train_dataset)}, test={len(test_dataset)}")
    logger.info(f"Number of classes: {len(train_dataset.classes)}")
    logger.info(f"Classes: {train_dataset.classes}")

    # Use BaseDataTranslator to convert to federated format
    translator = BaseDataTranslator(config, client_cfgs)

    # Create dataset tuple (train, val, test) - using test for both val and test
    dataset_tuple = (train_dataset, test_dataset, test_dataset)

    # Apply translator
    datadict = translator(dataset_tuple)

    logger.info(f"Successfully created federated SYN dataset with {len(datadict)} clients")

    return datadict, config


def call_syn_data(config, client_cfgs=None):
    """
    Entry point for SYN dataset registration.

    Args:
        config: Configuration object
        client_cfgs: Client-specific configurations (optional)

    Returns:
        tuple: (datadict, config) if dataset type matches, None otherwise
    """
    if config.data.type.lower() == "syn":
        logger.info("Loading SYN dataset...")
        return load_syn_dataset(config, client_cfgs)
    return None


# Register the SYN dataset
register_data("syn", call_syn_data)