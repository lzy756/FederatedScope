import os
from torch.utils.data import Dataset
from PIL import Image
from scipy.io import loadmat
import numpy as np
from torchvision import transforms
from federatedscope.register import register_data
import logging
from federatedscope.core.data import BaseDataTranslator
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def load_data(path):
    """Helper function for loading a MAT-File"""
    data = loadmat(path)
    return data['X'], data['y']

def rgb2gray(rgb_images):
    """
    Convert RGB images to grayscale using standard weights
    RGB to grayscale conversion: 0.2989 * R + 0.5870 * G + 0.1140 * B
    """
    return np.dot(rgb_images[...,:3], [0.2989, 0.5870, 0.1140])

def load_svhn(data_root, transform=None):
    """
    Load SVHN dataset with preprocessing following the data_preprocess.ipynb notebook:
    1. Load from .mat files and transpose
    2. Convert label 10's to 0's
    3. Grayscale conversion
    4. Normalization
    5. Create stratified validation split (13%)
    """
    logger.info("Loading SVHN data from .mat files...")

    # Load training and test data
    X_train, y_train = load_data(os.path.join(data_root, 'SVHN/train_32x32.mat'))
    X_test, y_test = load_data(os.path.join(data_root, 'SVHN/test_32x32.mat'))

    logger.info(f"Original Training Set: {X_train.shape}, {y_train.shape}")
    logger.info(f"Original Test Set: {X_test.shape}, {y_test.shape}")

    # Transpose the image arrays from (width, height, channels, size) -> (size, width, height, channels)
    X_train, y_train = X_train.transpose((3,0,1,2)), y_train[:,0]
    X_test, y_test = X_test.transpose((3,0,1,2)), y_test[:,0]

    logger.info(f"After transpose - Training Set: {X_train.shape}")
    logger.info(f"After transpose - Test Set: {X_test.shape}")

    # Calculate total number of images
    num_images = X_train.shape[0] + X_test.shape[0]
    logger.info(f"Total Number of Images: {num_images}")

    # Converting the Label 10's to 0's (as mentioned in notebook)
    y_train[y_train == 10] = 0
    y_test[y_test == 10] = 0

    # Create stratified validation split (13% of training data)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.13, random_state=42, stratify=y_train
    )

    logger.info(f"After validation split:")
    logger.info(f"  Training Set: {X_train.shape}")
    logger.info(f"  Validation Set: {X_val.shape}")
    logger.info(f"  Test Set: {X_test.shape}")

    # Greyscale conversion for easy computation (as mentioned in notebook)
    logger.info("Converting to grayscale...")
    train_greyscale = rgb2gray(X_train)
    val_greyscale = rgb2gray(X_val)
    test_greyscale = rgb2gray(X_test)

    # Normalization (following notebook approach)
    logger.info("Normalizing data...")
    train_mean = np.mean(train_greyscale)
    train_std = np.std(train_greyscale)

    logger.info(f"Training mean: {train_mean}, Training std: {train_std}")

    # Normalize all splits using training statistics
    train_greyscale_norm = (train_greyscale - train_mean) / train_std
    val_greyscale_norm = (val_greyscale - train_mean) / train_std
    test_greyscale_norm = (test_greyscale - train_mean) / train_std

    class SVHNDataset(Dataset):
        def __init__(self, X, y, transform=None):
            self.X = X
            self.y = y
            self.transform = transform

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            # Convert grayscale to PIL Image
            img_array = (self.X[idx] * 255).astype(np.uint8)  # Convert back to 0-255 range
            img = Image.fromarray(img_array, mode='L')  # Grayscale mode

            if self.transform:
                img = self.transform(img)
            else:
                # Default transform: convert to tensor and normalize to [-1, 1]
                img = transforms.ToTensor()(img)
                img = transforms.Normalize(mean=[0.5], std=[0.5])(img)

            return img, int(self.y[idx])

    # Create datasets
    train_dataset = SVHNDataset(train_greyscale_norm, y_train, transform)
    val_dataset = SVHNDataset(val_greyscale_norm, y_val, transform)
    test_dataset = SVHNDataset(test_greyscale_norm, y_test, transform)

    logger.info("SVHN preprocessing completed successfully!")

    return train_dataset, val_dataset, test_dataset

def load_svhn_data(config, client_cfgs=None):
    data_root = config.data.root

    # Define transform for grayscale images (since we convert to grayscale in preprocessing)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Ensure 32x32 size
        transforms.ToTensor(),  # Convert to tensor [0,1]
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1,1] for grayscale
    ])

    logger.info("Loading SVHN dataset with preprocessing...")

    # Load preprocessed SVHN data (returns train, val, test)
    svhn_train, svhn_val, svhn_test = load_svhn(data_root, transform)

    # Use validation set as the middle split for FederatedScope format
    split_datasets_svhn = (svhn_train, svhn_val, svhn_test)

    translator = BaseDataTranslator(config, client_cfgs)
    data_dict = translator(split_datasets_svhn)

    logger.info(f"SVHN dataset loaded successfully:")
    logger.info(f"  Training samples: {len(svhn_train)}")
    logger.info(f"  Validation samples: {len(svhn_val)}")
    logger.info(f"  Test samples: {len(svhn_test)}")

    return data_dict, config


def call_svhn_data(config, client_cfgs=None):
    if config.data.type == "svhn":
        return load_svhn_data(config, client_cfgs)

register_data("svhn", call_svhn_data)