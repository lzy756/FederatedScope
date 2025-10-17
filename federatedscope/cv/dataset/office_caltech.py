"""
Office-Caltech-10 Dataset for Domain Adaptation in Federated Learning

This dataset contains images from 4 domains:
- Amazon: images from amazon.com
- Webcam: low-resolution images from web camera
- DSLR: high-resolution images from digital SLR camera
- Caltech: images from Caltech-256

Reference:
    Gong, B., Shi, Y., Sha, F., & Grauman, K. (2012).
    Geodesic flow kernel for unsupervised domain adaptation.
    CVPR 2012.
"""

import os
import os.path as osp
import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


class OfficeCaltech(Dataset):
    """
    Office-Caltech-10 Dataset

    Args:
        root: root directory of the dataset
        domain: which domain to load ('amazon', 'webcam', 'dslr', 'caltech')
        split: 'train', 'val', or 'test'
        transform: transformation to apply to images
        train_ratio: ratio of training data (default: 0.8)
        val_ratio: ratio of validation data (default: 0.1)
        seed: random seed for splitting
    """

    # 10 common classes across all domains
    CLASSES = [
        'back_pack', 'bike', 'calculator', 'headphones', 'keyboard',
        'laptop_computer', 'monitor', 'mouse', 'mug', 'projector'
    ]

    DOMAINS = ['amazon', 'webcam', 'dslr', 'caltech']

    def __init__(self,
                 root,
                 domain='amazon',
                 split='train',
                 transform=None,
                 train_ratio=0.8,
                 val_ratio=0.1,
                 seed=123):

        assert domain in self.DOMAINS, f"Domain must be one of {self.DOMAINS}"
        assert split in ['train', 'val', 'test'], "Split must be train, val, or test"

        self.root = root
        self.domain = domain
        self.split = split
        self.transform = transform
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.seed = seed

        # Load data
        self.data, self.targets = self._load_data()

        logger.info(f"Loaded {len(self.data)} images from {domain} ({split} split)")

    def _load_data(self):
        """Load images and labels from the specified domain"""
        domain_dir = osp.join(self.root, self.domain)

        if not osp.exists(domain_dir):
            raise FileNotFoundError(
                f"Domain directory not found: {domain_dir}\n"
                f"Please download Office-Caltech-10 dataset first.\n"
                f"Run: python scripts/download_office_caltech.py"
            )

        all_images = []
        all_labels = []

        # Load images from each class
        for class_idx, class_name in enumerate(self.CLASSES):
            class_dir = osp.join(domain_dir, class_name)

            if not osp.exists(class_dir):
                logger.warning(f"Class directory not found: {class_dir}")
                continue

            # Get all image files
            image_files = [
                f for f in os.listdir(class_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]

            for img_file in image_files:
                img_path = osp.join(class_dir, img_file)
                all_images.append(img_path)
                all_labels.append(class_idx)

        if len(all_images) == 0:
            raise ValueError(f"No images found in {domain_dir}")

        # Split data
        np.random.seed(self.seed)
        indices = np.random.permutation(len(all_images))

        train_size = int(len(all_images) * self.train_ratio)
        val_size = int(len(all_images) * self.val_ratio)

        if self.split == 'train':
            indices = indices[:train_size]
        elif self.split == 'val':
            indices = indices[train_size:train_size + val_size]
        else:  # test
            indices = indices[train_size + val_size:]

        images = [all_images[i] for i in indices]
        labels = [all_labels[i] for i in indices]

        return images, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.targets[idx]

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), color='white')

        # Apply transform
        if self.transform is not None:
            image = self.transform(image)

        return image, label


def load_office_caltech_domain_data(root,
                                     domain,
                                     splits=(0.8, 0.1, 0.1),
                                     transform=None,
                                     val_transform=None,
                                     test_transform=None,
                                     seed=123):
    """
    Load Office-Caltech dataset for a single domain

    Args:
        root: root directory of the dataset
        domain: which domain to load
        splits: (train_ratio, val_ratio, test_ratio)
        transform: transform for training data
        val_transform: transform for validation data
        test_transform: transform for test data
        seed: random seed

    Returns:
        Dictionary with 'train', 'val', 'test' datasets
    """
    train_ratio, val_ratio, test_ratio = splits

    # Use default transforms if not provided
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    if val_transform is None:
        val_transform = transform
    if test_transform is None:
        test_transform = transform

    train_dataset = OfficeCaltech(
        root=root,
        domain=domain,
        split='train',
        transform=transform,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed
    )

    val_dataset = OfficeCaltech(
        root=root,
        domain=domain,
        split='val',
        transform=val_transform,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed
    )

    test_dataset = OfficeCaltech(
        root=root,
        domain=domain,
        split='test',
        transform=test_transform,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed
    )

    return {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }

