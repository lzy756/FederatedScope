import os
import logging
import random
import numpy as np
from typing import Tuple, List
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, SVHN, USPS, ImageFolder

from federatedscope.core.data.base_data import StandaloneDataDict, ClientData
from federatedscope.register import register_data

logger = logging.getLogger(__name__)


class MyDigits(torch.utils.data.Dataset):
    """Custom dataset wrapper for digit datasets (MNIST, USPS, SVHN)"""
    
    def __init__(self, root, train=True, transform=None, 
                 target_transform=None, download=True, data_name=None):
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.dataset = self._build_dataset()

    def _build_dataset(self):
        """Build the underlying dataset based on data_name"""
        if self.data_name == 'mnist':
            return MNIST(self.root, self.train, None, None, self.download)
        elif self.data_name == 'usps':
            # USPS data is stored in data/USPS/
            usps_path = os.path.join(self.root, 'USPS')
            return USPS(usps_path, self.train, None, None, self.download)
        elif self.data_name == 'svhn':
            # SVHN data is stored in data/SVHN/
            svhn_path = os.path.join(self.root, 'SVHN')
            split = 'train' if self.train else 'test'
            return SVHN(svhn_path, split, None, None, self.download)
        else:
            raise ValueError(f"Unknown data_name: {self.data_name}")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img, target = self.dataset[index]
        
        # Debug: print image info
        # print(f"Original image type: {type(img)}, shape: {getattr(img, 'shape', 'N/A')}")
        
        # Convert to PIL Image (preserve original format for transform pipeline)
        if isinstance(img, Image.Image):
            # Keep as PIL Image
            pass
        elif isinstance(img, np.ndarray):
            # Convert numpy to PIL Image
            if len(img.shape) == 2:  # HW grayscale
                img = Image.fromarray(img.astype(np.uint8), mode='L')
            elif len(img.shape) == 3:
                if img.shape[2] == 1:  # HW1
                    img = Image.fromarray(img.squeeze().astype(np.uint8), mode='L')
                else:  # Already HWC
                    img = Image.fromarray(img.astype(np.uint8))
            else:
                img = Image.fromarray(img.astype(np.uint8))
        else:
            # Convert tensor to PIL
            if hasattr(img, 'numpy'):
                img_np = img.numpy()
                if len(img_np.shape) == 2:  # HW
                    img = Image.fromarray(img_np.astype(np.uint8), mode='L')
                else:
                    img = Image.fromarray(img_np.astype(np.uint8))
            else:
                img_np = np.array(img)
                if len(img_np.shape) == 2:  # HW
                    img = Image.fromarray(img_np.astype(np.uint8), mode='L')
                else:
                    img = Image.fromarray(img_np.astype(np.uint8))

        # Apply transforms
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.dataset)


class ImageFolderCustom(torch.utils.data.Dataset):
    """Custom ImageFolder for synthetic digit dataset"""
    
    def __init__(self, data_name, root, train=True, transform=None, target_transform=None):
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        
        # Build path for synthetic data
        # SYN data is stored in data/SYN/imgs_train and data/SYN/imgs_valid
        if train:
            split_dir = 'imgs_train'
        else:
            split_dir = 'imgs_valid'  # Note: using imgs_valid instead of val
        data_path = os.path.join(root, 'SYN', split_dir)
        
        if os.path.exists(data_path):
            self.imagefolder_obj = ImageFolder(data_path, transform, target_transform)
        else:
            logger.warning(f"Synthetic data path {data_path} not found. Creating empty dataset.")
            self.imagefolder_obj = None

    def __getitem__(self, index):
        if self.imagefolder_obj is None:
            raise RuntimeError("Synthetic dataset not available")
        return self.imagefolder_obj[index]

    def __len__(self):
        if self.imagefolder_obj is None:
            return 0
        return len(self.imagefolder_obj)


def partition_digits_domain_skew(train_datasets: List, test_datasets: List, 
                                num_clients: int, percent_dict: dict = None, alpha: float = 0.1, config=None):
    """
    Partition digits datasets for federated learning with domain skew.
    This implementation matches RethinkFL exactly:
    - Training sets: Each client gets a subset from one domain based on percent_dict
    - Test sets: All domains' complete test sets (shared by all clients for evaluation)
    
    Args:
        train_datasets: List of training datasets for each domain
        test_datasets: List of test datasets for each domain  
        num_clients: Total number of clients
        percent_dict: Dictionary specifying the percentage of data each client uses from each domain
        alpha: Dirichlet alpha parameter (not used in current implementation)
    
    Returns:
        train_loaders: List of training datasets for each client
        test_loaders: List of complete test datasets for each domain (for global evaluation)
    """
    train_loaders = []
    test_loaders = []
    
    # Default percent_dict if not provided (matching RethinkFL)
    if percent_dict is None:
        percent_dict = {'mnist': 0.01, 'usps': 0.01, 'svhn': 0.01, 'syn': 0.01}
    
    # Domain names mapping
    domain_names = ['mnist', 'usps', 'svhn', 'syn']
    
    logger.info(f"Using percent_dict: {percent_dict}")
    
    # === TRAINING SET PARTITIONING (per client) ===
    # Track unused indices for each domain to avoid overlap between clients
    not_used_index_dict = {}
    ini_len_dict = {}
    
    # Initialize unused indices for each domain
    for domain_idx, train_dataset in enumerate(train_datasets):
        domain_name = domain_names[domain_idx] if domain_idx < len(domain_names) else f'domain_{domain_idx}'
        
        # Get dataset size
        if hasattr(train_dataset, 'dataset'):
            # For MyDigits wrapper
            if hasattr(train_dataset.dataset, 'targets'):
                train_size = len(train_dataset.dataset.targets)
            elif hasattr(train_dataset.dataset, 'labels'):
                train_size = len(train_dataset.dataset.labels)
            else:
                train_size = len(train_dataset.dataset)
        elif hasattr(train_dataset, 'imagefolder_obj'):
            # For ImageFolderCustom (synthetic data)
            train_size = len(train_dataset.imagefolder_obj.targets) if train_dataset.imagefolder_obj else 0
        else:
            train_size = len(train_dataset)
            
        not_used_index_dict[domain_name] = np.arange(train_size)
        ini_len_dict[domain_name] = train_size
        np.random.shuffle(not_used_index_dict[domain_name])  # Shuffle indices
        
        logger.info(f"Domain {domain_name}: {train_size} total training samples")
    
    # Create training loaders for each client
    # Match RethinkFL's random domain allocation strategy
    
    # Check if we should use random domain allocation (like RethinkFL)
    use_random_domain_allocation = True
    
    # Define domain names
    domains = ['mnist', 'usps', 'svhn', 'syn']
    domain_names = domains  # Use same names
    
    if use_random_domain_allocation:
        # RethinkFL-style random domain allocation
        max_clients_per_domain = 10
        is_valid_allocation = False
        
        # Try to get a valid random allocation (constraint: max 10 clients per domain)
        for attempt in range(100):  # Max attempts to avoid infinite loop
            # Randomly select domains for each client
            selected_domains = np.random.choice(domains, size=num_clients, replace=True)
            
            # Count clients per domain
            from collections import Counter
            domain_counts = Counter(selected_domains)
            
            # Check if allocation is valid (no domain has more than 10 clients)
            if all(count <= max_clients_per_domain for count in domain_counts.values()):
                is_valid_allocation = True
                break
        
        if not is_valid_allocation:
            logger.warning("Could not find valid random allocation, using uniform distribution")
            # Fallback to uniform distribution
            clients_per_domain = num_clients // len(domains)
            remainder = num_clients % len(domains)
            selected_domains = []
            for i, domain in enumerate(domains):
                count = clients_per_domain + (1 if i < remainder else 0)
                selected_domains.extend([domain] * count)
            selected_domains = np.array(selected_domains)
            domain_counts = Counter(selected_domains)
        
        logger.info(f"Random domain allocation: {dict(domain_counts)}")
        
        # Create client allocation based on random selection
        client_domain_map = {}
        domain_client_indices = {domain: [] for domain in domains}
        
        for client_idx, domain in enumerate(selected_domains):
            client_domain_map[client_idx] = domain
            domain_client_indices[domain].append(client_idx)
            
    else:
        # Original uniform allocation
        num_domains = len(train_datasets)
        clients_per_domain = num_clients // num_domains
        remaining_clients = num_clients % num_domains
        
        client_allocation = [clients_per_domain] * num_domains
        for i in range(remaining_clients):
            client_allocation[i] += 1
        
        logger.info(f"Uniform client allocation per domain: {client_allocation}")
        
        # Create domain mapping for uniform allocation
        client_domain_map = {}
        domain_client_indices = {domains[i]: [] for i in range(len(domains))}
        
        client_idx = 0
        for domain_idx in range(len(domains)):
            domain = domains[domain_idx]
            for _ in range(client_allocation[domain_idx]):
                client_domain_map[client_idx] = domain
                domain_client_indices[domain].append(client_idx)
                client_idx += 1
    
    # Assign training data to each client based on domain mapping
    for domain_idx, train_dataset in enumerate(train_datasets):
        if domain_idx < len(domains):
            domain_name = domains[domain_idx]
        else:
            continue  # Skip if we don't have this domain in our list
        domain_clients_list = domain_client_indices.get(domain_name, [])
        domain_clients = len(domain_clients_list)
        
        if domain_clients == 0:
            continue
        
        # Get percentage for this domain
        percent = percent_dict.get(domain_name, 0.01)
        samples_per_client = int(percent * ini_len_dict[domain_name])
        
        logger.info(f"Domain {domain_name}: {samples_per_client} samples per client (percent={percent}), {domain_clients} clients")
        
        # Create training subset for each client in this domain
        # Note: we need to maintain the order based on client IDs
        temp_client_data = {}
        
        for i in range(domain_clients):
            # Get indices for this client
            if len(not_used_index_dict[domain_name]) >= samples_per_client:
                selected_train_idx = not_used_index_dict[domain_name][:samples_per_client]
                not_used_index_dict[domain_name] = not_used_index_dict[domain_name][samples_per_client:]
            else:
                # If not enough unused indices, use all remaining
                selected_train_idx = not_used_index_dict[domain_name]
                not_used_index_dict[domain_name] = np.array([])
            
            # Create training subset for this client
            train_subset = torch.utils.data.Subset(train_dataset, selected_train_idx)
            
            # Map to actual client ID
            actual_client_id = domain_clients_list[i]
            temp_client_data[actual_client_id] = train_subset
            
            logger.info(f"Client {actual_client_id + 1}: {len(selected_train_idx)} samples from {domain_name}")
        
        # Store the client data temporarily
        if not hasattr(partition_digits_domain_skew, '_temp_client_data'):
            partition_digits_domain_skew._temp_client_data = {}
        partition_digits_domain_skew._temp_client_data.update(temp_client_data)
    
    # Convert temp client data to ordered list
    if hasattr(partition_digits_domain_skew, '_temp_client_data'):
        for client_id in range(num_clients):
            if client_id in partition_digits_domain_skew._temp_client_data:
                train_loaders.append(partition_digits_domain_skew._temp_client_data[client_id])
        # Clean up temporary data
        delattr(partition_digits_domain_skew, '_temp_client_data')
    
    # === TEST SET CREATION (per domain, matching RethinkFL) ===
    # In RethinkFL, test sets are created for each domain (not per client)
    # All clients share the same test sets for evaluation
    
    for domain_idx, test_dataset in enumerate(test_datasets):
        if domain_idx < len(domains):
            domain_name = domains[domain_idx]
        else:
            continue
        
        # Add complete test dataset for this domain
        test_loaders.append(test_dataset)
        
        test_size = len(test_dataset)
        logger.info(f"Test set for domain {domain_name}: {test_size} samples")
    
    logger.info(f"Created {len(train_loaders)} training sets for clients")
    logger.info(f"Created {len(test_loaders)} test sets for domains")
    
    return train_loaders, test_loaders


def load_fl_digits_data(config, client_cfgs=None):
    """
    Load FL-Digits dataset for FederatedScope.
    
    Args:
        config: FederatedScope configuration
        client_cfgs: Client-specific configurations
    
    Returns:
        data: StandaloneDataDict containing client data
        config: Updated configuration
    """
    logger.info("Loading FL-Digits dataset...")
    
    # Set random seeds for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # Set default data root to the data directory if not specified
    if hasattr(config.data, 'root') and config.data.root:
        data_root = config.data.root
    else:
        # Default to the data directory in FederatedScope root
        data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
    
    num_clients = config.federate.client_num
    
    logger.info(f"Using data root: {data_root}")
    
    # Define domains and their sampling percentages
    domains = ['mnist', 'usps', 'svhn', 'syn']
    
    # Default percent_dict - each client uses 1% of each domain's data
    # This can be customized based on requirements
    percent_dict = {'mnist': 0.01, 'usps': 0.01, 'svhn': 0.01, 'syn': 0.01}
        
    logger.info(f"Domain sampling percentages: {percent_dict}")
    
    # Define transforms - unified since we now convert all images to RGB in __getitem__
    # RethinkFL-style channel expansion transform
    def expand_channels(x):
        """Convert grayscale (1 channel) to RGB (3 channels) like RethinkFL"""
        if x.size(0) == 1:
            return x.repeat(3, 1, 1)
        return x
    
    # Training transform (for all datasets) - matches RethinkFL
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(expand_channels),  # RethinkFL-style channel expansion
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    ])
    
    # Test transform (for all datasets) - matches RethinkFL 
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(expand_channels),  # RethinkFL-style channel expansion
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Load datasets
    train_datasets = []
    test_datasets = []
    
    for domain in domains:
        logger.info(f"Loading domain: {domain}")
        
        if domain == 'syn':
            # Synthetic dataset - check if SYN directory exists
            syn_path = os.path.join(data_root, 'SYN')
            if not os.path.exists(syn_path):
                logger.warning(f"Synthetic dataset path {syn_path} not found, skipping {domain}")
                continue
                
            try:
                train_dataset = ImageFolderCustom(
                    data_name=domain, 
                    root=data_root, 
                    train=True,
                    transform=train_transform
                )
                test_dataset = ImageFolderCustom(
                    data_name=domain, 
                    root=data_root, 
                    train=False,
                    transform=test_transform
                )
            except Exception as e:
                logger.warning(f"Failed to load synthetic dataset: {e}")
                continue
        else:
            # Real datasets (MNIST, USPS, SVHN) - use unified transforms
            # Check if dataset exists before loading
            domain_path = os.path.join(data_root, domain.upper())
            if not os.path.exists(domain_path):
                logger.warning(f"Dataset path {domain_path} not found, skipping {domain}")
                continue
                
            train_dataset = MyDigits(
                root=data_root,
                train=True,
                download=False,  # Don't download, use existing data
                transform=train_transform,
                data_name=domain
            )
            test_dataset = MyDigits(
                root=data_root,
                train=False,
                download=False,  # Don't download, use existing data
                transform=test_transform,
                data_name=domain
            )
        
        if len(train_dataset) > 0 and len(test_dataset) > 0:
            train_datasets.append(train_dataset)
            test_datasets.append(test_dataset)
            logger.info(f"Loaded {domain}: train={len(train_dataset)}, test={len(test_dataset)}")
        else:
            logger.warning(f"Skipping {domain} due to empty dataset")
    
    if not train_datasets:
        raise ValueError("No datasets were successfully loaded!")
    
    # Partition datasets for federated learning
    train_data_list, test_data_list = partition_digits_domain_skew(
        train_datasets, test_datasets, num_clients, percent_dict, config=config
    )
    
    # Create ClientData for each client
    data_dict = {}
    
    # === SERVER DATA (for global evaluation, matching RethinkFL) ===
    # Match RethinkFL: Use domain-wise evaluation instead of merged test set
    # Create a domain evaluation setup where each domain is evaluated separately
    
    # Create a custom dataset that maintains domain separation for evaluation
    class DomainEvalDataset(torch.utils.data.Dataset):
        def __init__(self, domain_datasets, domain_names):
            self.domain_datasets = domain_datasets
            self.domain_names = domain_names
            self.domain_sizes = [len(ds) for ds in domain_datasets]
            self.total_size = sum(self.domain_sizes)
            
            # Create domain mapping
            self.domain_indices = []
            cumulative = 0
            for i, size in enumerate(self.domain_sizes):
                self.domain_indices.extend([(i, j) for j in range(size)])
                cumulative += size
        
        def __getitem__(self, idx):
            domain_idx, sample_idx = self.domain_indices[idx]
            sample, target = self.domain_datasets[domain_idx][sample_idx]
            # Add domain information as metadata (for potential domain-wise evaluation)
            return sample, target, domain_idx
        
        def __len__(self):
            return self.total_size
        
        def get_domain_data(self, domain_idx):
            """Get all samples from a specific domain"""
            domain_samples = []
            for i in range(len(self.domain_datasets[domain_idx])):
                sample, target = self.domain_datasets[domain_idx][i]
                domain_samples.append((sample, target))
            return domain_samples
    
    # Create domain evaluation dataset
    server_domain_eval = DomainEvalDataset(test_data_list, domains)
    
    # Use full test set by concatenating all domains (no balancing)
    server_test = data.ConcatDataset(test_data_list)
    
    # Create server ClientData with full test set
    data_dict[0] = ClientData(config, test=server_test)
    
    logger.info(f"Server data created with full test set: {len(server_test)} samples")
    # === CLIENT DATA (training + complete test set, matching server) ===
    # Each client has training data from one domain + complete test set from all domains
    # This ensures all clients can perform comprehensive evaluation
    
    for client_id in range(num_clients):
        if client_id < len(train_data_list):
            client_train = train_data_list[client_id]
            
            # Give each client the same complete test set as the server
            # This includes all domains' test data for comprehensive evaluation
            client_data_dict = {
                'train': client_train,
                'test': server_test  # Same complete test set as server
            }
            
            data_dict[client_id + 1] = ClientData(config, **client_data_dict)
            
            logger.info(f"Client {client_id + 1}: train={len(client_train)} samples, test={len(server_test)} samples (all domains)")
        else:
            logger.warning(f"No training data for client {client_id + 1}")
    
    # Create StandaloneDataDict
    standalone_data = StandaloneDataDict(data_dict, config)
    
    logger.info(f"FL-Digits dataset loaded successfully:")
    logger.info(f"  - {len(data_dict) - 1} clients with training data + complete test set")
    logger.info(f"  - Server with complete test set for global evaluation")
    logger.info(f"  - Total training samples: {sum(len(train_data_list[i]) for i in range(len(train_data_list)))}")
    logger.info(f"  - Total test samples per client/server: {sum(len(test_data_list[i]) for i in range(len(test_data_list)))}")
    
    return standalone_data, config


def call_fl_digits_data(config, client_cfgs=None):
    """Entry point for FL-Digits dataset registration"""
    if config.data.type.lower() == "fl_digits":
        data, modified_config = load_fl_digits_data(config, client_cfgs)
        return data, modified_config
    return None


# Register the dataset with FederatedScope
register_data("fl_digits", call_fl_digits_data)
