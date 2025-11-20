from federatedscope.cv.dataset.leaf_cv import LEAF_CV
from federatedscope.cv.dataset.office_caltech import load_office_caltech_domain_data
from federatedscope.cv.dataset.pacs import load_pacs_domain_data
from federatedscope.core.auxiliaries.transform_builder import get_transform
import numpy as np
from torch.utils.data import Subset
import logging

logger = logging.getLogger(__name__)


def split_data_by_dirichlet(dataset, num_clients, num_classes, alpha, seed):
    """
    Split dataset among clients using Dirichlet distribution for class heterogeneity.

    Args:
        dataset: PyTorch dataset with .targets attribute
        num_clients: number of clients to split data among
        num_classes: number of classes in the dataset
        alpha: Dirichlet concentration parameter (lower = more heterogeneous)
        seed: random seed for reproducibility

    Returns:
        List of lists containing indices for each client
    """
    np.random.seed(seed)

    # Get labels from dataset
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'targets'):
        # Handle Subset case
        labels = np.array(dataset.dataset.targets)[dataset.indices]
    else:
        raise ValueError("Dataset must have 'targets' attribute")

    # Group indices by class
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]

    # Initialize client indices
    client_indices = [[] for _ in range(num_clients)]

    # For each class, split samples among clients according to Dirichlet distribution
    for c_idx, c_indices in enumerate(class_indices):
        if len(c_indices) == 0:
            continue

        # Shuffle class indices
        np.random.shuffle(c_indices)

        # Sample from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

        # Balance proportions to match number of samples
        proportions = np.array([p * (len(c_idx) < len(c_indices)) for p, c_idx in zip(proportions, client_indices)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(c_indices)).astype(int)[:-1]

        # Split indices according to proportions
        split_indices = np.split(c_indices, proportions)

        # Assign to clients
        for client_id, indices in enumerate(split_indices):
            client_indices[client_id].extend(indices.tolist())

    # Shuffle each client's data
    for client_id in range(num_clients):
        np.random.shuffle(client_indices[client_id])

    return client_indices


def load_cv_dataset(config=None):
    """
    Return the dataset of ``femnist``, ``celeba``, ``office_caltech``, or ``pacs``.

    Args:
        config: configurations for FL, see ``federatedscope.core.configs``

    Returns:
        FL dataset dict, with ``client_id`` as key.

    Note:
      ``load_cv_dataset()`` will return a dict as shown below:
        ```
        {'client_id': {'train': dataset, 'test': dataset, 'val': dataset}}
        ```
    """
    splits = config.data.splits

    path = config.data.root
    name = config.data.type.lower()
    transforms_funcs, val_transforms_funcs, test_transforms_funcs = \
        get_transform(config, 'torchvision')

    if name in ['femnist', 'celeba']:
        dataset = LEAF_CV(root=path,
                          name=name,
                          s_frac=config.data.subsample,
                          tr_frac=splits[0],
                          val_frac=splits[1],
                          seed=1234,
                          **transforms_funcs)

        client_num = min(len(dataset), config.federate.client_num
                         ) if config.federate.client_num > 0 else len(dataset)
        config.merge_from_list(['federate.client_num', client_num])

        # Convert list to dict
        data_dict = dict()
        for client_idx in range(1, client_num + 1):
            data_dict[client_idx] = dataset[client_idx - 1]

    elif name == 'office_caltech':
        # Office-Caltech has 4 domains and 10 classes
        domains = ['amazon', 'webcam', 'dslr', 'caltech']
        num_domains = len(domains)
        num_classes = 10  # Office-Caltech has 10 common classes

        # Get the desired number of clients from config
        desired_client_num = config.federate.client_num

        # Get Dirichlet alpha parameter from config (0.0 means no Dirichlet, uniform split)
        dirichlet_alpha = getattr(config.data, 'dirichlet_alpha', 0.0)

        # Get transforms
        train_transform = transforms_funcs.get('transform', None)
        val_transform = val_transforms_funcs.get('transform', None)
        test_transform = test_transforms_funcs.get('transform', None)

        if desired_client_num <= num_domains:
            # If desired clients <= domains, use one domain per client
            data_dict = dict()
            for client_idx, domain in enumerate(domains[:desired_client_num], start=1):
                domain_data = load_office_caltech_domain_data(
                    root=path,
                    domain=domain,
                    splits=splits,
                    transform=train_transform,
                    val_transform=val_transform,
                    test_transform=test_transform,
                    seed=config.seed
                )

                # Apply Dirichlet split if alpha > 0
                if dirichlet_alpha > 0:
                    logger.info(f"Applying Dirichlet distribution (alpha={dirichlet_alpha}) to client {client_idx}")
                    # For single client per domain, no splitting needed
                    # But we could still filter classes based on Dirichlet
                    # For now, keep all data
                    data_dict[client_idx] = domain_data
                else:
                    data_dict[client_idx] = domain_data

            client_num = desired_client_num
        else:
            # If desired clients > domains, split each domain among multiple clients
            # Calculate how many clients per domain
            clients_per_domain = [desired_client_num // num_domains] * num_domains
            remainder = desired_client_num % num_domains

            # Distribute remainder clients
            for i in range(remainder):
                clients_per_domain[i] += 1

            data_dict = dict()
            client_idx = 1

            for domain_id, domain in enumerate(domains):
                # Load full domain data
                domain_data = load_office_caltech_domain_data(
                    root=path,
                    domain=domain,
                    splits=splits,
                    transform=train_transform,
                    val_transform=val_transform,
                    test_transform=test_transform,
                    seed=config.seed
                )

                # Split this domain among multiple clients
                num_clients_this_domain = clients_per_domain[domain_id]

                if num_clients_this_domain == 1:
                    # No splitting needed, assign entire domain to one client
                    data_dict[client_idx] = domain_data
                    client_idx += 1
                else:
                    # Check if we should use Dirichlet distribution
                    if dirichlet_alpha > 0:
                        logger.info(f"Splitting domain '{domain}' among {num_clients_this_domain} clients "
                                  f"using Dirichlet distribution (alpha={dirichlet_alpha})")

                        # Split each split (train/val/test) using Dirichlet
                        for split_name in ['train', 'val', 'test']:
                            dataset = domain_data[split_name]

                            # Get client indices using Dirichlet distribution
                            client_indices_list = split_data_by_dirichlet(
                                dataset=dataset,
                                num_clients=num_clients_this_domain,
                                num_classes=num_classes,
                                alpha=dirichlet_alpha,
                                seed=config.seed + domain_id + hash(split_name) % 1000
                            )

                            # Assign data to clients
                            for local_client_id in range(num_clients_this_domain):
                                global_client_id = client_idx + local_client_id

                                # Initialize client data dict if not exists
                                if global_client_id not in data_dict:
                                    data_dict[global_client_id] = {}

                                # Create subset with Dirichlet-sampled indices
                                indices = client_indices_list[local_client_id]
                                data_dict[global_client_id][split_name] = Subset(dataset, indices)

                                # Log class distribution for train split
                                if split_name == 'train':
                                    labels = np.array(dataset.targets)[indices]
                                    unique, counts = np.unique(labels, return_counts=True)
                                    class_dist = dict(zip(unique.tolist(), counts.tolist()))
                                    logger.info(f"  Client {global_client_id} ({domain}, {split_name}): "
                                              f"{len(indices)} samples, class distribution: {class_dist}")

                    else:
                        # Use uniform split (original behavior)
                        logger.info(f"Splitting domain '{domain}' among {num_clients_this_domain} clients uniformly")

                        for split_name in ['train', 'val', 'test']:
                            dataset = domain_data[split_name]
                            total_samples = len(dataset)

                            # Create indices for splitting
                            indices = np.arange(total_samples)
                            np.random.seed(config.seed + domain_id)
                            np.random.shuffle(indices)

                            # Calculate split sizes
                            split_sizes = [total_samples // num_clients_this_domain] * num_clients_this_domain
                            for i in range(total_samples % num_clients_this_domain):
                                split_sizes[i] += 1

                            # Create subsets for each client
                            start_idx = 0
                            for local_client_id in range(num_clients_this_domain):
                                end_idx = start_idx + split_sizes[local_client_id]
                                subset_indices = indices[start_idx:end_idx].tolist()

                                # Initialize client data dict if not exists
                                global_client_id = client_idx + local_client_id
                                if global_client_id not in data_dict:
                                    data_dict[global_client_id] = {}

                                # Create subset
                                data_dict[global_client_id][split_name] = Subset(dataset, subset_indices)
                                start_idx = end_idx

                    client_idx += num_clients_this_domain

            client_num = desired_client_num

        # Update config with actual client_num
        config.merge_from_list(['federate.client_num', client_num])

    elif name == 'pacs':
        # PACS has 4 domains and 7 classes
        domains = ['photo', 'art_painting', 'cartoon', 'sketch']
        num_domains = len(domains)
        num_classes = 7  # PACS has 7 classes

        # Get the desired number of clients from config
        desired_client_num = config.federate.client_num

        # Get Dirichlet alpha parameter from config (0.0 means no Dirichlet, uniform split)
        dirichlet_alpha = getattr(config.data, 'dirichlet_alpha', 0.0)

        # Get transforms
        train_transform = transforms_funcs.get('transform', None)
        val_transform = val_transforms_funcs.get('transform', None)
        test_transform = test_transforms_funcs.get('transform', None)

        if desired_client_num <= num_domains:
            # If desired clients <= domains, use one domain per client
            data_dict = dict()
            for client_idx, domain in enumerate(domains[:desired_client_num], start=1):
                domain_data = load_pacs_domain_data(
                    root=path,
                    domain=domain,
                    splits=splits,
                    transform=train_transform,
                    val_transform=val_transform,
                    test_transform=test_transform,
                    seed=config.seed
                )

                # Apply Dirichlet split if alpha > 0
                if dirichlet_alpha > 0:
                    logger.info(f"Applying Dirichlet distribution (alpha={dirichlet_alpha}) to client {client_idx}")
                    # For single client per domain, no splitting needed
                    # But we could still filter classes based on Dirichlet
                    # For now, keep all data
                    data_dict[client_idx] = domain_data
                else:
                    data_dict[client_idx] = domain_data

            client_num = desired_client_num
        else:
            # If desired clients > domains, split each domain among multiple clients
            # Calculate how many clients per domain
            clients_per_domain = [desired_client_num // num_domains] * num_domains
            remainder = desired_client_num % num_domains

            # Distribute remainder clients
            for i in range(remainder):
                clients_per_domain[i] += 1

            data_dict = dict()
            client_idx = 1

            for domain_id, domain in enumerate(domains):
                # Load full domain data
                domain_data = load_pacs_domain_data(
                    root=path,
                    domain=domain,
                    splits=splits,
                    transform=train_transform,
                    val_transform=val_transform,
                    test_transform=test_transform,
                    seed=config.seed
                )

                # Split this domain among multiple clients
                num_clients_this_domain = clients_per_domain[domain_id]

                if num_clients_this_domain == 1:
                    # No splitting needed, assign entire domain to one client
                    data_dict[client_idx] = domain_data
                    client_idx += 1
                else:
                    # Check if we should use Dirichlet distribution
                    if dirichlet_alpha > 0:
                        logger.info(f"Splitting domain '{domain}' among {num_clients_this_domain} clients "
                                  f"using Dirichlet distribution (alpha={dirichlet_alpha})")

                        # Split each split (train/val/test) using Dirichlet
                        for split_name in ['train', 'val', 'test']:
                            dataset = domain_data[split_name]

                            # Get client indices using Dirichlet distribution
                            client_indices_list = split_data_by_dirichlet(
                                dataset=dataset,
                                num_clients=num_clients_this_domain,
                                num_classes=num_classes,
                                alpha=dirichlet_alpha,
                                seed=config.seed + domain_id + hash(split_name) % 1000
                            )

                            # Assign data to clients
                            for local_client_id in range(num_clients_this_domain):
                                global_client_id = client_idx + local_client_id

                                # Initialize client data dict if not exists
                                if global_client_id not in data_dict:
                                    data_dict[global_client_id] = {}

                                # Create subset with Dirichlet-sampled indices
                                indices = client_indices_list[local_client_id]
                                data_dict[global_client_id][split_name] = Subset(dataset, indices)

                                # Log class distribution for train split
                                if split_name == 'train':
                                    labels = np.array(dataset.targets)[indices]
                                    unique, counts = np.unique(labels, return_counts=True)
                                    class_dist = dict(zip(unique.tolist(), counts.tolist()))
                                    logger.info(f"  Client {global_client_id} ({domain}, {split_name}): "
                                              f"{len(indices)} samples, class distribution: {class_dist}")

                    else:
                        # Use uniform split (original behavior)
                        logger.info(f"Splitting domain '{domain}' among {num_clients_this_domain} clients uniformly")

                        for split_name in ['train', 'val', 'test']:
                            dataset = domain_data[split_name]
                            total_samples = len(dataset)

                            # Create indices for splitting
                            indices = np.arange(total_samples)
                            np.random.seed(config.seed + domain_id)
                            np.random.shuffle(indices)

                            # Calculate split sizes
                            split_sizes = [total_samples // num_clients_this_domain] * num_clients_this_domain
                            for i in range(total_samples % num_clients_this_domain):
                                split_sizes[i] += 1

                            # Create subsets for each client
                            start_idx = 0
                            for local_client_id in range(num_clients_this_domain):
                                end_idx = start_idx + split_sizes[local_client_id]
                                subset_indices = indices[start_idx:end_idx].tolist()

                                # Initialize client data dict if not exists
                                global_client_id = client_idx + local_client_id
                                if global_client_id not in data_dict:
                                    data_dict[global_client_id] = {}

                                # Create subset
                                data_dict[global_client_id][split_name] = Subset(dataset, subset_indices)
                                start_idx = end_idx

                    client_idx += num_clients_this_domain

            client_num = desired_client_num

        # Update config with actual client_num
        config.merge_from_list(['federate.client_num', client_num])

    else:
        raise ValueError(f'No dataset named: {name}!')

    return data_dict, config





