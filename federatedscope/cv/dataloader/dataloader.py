from federatedscope.cv.dataset.leaf_cv import LEAF_CV
from federatedscope.cv.dataset.office_caltech import load_office_caltech_domain_data
from federatedscope.cv.dataset.pacs import load_pacs_domain_data
from federatedscope.cv.dataset.office_home import load_office_home_domain_data
from federatedscope.core.auxiliaries.transform_builder import get_transform
import numpy as np
from torch.utils.data import Subset
import logging

logger = logging.getLogger(__name__)


def apply_label_distribution_skew(dataset, dirichlet_proportions, num_classes, seed=None):
    """
    Apply Label Distribution Skew (LDS) to a dataset using Dirichlet proportions.

    For each class, sample a proportion of samples based on the Dirichlet distribution.
    This creates highly imbalanced class distributions where some classes have many samples
    and others have few or zero samples.

    Args:
        dataset: PyTorch dataset with .targets attribute
        dirichlet_proportions: array of length num_classes with proportions for each class (should sum to ~1.0)
        num_classes: number of classes in the dataset
        seed: random seed for reproducibility

    Returns:
        List of indices to keep for this client
    """
    if seed is not None:
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

    # Sample indices based on Dirichlet proportions
    sampled_indices = []
    class_stats = {}

    for class_idx, indices in enumerate(class_indices):
        if len(indices) == 0:
            class_stats[class_idx] = 0
            continue

        # Get proportion for this class
        proportion = dirichlet_proportions[class_idx]

        # Calculate number of samples to keep
        num_to_keep = int(proportion * len(indices))

        # Shuffle and sample
        np.random.shuffle(indices)
        sampled = indices[:num_to_keep]

        sampled_indices.extend(sampled.tolist())
        class_stats[class_idx] = len(sampled)

    logger.info(f"  LDS sampling: {len(sampled_indices)} total samples from {num_classes} classes")
    logger.info(f"  Class distribution: min={min(class_stats.values())}, max={max(class_stats.values())}, "
                f"classes_with_0_samples={sum(1 for v in class_stats.values() if v == 0)}")

    return sampled_indices


def split_data_by_dirichlet(dataset, num_clients, num_classes, alpha, seed, min_samples_per_client=5):
    """
    Split dataset among clients using Dirichlet distribution for class heterogeneity.
    Ensures each client gets at least min_samples_per_client samples.

    Args:
        dataset: PyTorch dataset with .targets attribute
        num_clients: number of clients to split data among
        num_classes: number of classes in the dataset
        alpha: Dirichlet concentration parameter (lower = more heterogeneous)
        seed: random seed for reproducibility
        min_samples_per_client: minimum number of samples each client must have (default: 5)

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

    # Post-processing: Ensure minimum samples per client
    if min_samples_per_client > 0:
        # Find clients with insufficient samples
        clients_need_samples = []
        clients_have_extra = []

        for client_id in range(num_clients):
            if len(client_indices[client_id]) < min_samples_per_client:
                clients_need_samples.append(client_id)
            elif len(client_indices[client_id]) > min_samples_per_client * 2:
                # Only consider clients with more than 2x minimum as donors
                clients_have_extra.append(client_id)

        if clients_need_samples:
            logger.info(f"Found {len(clients_need_samples)} clients with < {min_samples_per_client} samples")

            # Redistribute samples from clients with extra to clients with insufficient
            for client_id in clients_need_samples:
                needed = min_samples_per_client - len(client_indices[client_id])

                if needed > 0:
                    # Try to take samples from clients with extra
                    samples_collected = []

                    for donor_id in clients_have_extra:
                        if len(samples_collected) >= needed:
                            break

                        donor_samples = client_indices[donor_id]
                        # Calculate how many we can take (leave donor with at least min_samples_per_client)
                        can_donate = max(0, len(donor_samples) - min_samples_per_client)
                        to_take = min(can_donate, needed - len(samples_collected))

                        if to_take > 0:
                            # Take samples from the end of donor's list
                            taken = donor_samples[-to_take:]
                            client_indices[donor_id] = donor_samples[:-to_take]
                            samples_collected.extend(taken)

                    # Add collected samples to the client
                    client_indices[client_id].extend(samples_collected)

                    logger.info(f"  Client {client_id}: {len(client_indices[client_id]) - len(samples_collected)} -> {len(client_indices[client_id])} samples (added {len(samples_collected)})")

            # Final check: if still some clients don't have enough, warn
            still_insufficient = [cid for cid in range(num_clients)
                                 if len(client_indices[cid]) < min_samples_per_client]
            if still_insufficient:
                logger.warning(f"WARNING: {len(still_insufficient)} clients still have < {min_samples_per_client} samples after redistribution")
                logger.warning(f"  Insufficient clients: {still_insufficient}")
                for cid in still_insufficient:
                    logger.warning(f"    Client {cid}: {len(client_indices[cid])} samples")

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

        # Load server held-out test indices to prevent data leakage
        # This ensures client train/val/test data excludes server test samples
        excluded_indices_per_domain = {}
        if config.federate.make_global_eval:
            try:
                from federatedscope.cv.dataset.office_caltech import load_balanced_office_caltech_data

                # Support both fixed samples_per_class and ratio-based test_ratio
                samples_per_class_cfg = getattr(config.data, 'server_test_samples_per_class', 0)
                test_ratio_cfg = getattr(config.data, 'server_test_ratio', 0.1)

                # Determine which sampling mode to use
                if samples_per_class_cfg > 0:
                    samples_per_class = samples_per_class_cfg
                    test_ratio = None
                    logger.info(f"Loading server held-out test set (samples_per_class={samples_per_class}) to prevent data leakage...")
                else:
                    samples_per_class = None
                    test_ratio = test_ratio_cfg if test_ratio_cfg > 0 else 0.1
                    logger.info(f"Loading server held-out test set (test_ratio={test_ratio}) to prevent data leakage...")

                result = load_balanced_office_caltech_data(
                    root=path,
                    samples_per_class=samples_per_class,
                    test_ratio=test_ratio,
                    transform=None,
                    seed=config.seed
                )
                excluded_indices_per_domain = result['excluded_indices']
                logger.info(f"Server held-out indices loaded for {len(excluded_indices_per_domain)} domains")
                for domain, indices in excluded_indices_per_domain.items():
                    logger.info(f"  Domain '{domain}': {len(indices)} samples excluded from client data")
            except Exception as e:
                logger.warning(f"Failed to load server held-out indices: {e}")
                logger.warning("Proceeding without data exclusion - may have data leakage!")
                excluded_indices_per_domain = {}

        if desired_client_num <= num_domains:
            # If desired clients <= domains, use one domain per client
            data_dict = dict()
            for client_idx, domain in enumerate(domains[:desired_client_num], start=1):
                # Get excluded indices for this domain
                exclude_indices = excluded_indices_per_domain.get(domain, None)

                domain_data = load_office_caltech_domain_data(
                    root=path,
                    domain=domain,
                    splits=splits,
                    transform=train_transform,
                    val_transform=val_transform,
                    test_transform=test_transform,
                    seed=config.seed,
                    exclude_indices=exclude_indices
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
                # Get excluded indices for this domain
                exclude_indices = excluded_indices_per_domain.get(domain, None)

                # Load full domain data
                domain_data = load_office_caltech_domain_data(
                    root=path,
                    domain=domain,
                    splits=splits,
                    transform=train_transform,
                    val_transform=val_transform,
                    test_transform=test_transform,
                    seed=config.seed,
                    exclude_indices=exclude_indices
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
                            # Only apply minimum samples constraint to training set
                            min_samples = getattr(config.data, 'min_samples_per_client', 5)
                            client_indices_list = split_data_by_dirichlet(
                                dataset=dataset,
                                num_clients=num_clients_this_domain,
                                num_classes=num_classes,
                                alpha=dirichlet_alpha,
                                seed=config.seed + domain_id + hash(split_name) % 1000,
                                min_samples_per_client=min_samples if split_name == 'train' else 0
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

        # Load server held-out test indices to prevent data leakage
        # This ensures client train/val/test data excludes server test samples
        excluded_indices_per_domain = {}
        if config.federate.make_global_eval:
            try:
                from federatedscope.cv.dataset.pacs import load_balanced_pacs_data

                # Support both fixed samples_per_class and ratio-based test_ratio
                samples_per_class_cfg = getattr(config.data, 'server_test_samples_per_class', 0)
                test_ratio_cfg = getattr(config.data, 'server_test_ratio', 0.1)

                # Determine which sampling mode to use
                if samples_per_class_cfg > 0:
                    samples_per_class = samples_per_class_cfg
                    test_ratio = None
                    logger.info(f"Loading server held-out test set (samples_per_class={samples_per_class}) to prevent data leakage...")
                else:
                    samples_per_class = None
                    test_ratio = test_ratio_cfg if test_ratio_cfg > 0 else 0.1
                    logger.info(f"Loading server held-out test set (test_ratio={test_ratio}) to prevent data leakage...")

                result = load_balanced_pacs_data(
                    root=path,
                    samples_per_class=samples_per_class,
                    test_ratio=test_ratio,
                    transform=None,
                    seed=config.seed
                )
                excluded_indices_per_domain = result['excluded_indices']
                logger.info(f"Server held-out indices loaded for {len(excluded_indices_per_domain)} domains")
                for domain, indices in excluded_indices_per_domain.items():
                    logger.info(f"  Domain '{domain}': {len(indices)} samples excluded from client data")
            except Exception as e:
                logger.warning(f"Failed to load server held-out indices: {e}")
                logger.warning("Proceeding without data exclusion - may have data leakage!")
                excluded_indices_per_domain = {}

        if desired_client_num <= num_domains:
            # If desired clients <= domains, use one domain per client
            data_dict = dict()
            for client_idx, domain in enumerate(domains[:desired_client_num], start=1):
                # Get excluded indices for this domain
                exclude_indices = excluded_indices_per_domain.get(domain, None)

                domain_data = load_pacs_domain_data(
                    root=path,
                    domain=domain,
                    splits=splits,
                    transform=train_transform,
                    val_transform=val_transform,
                    test_transform=test_transform,
                    seed=config.seed,
                    exclude_indices=exclude_indices
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
                # Get excluded indices for this domain
                exclude_indices = excluded_indices_per_domain.get(domain, None)

                # Load full domain data
                domain_data = load_pacs_domain_data(
                    root=path,
                    domain=domain,
                    splits=splits,
                    transform=train_transform,
                    val_transform=val_transform,
                    test_transform=test_transform,
                    seed=config.seed,
                    exclude_indices=exclude_indices
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
                            # Only apply minimum samples constraint to training set
                            min_samples = getattr(config.data, 'min_samples_per_client', 5)
                            client_indices_list = split_data_by_dirichlet(
                                dataset=dataset,
                                num_clients=num_clients_this_domain,
                                num_classes=num_classes,
                                alpha=dirichlet_alpha,
                                seed=config.seed + domain_id + hash(split_name) % 1000,
                                min_samples_per_client=min_samples if split_name == 'train' else 0
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

    elif name == 'officehome':
        # OfficeHome has 4 domains and 65 classes
        domains = ['Art', 'Clipart', 'Product', 'Real_World']
        num_domains = len(domains)
        num_classes = 65  # OfficeHome has 65 classes

        # Get the desired number of clients from config
        desired_client_num = config.federate.client_num

        # Get Dirichlet alpha parameter from config (0.0 means no Dirichlet, uniform split)
        dirichlet_alpha = getattr(config.data, 'dirichlet_alpha', 0.0)

        # Get transforms
        train_transform = transforms_funcs.get('transform', None)
        val_transform = val_transforms_funcs.get('transform', None)
        test_transform = test_transforms_funcs.get('transform', None)

        # Load server held-out test indices to prevent data leakage
        # This ensures client train/val/test data excludes server test samples
        excluded_indices_per_domain = {}
        if config.federate.make_global_eval:
            try:
                from federatedscope.cv.dataset.office_home import load_balanced_office_home_data

                # Support both fixed samples_per_class and ratio-based test_ratio
                samples_per_class_cfg = getattr(config.data, 'server_test_samples_per_class', 0)
                test_ratio_cfg = getattr(config.data, 'server_test_ratio', 0.1)

                # Determine which sampling mode to use
                if samples_per_class_cfg > 0:
                    samples_per_class = samples_per_class_cfg
                    test_ratio = None
                    logger.info(f"Loading server held-out test set (samples_per_class={samples_per_class}) to prevent data leakage...")
                else:
                    samples_per_class = None
                    test_ratio = test_ratio_cfg if test_ratio_cfg > 0 else 0.1
                    logger.info(f"Loading server held-out test set (test_ratio={test_ratio}) to prevent data leakage...")

                result = load_balanced_office_home_data(
                    root=path,
                    samples_per_class=samples_per_class,
                    test_ratio=test_ratio,
                    transform=None,
                    seed=config.seed
                )
                excluded_indices_per_domain = result['excluded_indices']
                logger.info(f"Server held-out indices loaded for {len(excluded_indices_per_domain)} domains")
                for domain, indices in excluded_indices_per_domain.items():
                    logger.info(f"  Domain '{domain}': {len(indices)} samples excluded from client data")
            except Exception as e:
                logger.warning(f"Failed to load server held-out indices: {e}")
                logger.warning("Proceeding without data exclusion - may have data leakage!")
                excluded_indices_per_domain = {}

        # Check if using Label Distribution Skew (LDS)
        use_lds = (config.data.splitter == 'lda_domain')
        lds_alpha = None

        if use_lds:
            # Get alpha from splitter_args
            if hasattr(config.data, 'splitter_args') and len(config.data.splitter_args) > 0:
                lds_alpha = config.data.splitter_args[0].get('alpha', 0.1)
            else:
                lds_alpha = 0.1  # Default alpha
            logger.info(f"="*80)
            logger.info(f"Using Label Distribution Skew (LDS) with alpha={lds_alpha}")
            logger.info(f"Generating {desired_client_num}x{num_classes} Dirichlet distribution matrix")
            logger.info(f"="*80)

            # Generate Dirichlet matrix: num_clients x num_classes
            # Each row represents a client's distribution over classes
            np.random.seed(config.seed)
            dirichlet_matrix = np.random.dirichlet([lds_alpha] * desired_client_num, num_classes).T

            logger.info(f"Dirichlet matrix shape: {dirichlet_matrix.shape}")
            logger.info(f"Matrix column sums (should be ~1.0): {dirichlet_matrix.sum(axis=0)}")

        if desired_client_num <= num_domains:
            # If desired clients <= domains, use one domain per client
            data_dict = dict()
            for client_idx, domain in enumerate(domains[:desired_client_num], start=1):
                # Get excluded indices for this domain
                exclude_indices = excluded_indices_per_domain.get(domain, None)

                domain_data = load_office_home_domain_data(
                    root=path,
                    domain=domain,
                    splits=splits,
                    transform=train_transform,
                    val_transform=val_transform,
                    test_transform=test_transform,
                    seed=config.seed,
                    exclude_indices=exclude_indices
                )

                # Apply Label Distribution Skew (LDS) if enabled
                if use_lds:
                    logger.info(f"Client {client_idx} ({domain}): Applying LDS")

                    # Get Dirichlet proportions for this client (one row of the matrix)
                    client_proportions = dirichlet_matrix[client_idx - 1]  # client_idx is 1-based

                    # Apply LDS to training set only (keep val/test unchanged for fair evaluation)
                    train_dataset = domain_data['train']
                    sampled_indices = apply_label_distribution_skew(
                        dataset=train_dataset,
                        dirichlet_proportions=client_proportions,
                        num_classes=num_classes,
                        seed=config.seed + client_idx
                    )

                    # Create subset with sampled indices
                    domain_data['train'] = Subset(train_dataset, sampled_indices)

                    logger.info(f"Client {client_idx} ({domain}): Original train size={len(train_dataset)}, "
                              f"After LDS={len(sampled_indices)}")

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
                # Get excluded indices for this domain
                exclude_indices = excluded_indices_per_domain.get(domain, None)

                # Load full domain data
                domain_data = load_office_home_domain_data(
                    root=path,
                    domain=domain,
                    splits=splits,
                    transform=train_transform,
                    val_transform=val_transform,
                    test_transform=test_transform,
                    seed=config.seed,
                    exclude_indices=exclude_indices
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
                            # Only apply minimum samples constraint to training set
                            min_samples = getattr(config.data, 'min_samples_per_client', 5)
                            client_indices_list = split_data_by_dirichlet(
                                dataset=dataset,
                                num_clients=num_clients_this_domain,
                                num_classes=num_classes,
                                alpha=dirichlet_alpha,
                                seed=config.seed + domain_id + hash(split_name) % 1000,
                                min_samples_per_client=min_samples if split_name == 'train' else 0
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







