from federatedscope.cv.dataset.leaf_cv import LEAF_CV
from federatedscope.cv.dataset.office_caltech import load_office_caltech_domain_data
from federatedscope.core.auxiliaries.transform_builder import get_transform


def load_cv_dataset(config=None):
    """
    Return the dataset of ``femnist``, ``celeba``, or ``office_caltech``.

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
        # Office-Caltech has 4 domains, each domain is treated as a client
        domains = ['amazon', 'webcam', 'dslr', 'caltech']

        # Get transforms
        train_transform = transforms_funcs.get('transform', None)
        val_transform = val_transforms_funcs.get('transform', None)
        test_transform = test_transforms_funcs.get('transform', None)

        data_dict = dict()
        for client_idx, domain in enumerate(domains, start=1):
            domain_data = load_office_caltech_domain_data(
                root=path,
                domain=domain,
                splits=splits,
                transform=train_transform,
                val_transform=val_transform,
                test_transform=test_transform,
                seed=config.seed
            )

            # Return raw datasets (not DataLoader)
            # FederatedScope will create DataLoaders automatically
            data_dict[client_idx] = domain_data

        # Set client_num to 4 (number of domains)
        client_num = len(domains)
        config.merge_from_list(['federate.client_num', client_num])

    else:
        raise ValueError(f'No dataset named: {name}!')

    return data_dict, config


