from federatedscope.nlp.dataset.leaf_nlp import LEAF_NLP
from federatedscope.nlp.dataset.leaf_twitter import LEAF_TWITTER
from federatedscope.nlp.dataset.leaf_synthetic import LEAF_SYNTHETIC
from federatedscope.core.auxiliaries.transform_builder import get_transform
from federatedscope.core.data import BaseDataTranslator
from torch.utils.data import DataLoader


def load_nlp_dataset(config=None):
    """
    Return the dataset of ``shakespeare``, ``subreddit``, ``twitter``, \
    or ``synthetic``.

    Args:
        config: configurations for FL, see ``federatedscope.core.configs``

    Returns:
        FL dataset dict, with ``client_id`` as key.

    Note:
      ``load_nlp_dataset()`` will return a dict as shown below:
        ```
        {'client_id': {'train': dataset, 'test': dataset, 'val': dataset}}
        ```
    """
    splits = config.data.splits

    path = config.data.root
    name = config.data.type.lower()
    transforms_funcs, _, _ = get_transform(config, 'torchtext')

    if name in ['shakespeare', 'subreddit']:
        dataset = LEAF_NLP(root=path,
                           name=name,
                           s_frac=config.data.subsample,
                           tr_frac=splits[0],
                           val_frac=splits[1],
                           seed=config.seed,
                           **transforms_funcs)
        # 加载全量数据
        train_data = dataset.get_train_data()
        val_data = dataset.get_val_data()
        test_data = dataset.get_test_data()

        # 用BaseDataTranslator进行ML和FL划分
        split_datasets = (train_data, val_data, test_data)
        translator = BaseDataTranslator(config)
        data_dict = translator(split_datasets)

        # 确保返回的是 Dataset 而不是 DataLoader
        for client_id in data_dict:
            for split in data_dict[client_id]:
                if isinstance(data_dict[client_id][split], DataLoader):
                    data_dict[client_id][split] = data_dict[client_id][
                        split].dataset

        return data_dict, config
    elif name == 'twitter':
        dataset = LEAF_TWITTER(root=path,
                               name='twitter',
                               s_frac=config.data.subsample,
                               tr_frac=splits[0],
                               val_frac=splits[1],
                               seed=config.seed,
                               **transforms_funcs)
    elif name == 'synthetic':
        dataset = LEAF_SYNTHETIC(root=path)
    else:
        raise ValueError(f'No dataset named: {name}!')

    client_num = min(len(dataset), config.federate.client_num
                     ) if config.federate.client_num > 0 else len(dataset)
    config.merge_from_list(['federate.client_num', client_num])

    # get local dataset
    data_dict = dict()
    for client_idx in range(1, client_num + 1):
        data_dict[client_idx] = dataset[client_idx - 1]

    return data_dict, config
