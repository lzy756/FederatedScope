from importlib import import_module
import federatedscope.register as register

# 支持自定义和 PyG transforms
import torch_geometric.transforms as _pyg_transforms
import federatedscope.contrib.data.transforms as _custom_transforms

def get_transform(config, package):
    """
    This function is to build transforms applying to dataset.
    Args:
        config: CN from config.py
        package: one of ['torchvision','torch_geometric',...]
    Returns:
        Tuple of dicts: (train_funcs, val_funcs, test_funcs)
    """
    transform_funcs = {}
    for name in ['transform', 'target_transform', 'pre_transform']:
        if config.data.get(name):
            transform_funcs[name] = config.data[name]

    val_transform_funcs = {}
    for name in ['val_transform', 'val_target_transform', 'val_pre_transform']:
        suf = name.split('val_')[1]
        if config.data.get(name):
            val_transform_funcs[suf] = config.data[name]

    test_transform_funcs = {}
    for name in ['test_transform', 'test_target_transform', 'test_pre_transform']:
        suf = name.split('test_')[1]
        if config.data.get(name):
            test_transform_funcs[suf] = config.data[name]

    # no transforms
    if not transform_funcs and not val_transform_funcs and not test_transform_funcs:
        return {}, {}, {}

    # import package transforms module
    transforms_module = getattr(import_module(package), 'transforms')

    def convert(trans):
        # list-form: ["Name", {args}]
        if isinstance(trans, list) and trans and isinstance(trans[0], str):
            if len(trans) == 1:
                trans.append({})
            name, args = trans[0], trans[1]
            # 1. 注册表优先
            for func in register.transform_dict.values():
                tf = func(name, args)
                if tf is not None:
                    return tf
            # 2. PyG transforms
            if hasattr(_pyg_transforms, name):
                return getattr(_pyg_transforms, name)(**args)
            # 3. 自定义 transforms
            if hasattr(_custom_transforms, name):
                return getattr(_custom_transforms, name)(**args)
            # 4. 包内 transforms
            if hasattr(transforms_module, name):
                return getattr(transforms_module, name)(**args)
            raise ValueError(f"Unknown transform: {name}")
        # recursive for nested lists
        elif isinstance(trans, list):
            built = [convert(x) for x in trans]
            if hasattr(transforms_module, 'Compose'):
                return transforms_module.Compose(built)
            elif hasattr(transforms_module, 'Sequential'):
                return transforms_module.Sequential(built)
            else:
                return built
        else:
            # assume already a callable or transform
            return trans

    # apply convert to each entry
    if transform_funcs:
        for k in list(transform_funcs):
            transform_funcs[k] = convert(transform_funcs[k])
    if val_transform_funcs:
        for k in list(val_transform_funcs):
            val_transform_funcs[k] = convert(val_transform_funcs[k])
    else:
        val_transform_funcs = transform_funcs
    if test_transform_funcs:
        for k in list(test_transform_funcs):
            test_transform_funcs[k] = convert(test_transform_funcs[k])
    else:
        test_transform_funcs = transform_funcs

    return transform_funcs, val_transform_funcs, test_transform_funcs
