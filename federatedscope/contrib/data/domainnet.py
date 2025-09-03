from federatedscope.core.data import BaseDataTranslator
from federatedscope.register import register_data
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from sklearn.model_selection import train_test_split
import numpy as np
import os
from PIL import Image
import json


class DomainnetDataset(Dataset):
    """DomainNet 数据集读取器，支持多个域，每个域下按类组织。

    结构假设：
    root_dir/
      <domain_A>/
        <class_1>/*.jpg
        <class_2>/*.jpg
        ...
      <domain_B>/
      <domain_C>/
        ...

    输出仅为 (image, label)，但对象内部保留 `labels`、
    `image_paths`、`domain_ids` 便于 splitter 使用。
    """
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []  # 全局类别 id（域之间共享一个 label 空间）
        self.domain_ids = []  # 每个样本所属域 id [0..num_domains-1]
        self.class_to_idx = {}
        self.idx_to_class = []
        self.domain_names = []

        # 枚举域
        domains = [
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]
        domains.sort()
        domains = domains[:3]
        self.domain_names = domains

        domain_to_selected = {}
        selected_class_names = set()
        for d in domains:
            dpath = os.path.join(root_dir, d)
            class_names = [
                cls for cls in os.listdir(dpath)
                if os.path.isdir(os.path.join(dpath, cls))
            ]
            class_names.sort()
            picked = class_names[:20]
            domain_to_selected[d] = set(picked)
            selected_class_names.update(picked)
        self.idx_to_class = sorted(selected_class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.idx_to_class)}

        # 逐域逐类收集样本
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
        for dom_id, d in enumerate(domains):
            dpath = os.path.join(root_dir, d)
            for cls in sorted(domain_to_selected.get(d, [])):
                cdir = os.path.join(dpath, cls)
                if not os.path.isdir(cdir):
                    continue
                cls_idx = self.class_to_idx[cls]
                files = [
                    f for f in os.listdir(cdir)
                    if f.lower().endswith(valid_ext)
                ]
                for f in files:
                    self.image_paths.append(os.path.join(cdir, f))
                    self.labels.append(cls_idx)
                    self.domain_ids.append(dom_id)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        return img, label


class TransformWrapper(Dataset):
    """对任意 Dataset/Subset 应用变换，不改变取样协议。

    注意：不改变 __getitem__ 的返回，只做图像增强与标准化。
    """
    def __init__(self, dataset: Dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        # 允许外部访问底层数据集的属性（如 labels/domain_ids ）
        if name in ('dataset', 'transform'):
            return super().__getattribute__(name)
        return getattr(self.dataset, name)


def _apply_split_transform(full_dataset: Dataset, idx_list, transform):
    subset = Subset(full_dataset, idx_list)
    return TransformWrapper(subset, transform=transform)


def _load_preprocessor_config(path: str):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def _pil_resample_to_interpolation_mode(resample: int) -> InterpolationMode:
    mapping = {
        0: InterpolationMode.NEAREST,  # PIL.Image.NEAREST
        1: InterpolationMode.LANCZOS,  # PIL.Image.LANCZOS
        2: InterpolationMode.BILINEAR,  # PIL.Image.BILINEAR
        3: InterpolationMode.BICUBIC,  # PIL.Image.BICUBIC
        4: InterpolationMode.BOX,  # PIL.Image.BOX
        5: InterpolationMode.HAMMING,  # PIL.Image.HAMMING
    }
    return mapping.get(resample, InterpolationMode.BICUBIC)


def _build_clip_transforms(pre_cfg: dict):
    # 默认 CLIP 预处理参数（与 openai/clip-vit-base-patch16 一致）
    size = pre_cfg.get('size', 224)
    crop_size = pre_cfg.get('crop_size', size)
    do_resize = pre_cfg.get('do_resize', True)
    do_center_crop = pre_cfg.get('do_center_crop', True)
    image_mean = pre_cfg.get('image_mean', [0.48145466, 0.4578275, 0.40821073])
    image_std = pre_cfg.get('image_std', [0.26862954, 0.26130258, 0.27577711])
    resample = _pil_resample_to_interpolation_mode(pre_cfg.get('resample', 3))

    steps = []
    if do_resize:
        # 按最短边等比缩放到 size
        steps.append(transforms.Resize(size, interpolation=resample))
    if do_center_crop:
        steps.append(transforms.CenterCrop(crop_size))
    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(image_mean, image_std),
    ])
    t = transforms.Compose(steps)
    # 训练与评测都使用相同预处理，以与 CLIP 预处理保持一致
    return t, t


def load_domainnet_data(config, client_cfgs=None):
    # 读取 CLIP 预处理配置
    # 优先从 config.data.preprocessor_path 获取；否则使用仓库内默认路径
    preprocessor_path = getattr(getattr(config, 'data', object()),
                                'preprocessor_path', None)
    if not preprocessor_path:
        preprocessor_path = \
            'pretrained_models/clip-vit-base-patch16/preprocessor_config.json'

    pre_cfg = _load_preprocessor_config(preprocessor_path) or {}
    train_transform, eval_transform = _build_clip_transforms(pre_cfg)

    # 加载完整数据集
    root = "/root/domainnet"
    full_dataset = DomainnetDataset(root)

    # 按 60/20/20 进行分层拆分（train/val/test）
    indices = np.arange(len(full_dataset))
    labels = [full_dataset.labels[i] for i in indices]

    # 先划分出 20% 测试集
    train_tmp_idx, test_idx = train_test_split(indices,
                                               test_size=0.2,
                                               stratify=labels,
                                               random_state=config.seed)
    # 再从剩余 80% 中划分出 25% 作为验证集（即总量的 20%）
    train_idx, val_idx = train_test_split(
        train_tmp_idx,
        test_size=0.25,
        stratify=[labels[i] for i in train_tmp_idx],
        random_state=config.seed)

    train_set = _apply_split_transform(full_dataset, train_idx,
                                       train_transform)
    val_set = _apply_split_transform(full_dataset, val_idx, eval_transform)
    test_set = _apply_split_transform(full_dataset, test_idx, eval_transform)

    split_datasets = (train_set, val_set, test_set)

    translator = BaseDataTranslator(config, client_cfgs)
    fs_data = translator(split_datasets)
    return fs_data, config


def call_domainnet_data(config, client_cfgs=None):
    if config.data.type == "Domainnet":
        return load_domainnet_data(config, client_cfgs=client_cfgs)


register_data("Domainnet", call_domainnet_data)
