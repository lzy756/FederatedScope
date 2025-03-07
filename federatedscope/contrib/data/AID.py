from federatedscope.core.data import BaseDataTranslator
from federatedscope.register import register_data
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


# --- 自定义Dataset类（适应AID目录结构）---
class AIDDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.label_to_idx = {}  # 类别名称映射到数字标签

        # 遍历子目录收集数据（假设目录结构：root_dir/类别名/*.jpg）[<sup>7</sup>](https://blog.csdn.net/hhhhhhhhhhwwwwwwwwww/article/details/123049220)[<sup>15</sup>](https://blog.csdn.net/qq_47233366/article/details/128169498)
        classes = sorted(os.listdir(root_dir))
        for idx, cls in enumerate(classes):
            self.label_to_idx[cls] = idx
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                self.image_paths.append(os.path.join(cls_dir, img_name))
                self.labels.append(idx)

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def load_my_data(config, client_cfgs=None):
    # --- 数据预处理 ---
    image_size = 224  # ViT标准输入尺寸
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )  # 标准化 [<sup>2</sup>](https://www.zhihu.com/question/499121341?write)
    ])

    # --- 加载AID数据集 ---
    dataset = AIDDataset(root_dir="/home/lzy/OpenDataLab___AID/data/AID",
                         transform=transform)

    translator = BaseDataTranslator(config, client_cfgs)
    fs_data = translator(dataset)

    return fs_data, config


def call_my_data(config, client_cfgs=None):
    if config.data.type == "AID":
        data, modified_config = load_my_data(config, client_cfgs)
        return data, modified_config


register_data("AID", call_my_data)
