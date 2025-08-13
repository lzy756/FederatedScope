from torchvision import datasets, transforms
from federatedscope.register import register_data
import logging
from federatedscope.core.data import BaseDataTranslator

logger = logging.getLogger(__name__)

def load_mnist_data(config, client_cfgs=None):
    data_root = config.data.root
    
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),  # 统一缩放到32x32
            # 将灰度图复制为3通道（如果是RGB则无变化）
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),  # 转为Tensor并归一化到[0,1]
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),  # 标准化到[-1,1]
        ]
    )
    
    logger.info("Loading MNIST dataset...")
    mnist_train = datasets.MNIST(root=data_root, train=True, transform=transform, download=True)
    mnist_test = datasets.MNIST(root=data_root, train=False, transform=transform, download=True)
    split_datasets_minist = (mnist_train, mnist_test, mnist_test)
    
    translator = BaseDataTranslator(config, client_cfgs)
    data_dict = translator(split_datasets_minist)
    
    return data_dict, config
    
    

def call_mnist_data(config, client_cfgs=None):
    if config.data.type == "mnist":
        return load_mnist_data(config, client_cfgs)

register_data("mnist", call_mnist_data)