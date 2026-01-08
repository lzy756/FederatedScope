"""
验证脚本：使用论文预计算的数据运行GGEUR
直接使用论文的CLIP特征和协方差矩阵，验证训练流程是否正确
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from glob import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# 路径配置
PAPER_DATA_ROOT = "D:/Projects/2025CVPR_GGEUR/Multi Domain/Office-Home-LDS"
TRAIN_FEATURES_DIR = os.path.join(PAPER_DATA_ROOT, "backup_vit_b32")
TEST_FEATURES_DIR = os.path.join(PAPER_DATA_ROOT, "clip_office_home_test_features")
COV_MATRIX_DIR = os.path.join(PAPER_DATA_ROOT, "cov_matrix_output")

DOMAINS = ['Art', 'Clipart', 'Product', 'Real_World']
NUM_CLASSES = 65


class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features).float() if isinstance(features, np.ndarray) else features.float()
        self.labels = torch.from_numpy(labels).long() if isinstance(labels, np.ndarray) else labels.long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# 论文的模型 - 带softmax输出
class MyNet(nn.Module):
    def __init__(self, num_classes=65):
        super(MyNet, self).__init__()
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        return F.softmax(self.fc3(x), dim=1)


# 不带softmax的模型 (用于对比)
class MyNetLogits(nn.Module):
    def __init__(self, num_classes=65):
        super(MyNetLogits, self).__init__()
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.fc3(x)


def nearest_pos_def(cov_matrix):
    """论文的nearest_pos_def实现"""
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 缩放前10个最小特征值
    scale_factors = np.ones_like(eigenvalues)
    scale_factors[:10] = np.linspace(5, 0.5, 10)
    eigenvalues = eigenvalues * scale_factors
    eigenvalues[eigenvalues < 0] = 0

    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def generate_samples(feature, cov_matrix, num_samples):
    """论文的样本生成函数"""
    cov_matrix = nearest_pos_def(cov_matrix)

    jitter = 1e-6
    for _ in range(10):
        try:
            B = np.linalg.cholesky(cov_matrix + jitter * np.eye(512))
            break
        except np.linalg.LinAlgError:
            jitter *= 10

    return np.random.multivariate_normal(feature, B @ B.T, num_samples)


DOMAIN_TO_CLIENT = {
    'Art': 0,
    'Clipart': 1,
    'Product': 2,
    'Real_World': 3
}


def load_train_features(domain):
    """加载训练特征"""
    domain_dir = os.path.join(TRAIN_FEATURES_DIR, domain)
    client_id = DOMAIN_TO_CLIENT[domain]
    all_features = []
    all_labels = []

    for class_id in range(NUM_CLASSES):
        feat_file = os.path.join(domain_dir, f"client_{client_id}_class_{class_id}_original_features.npy")
        label_file = os.path.join(domain_dir, f"client_{client_id}_class_{class_id}_labels.npy")

        if os.path.exists(feat_file) and os.path.exists(label_file):
            features = np.load(feat_file)
            labels = np.load(label_file)
            if len(features) > 0:
                all_features.append(features)
                all_labels.append(labels)

    if len(all_features) > 0:
        return np.vstack(all_features), np.concatenate(all_labels)
    return np.empty((0, 512)), np.empty(0, dtype=np.int64)


def load_test_features(domain):
    """加载测试特征"""
    feat_file = os.path.join(TEST_FEATURES_DIR, domain, f"{domain}_test_features.npy")
    label_file = os.path.join(TEST_FEATURES_DIR, domain, f"{domain}_test_labels.npy")

    features = np.load(feat_file)
    labels = np.load(label_file)
    return features, labels


def load_covariance_matrix(class_id):
    """加载协方差矩阵"""
    cov_file = os.path.join(COV_MATRIX_DIR, f"class_{class_id}_cov_matrix.npy")
    if os.path.exists(cov_file):
        return np.load(cov_file)
    return np.zeros((512, 512))


def augment_features(features_by_class, prototypes_by_domain, cov_matrices, target_size=50):
    """
    论文的数据增强流程

    Args:
        features_by_class: {class_id: features}
        prototypes_by_domain: {domain: {class_id: prototype}}
        cov_matrices: {class_id: cov_matrix}
        target_size: 每类目标样本数
    """
    augmented_features = []
    augmented_labels = []

    for class_id in range(NUM_CLASSES):
        original = features_by_class.get(class_id, np.empty((0, 512)))
        cov = cov_matrices.get(class_id, np.zeros((512, 512)))

        num_original = len(original)

        # Case 1: 足够的原始样本
        if num_original >= target_size:
            indices = np.random.choice(num_original, target_size, replace=False)
            final = original[indices]
        else:
            # Case 2: 需要生成样本
            # Step 1: 从原始样本扩展
            if num_original > 0 and not np.all(cov == 0):
                expanded = []
                for feat in original:
                    new_samples = generate_samples(feat, cov, 50)
                    expanded.append(new_samples)
                expanded = np.vstack(expanded)

                if len(expanded) >= 50:
                    indices = np.random.choice(len(expanded), 50, replace=False)
                    expanded = expanded[indices]
            else:
                expanded = np.empty((0, 512))

            # Step 2: 从其他域原型生成
            other_generated = []
            for domain, domain_prototypes in prototypes_by_domain.items():
                if class_id in domain_prototypes:
                    proto = domain_prototypes[class_id]
                    if len(proto) > 0 and not np.all(cov == 0):
                        new_samples = generate_samples(proto, cov, 50)
                        other_generated.append(new_samples)

            if len(other_generated) > 0:
                other_generated = np.vstack(other_generated)
            else:
                other_generated = np.empty((0, 512))

            # Step 3: combine_samples
            if num_original == 0:
                if len(other_generated) >= target_size:
                    indices = np.random.choice(len(other_generated), target_size, replace=False)
                    final = other_generated[indices]
                else:
                    final = other_generated
            else:
                num_needed = target_size - num_original
                all_generated = []
                if len(expanded) > 0:
                    all_generated.append(expanded)
                if len(other_generated) > 0:
                    all_generated.append(other_generated)

                if len(all_generated) > 0:
                    all_generated = np.vstack(all_generated)
                    if len(all_generated) >= num_needed:
                        indices = np.random.choice(len(all_generated), num_needed, replace=False)
                        selected = all_generated[indices]
                    else:
                        selected = all_generated
                    final = np.vstack([original, selected])
                else:
                    final = original

        if len(final) > 0:
            augmented_features.append(final)
            augmented_labels.append(np.full(len(final), class_id, dtype=np.int64))

    return np.vstack(augmented_features), np.concatenate(augmented_labels)


def train_paper_style(model, dataloader, optimizer, device):
    """论文的训练方式：softmax输出 + CrossEntropyLoss"""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)  # 概率输出
        loss = criterion(outputs, labels)  # 这是double softmax!
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / total, correct / total


def train_correct_style(model, dataloader, optimizer, device):
    """正确的训练方式：logits输出 + CrossEntropyLoss"""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)  # logits输出
        loss = criterion(outputs, labels)  # 正确使用
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / total, correct / total


def train_nll_style(model, dataloader, optimizer, device):
    """修正的训练方式：softmax输出 + NLLLoss"""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)  # 概率输出
        # 正确处理：log(prob) + NLLLoss
        log_probs = torch.log(torch.clamp(outputs, min=1e-7))
        loss = F.nll_loss(log_probs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / total, correct / total


def test(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def federated_averaging(global_model, client_models):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([cm.state_dict()[k].float() for cm in client_models], 0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model


def run_experiment(train_style='paper', num_rounds=50, local_epochs=1, lr=0.001):
    """运行实验"""
    logger.info("=" * 60)
    logger.info(f"实验配置: style={train_style}, rounds={num_rounds}, local_epochs={local_epochs}, lr={lr}")
    logger.info("=" * 60)

    np.random.seed(12345)
    torch.manual_seed(12345)

    # 1. 加载数据
    logger.info("加载训练和测试数据...")

    client_features = {}
    test_features_all = []
    test_labels_all = []

    for domain in DOMAINS:
        train_feats, train_labels = load_train_features(domain)
        test_feats, test_labels = load_test_features(domain)

        logger.info(f"  {domain}: train={len(train_feats)}, test={len(test_feats)}")

        # 按类别组织
        client_features[domain] = {}
        for class_id in range(NUM_CLASSES):
            mask = train_labels == class_id
            client_features[domain][class_id] = train_feats[mask]

        test_features_all.append(test_feats)
        test_labels_all.append(test_labels)

    test_features = np.vstack(test_features_all)
    test_labels = np.concatenate(test_labels_all)
    logger.info(f"测试集总计: {len(test_features)} 样本")

    # 2. 加载协方差矩阵
    logger.info("加载协方差矩阵...")
    cov_matrices = {}
    for class_id in range(NUM_CLASSES):
        cov_matrices[class_id] = load_covariance_matrix(class_id)

    # 3. 计算原型
    logger.info("计算原型...")
    prototypes = {}
    for domain in DOMAINS:
        prototypes[domain] = {}
        for class_id in range(NUM_CLASSES):
            feats = client_features[domain][class_id]
            if len(feats) > 0:
                prototypes[domain][class_id] = np.mean(feats, axis=0)

    # 4. 数据增强 (离线!)
    logger.info("执行数据增强 (离线)...")
    augmented_data = {}
    for domain_idx, domain in enumerate(DOMAINS):
        # 获取其他域的原型
        other_protos = {d: prototypes[d] for d in DOMAINS if d != domain}

        # 增强
        aug_feats, aug_labels = augment_features(
            client_features[domain],
            other_protos,
            cov_matrices,
            target_size=50
        )
        augmented_data[domain] = (aug_feats, aug_labels)
        logger.info(f"  {domain}: 增强后 {len(aug_feats)} 样本")

    # 5. 创建模型
    if train_style == 'paper':
        global_model = MyNet(NUM_CLASSES).to(device)
        client_models = [MyNet(NUM_CLASSES).to(device) for _ in DOMAINS]
        train_fn = train_paper_style
    elif train_style == 'nll':
        global_model = MyNet(NUM_CLASSES).to(device)
        client_models = [MyNet(NUM_CLASSES).to(device) for _ in DOMAINS]
        train_fn = train_nll_style
    else:  # 'correct'
        global_model = MyNetLogits(NUM_CLASSES).to(device)
        client_models = [MyNetLogits(NUM_CLASSES).to(device) for _ in DOMAINS]
        train_fn = train_correct_style

    # 6. 准备数据加载器
    client_loaders = []
    for domain in DOMAINS:
        features, labels = augmented_data[domain]
        dataset = FeatureDataset(features, labels)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        client_loaders.append(loader)

    test_dataset = FeatureDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 7. 联邦训练
    logger.info("开始联邦训练...")
    best_acc = 0.0

    for round_idx in range(num_rounds):
        # 分发全局模型
        for client_model in client_models:
            client_model.load_state_dict(global_model.state_dict())

        # 本地训练
        for client_idx, loader in enumerate(client_loaders):
            client_model = client_models[client_idx]
            optimizer = optim.Adam(client_model.parameters(), lr=lr)

            for epoch in range(local_epochs):
                train_loss, train_acc = train_fn(client_model, loader, optimizer, device)

        # FedAvg
        global_model = federated_averaging(global_model, client_models)

        # 评估
        test_acc = test(global_model, test_loader, device)
        if test_acc > best_acc:
            best_acc = test_acc

        if (round_idx + 1) % 10 == 0 or round_idx == 0:
            logger.info(f"Round {round_idx + 1}/{num_rounds}: Test Acc = {test_acc:.4f} ({test_acc*100:.2f}%), Best = {best_acc:.4f}")

    logger.info(f"\n最终结果: {best_acc:.4f} ({best_acc*100:.2f}%)")
    return best_acc


if __name__ == "__main__":
    print("\n" + "="*70)
    print("使用论文预计算数据验证GGEUR实现")
    print("="*70)

    # 检查数据是否存在
    if not os.path.exists(TRAIN_FEATURES_DIR):
        print(f"错误: 找不到训练数据目录 {TRAIN_FEATURES_DIR}")
        exit(1)

    # 实验1: 论文的方式 (softmax + CrossEntropyLoss)
    print("\n" + "="*70)
    print("实验1: 论文方式 (softmax + CrossEntropyLoss - Double Softmax)")
    print("="*70)
    paper_acc = run_experiment(train_style='paper', num_rounds=50, local_epochs=1, lr=0.001)

    # 实验2: 修正方式 (softmax + NLLLoss)
    print("\n" + "="*70)
    print("实验2: 修正方式 (softmax + NLLLoss)")
    print("="*70)
    nll_acc = run_experiment(train_style='nll', num_rounds=50, local_epochs=1, lr=0.001)

    # 实验3: 标准方式 (logits + CrossEntropyLoss)
    print("\n" + "="*70)
    print("实验3: 标准方式 (logits + CrossEntropyLoss)")
    print("="*70)
    correct_acc = run_experiment(train_style='correct', num_rounds=50, local_epochs=1, lr=0.001)

    # 结果比较
    print("\n" + "="*70)
    print("实验结果比较")
    print("="*70)
    print(f"论文方式 (softmax + CE):  {paper_acc:.4f} ({paper_acc*100:.2f}%)")
    print(f"修正方式 (softmax + NLL): {nll_acc:.4f} ({nll_acc*100:.2f}%)")
    print(f"标准方式 (logits + CE):   {correct_acc:.4f} ({correct_acc*100:.2f}%)")

    print("\n结论:")
    if paper_acc > 0.5:
        print("- 论文方式有效，可能是因为double softmax意外产生了某种正则化效果")
    elif nll_acc > paper_acc:
        print("- 修正方式更好，证明double softmax确实有问题")
    if correct_acc > max(paper_acc, nll_acc):
        print("- 标准方式最好，应该使用logits+CrossEntropyLoss")
