# FederatedScope 分布式数据加载器设计方案

## 1. 当前实现分析

### 1.1 分布式训练架构
- 启动脚本通过多进程模拟分布式环境（1服务器+3客户端）
- 使用gRPC进行通信，服务器监听50051端口，客户端分别使用50052-50054端口
- 数据通过`data_idx`参数逻辑分配，所有数据存储在同一位置

### 1.2 当前数据加载机制
- 所有客户端数据集中存储在同一台机器的`data/`目录下
- 通过`data_idx`参数静态分配数据给不同客户端
- 数据在训练前完全加载到内存中
- 每个客户端访问预分配的数据子集，无法动态获取新数据

### 1.3 与真实联邦学习环境的差异
1. **数据存储**：集中式vs分布式存储
2. **数据隔离**：逻辑隔离vs物理隔离
3. **网络通信**：理想网络vs真实互联网环境
4. **客户端可用性**：始终在线vs动态参与
5. **数据异构性**：预先确定vs动态变化

## 2. 分布式数据加载器设计方案

### 2.1 核心设计理念
- 实现真正的数据分布式存储和访问
- 支持流式数据加载，减少内存占用
- 模拟真实联邦学习环境中的数据隔离
- 保持与现有代码的兼容性

### 2.2 架构设计

```
分布式数据加载架构
├── 服务器端
│   ├── 服务器管理器
│   └── 配置管理器
├── 客户端1
│   ├── 分布式数据加载器
│   ├── 数据管理器
│   └── 本地元数据
├── 客户端2
│   ├── 分布式数据加载器
│   ├── 数据管理器
│   └── 本地元数据
└── 客户端N
    ├── 分布式数据加载器
    ├── 数据管理器
    └── 本地元数据

数据存储
├── 客户端1数据目录
├── 客户端2数据目录
├── 客户端N数据目录
└── 中央数据存储
```

### 2.3 核心组件设计

#### 2.3.1 分布式数据加载器 (DistributedDataLoader)

```python
class DistributedDataLoader:
    """
    分布式数据加载器，实现真正的数据分布和流式加载
    """
    def __init__(self, client_id, config):
        self.client_id = client_id
        self.config = config
        self.local_data_path = self._get_client_data_path()
        self.data = None
        self.metadata = None
        
    def _get_client_data_path(self):
        """获取客户端专属数据目录"""
        return os.path.join(self.config.data.root, f"client_{self.client_id}_data")
    
    def initialize_local_data(self):
        """初始化本地数据，从中央存储获取数据"""
        # 确保数据目录存在
        os.makedirs(self.local_data_path, exist_ok=True)
        
        # 检查是否已有数据
        if self._has_local_data():
            self._load_existing_data()
        else:
            self._extract_from_central_storage()
        
        # 加载元数据
        self._load_metadata()
    
    def _has_local_data(self):
        """检查是否已有本地数据"""
        return os.path.exists(os.path.join(self.local_data_path, "train.pt"))
    
    def _load_existing_data(self):
        """加载已存在的本地数据"""
        train_path = os.path.join(self.local_data_path, "train.pt")
        test_path = os.path.join(self.local_data_path, "test.pt")
        
        if os.path.exists(train_path):
            self.data = {
                'train': torch.load(train_path),
                'test': torch.load(test_path) if os.path.exists(test_path) else None
            }
    
    def _extract_from_central_storage(self):
        """从中央存储中提取当前客户端的数据"""
        # 加载原始FEMNIST数据
        global_data_path = os.path.join(self.config.data.root, "femnist", "processed")
        
        if not os.path.exists(global_data_path):
            raise FileNotFoundError(f"FEMNIST data not found at {global_data_path}")
        
        # 根据客户端ID分配数据
        total_clients = self.config.federate.client_num
        user_files = sorted([f for f in os.listdir(global_data_path) 
                          if f.startswith('task_')])
        
        # 为当前客户端分配用户文件
        assigned_files = []
        for i, user_file in enumerate(user_files):
            if (i % total_clients) + 1 == self.client_id:
                assigned_files.append(user_file)
        
        # 加载分配的数据
        train_data, train_targets = [], []
        test_data, test_targets = [], []
        
        for user_file in assigned_files:
            user_path = os.path.join(global_data_path, user_file)
            if os.path.exists(os.path.join(user_path, 'train.pt')):
                user_train, user_train_targets = torch.load(
                    os.path.join(user_path, 'train.pt'))
                train_data.append(user_train)
                train_targets.append(user_train_targets)
                
                user_test, user_test_targets = torch.load(
                    os.path.join(user_path, 'test.pt'))
                test_data.append(user_test)
                test_targets.append(user_test_targets)
        
        # 合并数据
        if train_data:
            self.data = {
                'train': (torch.cat(train_data), torch.cat(train_targets)),
                'test': (torch.cat(test_data), torch.cat(test_targets))
            }
    
    def _load_metadata(self):
        """加载数据元数据"""
        metadata_path = os.path.join(self.local_data_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'client_id': self.client_id,
                'total_samples': len(self.data['train'][0]) if self.data else 0,
                'data_type': 'femnist',
                'last_updated': time.time()
            }
            self._save_metadata()
    
    def _save_metadata(self):
        """保存数据元数据"""
        metadata_path = os.path.join(self.local_data_path, "metadata.json")
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_data(self):
        """获取数据"""
        return self.data
    
    def get_client_id(self):
        """获取客户端ID"""
        return self.client_id
```

#### 2.3.2 修改数据构建器

```python
# 在federatedscope/core/auxiliaries/data_builder.py中添加
def load_distributed_dataset(config, client_cfgs=None):
    """
    加载分布式数据集，支持数据隔离
    """
    if config.federate.mode.lower() == 'distributed':
        if config.distribute.role == 'client':
            # 获取客户端ID
            client_id = getattr(config.distribute, 'client_id', None) or \
                      getattr(config.distribute, 'data_idx', None)
            
            if client_id is not None:
                # 创建分布式数据加载器
                data_loader = DistributedDataLoader(client_id, config)
                data_loader.initialize_local_data()
                return data_loader, config
            else:
                # 回退到原有逻辑
                return load_dataset(config, client_cfgs)
        else:
            # 服务器不持有数据
            return None, config
    else:
        # 使用原有逻辑
        return load_dataset(config, client_cfgs)
```

#### 2.3.3 修改客户端初始化

```python
# 在federatedscope/core/workers/client.py中修改
class Client(BaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 检查是否使用分布式数据加载器
        if (self._cfg.federate.mode.lower() == 'distributed' and 
            hasattr(self._cfg, 'distributed_data') and 
            self._cfg.distributed_data.use_distributed_loader and
            isinstance(self.data, DistributedDataLoader)):
            
            # 更新训练器的数据引用
            if hasattr(self, 'trainer') and self.trainer:
                self.trainer.data = self.data.get_data()
```

#### 2.3.4 配置文件扩展

```yaml
# 在客户端配置文件中添加
distribute:
  use: True
  server_host: '127.0.0.1'
  server_port: 50051
  client_host: '127.0.0.1'
  client_port: 50052
  role: 'client'
  client_id: 1  # 明确的客户端ID，替代data_idx
```

### 2.4 实施步骤

1. **创建分布式数据加载器**（1天）
   - 实现`DistributedDataLoader`类
   - 实现数据隔离和本地存储

2. **修改数据构建器**（1天）
   - 修改`load_distributed_dataset`函数
   - 集成分布式数据加载器

3. **修改客户端初始化**（1天）
   - 更新`Client`类
   - 支持分布式数据加载器

4. **扩展配置系统**（0.5天）
   - 添加分布式数据配置项
   - 保持向后兼容性

5. **测试验证**（0.5天）
   - 创建简单测试脚本
   - 验证数据隔离功能

### 2.5 实施优势

1. **数据隔离**：每个客户端只能访问自己的数据
2. **简单实现**：避免复杂的流式加载和网络模拟
3. **向后兼容**：通过配置开关控制，不影响现有代码
4. **易于测试**：实现简单，便于验证功能正确性

## 3. 关键文件修改清单

### 3.1 需要创建的新文件
- `federatedscope/core/data/distributed_loader.py` - 分布式数据加载器实现
- `tests/test_distributed_loader.py` - 单元测试

### 3.2 需要修改的现有文件
- `federatedscope/core/data/utils.py` - 修改数据加载入口点
- `federatedscope/core/auxiliaries/data_builder.py` - 集成分布式数据加载器
- `federatedscope/core/workers/client.py` - 修改客户端初始化
- `scripts/distributed_scripts/distributed_configs/distributed_femnist_client_*.yaml` - 更新配置文件

## 4. 使用方法

### 4.1 启用分布式数据加载器
在配置文件中设置：
```yaml
distributed_data:
  use_distributed_loader: true
```

### 4.2 运行分布式训练
```bash
# 使用原有的启动脚本，但配置已更新为支持分布式数据加载
python federatedscope/main.py --cfg scripts/distributed_scripts/distributed_configs/distributed_femnist_client_1.yaml
```

## 5. 预期效果

1. **数据隔离**：每个客户端只能访问自己目录下的数据
2. **真实模拟**：更贴近真实联邦学习环境的数据分布
3. **向后兼容**：不影响现有代码的使用
4. **易于扩展**：为未来功能扩展提供基础

这个设计方案专注于解决最关键的问题：数据隔离，同时保持了实现的简洁性和可维护性。