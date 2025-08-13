# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FederatedScope is a comprehensive federated learning platform by Alibaba DAMO Academy, built on an event-driven architecture. It provides convenient usage and flexible customization for various federated learning tasks in both academia and industry.

**Version:** 0.3.0  
**Python Requirements:** >= 3.9  
**License:** Apache 2.0

## Installation & Setup Commands

### Environment Setup
```bash
# Clone and navigate
git clone https://github.com/alibaba/FederatedScope.git
cd FederatedScope

# Create conda environment
conda create -n fs python=3.9
conda activate fs

# Install PyTorch (adjust CUDA version as needed)
conda install -y pytorch=1.10.1 torchvision=0.11.2 torchaudio=0.10.1 torchtext=0.11.1 cudatoolkit=11.3 -c pytorch -c conda-forge

# Install FederatedScope
pip install -e .                    # Basic installation
pip install -e .[dev]              # Development mode with pre-commit hooks
pip install -e .[app]              # With application dependencies (graph, NLP, etc.)

# For development
pre-commit install
```

### Docker Alternative
```bash
# Basic environment
docker build -f environment/docker_files/federatedscope-torch1.10.Dockerfile -t alibaba/federatedscope:base-env-torch1.10 .
docker run --gpus device=all --rm -it --name "fedscope" -w $(pwd) alibaba/federatedscope:base-env-torch1.10 /bin/bash

# Application environment (for graph, NLP tasks)
docker build -f environment/docker_files/federatedscope-torch1.10-application.Dockerfile -t alibaba/federatedscope:app-env-torch1.10 .
```

## Common Development Commands

### Running Experiments
```bash
# Standalone mode (simulation on single device)
python federatedscope/main.py --cfg scripts/example_configs/femnist.yaml

# With custom parameters
python federatedscope/main.py --cfg scripts/example_configs/femnist.yaml federate.total_round_num 50 dataloader.batch_size 128

# Distributed mode - start server first
python federatedscope/main.py --cfg scripts/distributed_scripts/distributed_configs/distributed_server.yaml

# Then start clients
python federatedscope/main.py --cfg scripts/distributed_scripts/distributed_configs/distributed_client_1.yaml
```

### Testing
```bash
# Run all tests
python tests/run.py

# Run specific test pattern
python tests/run.py --pattern "test_femnist*"

# List available tests
python tests/run.py --list_tests

# Using pytest (if available)
pytest tests/ -v --cov=federatedscope --cov-report=html
```

### Development Tools
```bash
# Pre-commit checks
pre-commit run --all-files

# Generate documentation
cd doc
pip install -r requirements.txt
make html
```

## Architecture Overview

### Core Event-Driven Architecture

**Workers** (`federatedscope/core/workers/`):
- `BaseServer`: Orchestrates FL rounds, handles model aggregation
- `BaseClient`: Manages local training and model updates
- Communication via gRPC or message passing

**Trainers** (`federatedscope/core/trainers/`):
- `BaseTrainer`: Abstract training interface
- `TorchTrainer`: PyTorch-based implementation
- Specialized trainers for FL algorithms (FedProx, Ditto, etc.)

**Aggregators** (`federatedscope/core/aggregators/`):
- `ClientsAvgAggregator`: FedAvg implementation
- `KrumAggregator`: Byzantine-robust aggregation
- `FedOptAggregator`: Server-side optimization

**Configuration System** (`federatedscope/core/configs/`):
- YAML-based hierarchical configuration
- Modular config components (`cfg_*.py`)
- Runtime configuration merging

### Domain-Specific Modules

- **`federatedscope/cv/`**: Computer vision FL (CNNs, image datasets)
- **`federatedscope/nlp/`**: NLP FL (transformers, text processing)
- **`federatedscope/gfl/`**: Graph federated learning (GNNs)
- **`federatedscope/vertical_fl/`**: Vertical federated learning

### Extension System

The **`federatedscope/contrib/`** directory provides registration mechanisms for:
- Custom datasets, models, optimizers
- New FL algorithms and aggregation methods
- Specialized trainers and metrics

## Key Configuration Patterns

### Datasets
```yaml
data:
  type: 'femnist'  # Built-in datasets via DataZoo
  # Custom datasets require registration
```

### Models
```yaml
model:
  type: 'convnet2'  # Built-in models via ModelZoo
  # Custom models require registration
```

### Federated Learning Settings
```yaml
federate:
  mode: 'standalone'  # or 'distributed'
  total_round_num: 20
  client_num: 10
```

## Development Workflows

### Adding Custom FL Algorithm
1. Create trainer in `federatedscope/core/trainers/`
2. Add aggregator in `federatedscope/core/aggregators/` 
3. Register components in `federatedscope/register.py`
4. Add config options in `federatedscope/core/configs/`
5. Create test cases in `tests/`
6. Add example config in `scripts/example_configs/`

### Working with Benchmarks
The project includes three major benchmark suites:
- **pFL-Bench**: Personalized FL (10+ datasets, 20+ baselines)
- **FedHPOBench**: Hyperparameter optimization
- **B-FHTL**: Federated hetero-task learning

### Execution Modes
- **Standalone**: Single-device simulation of multiple participants
- **Distributed**: Multi-process/multi-machine deployment with real network communication

## Important Notes

- Always check existing implementations before creating new components
- Use the registration system in `contrib/` for extensibility
- Follow the event-driven architecture patterns
- YAML configurations should follow the modular structure in `federatedscope/core/configs/`
- The platform supports both research prototyping and production deployment