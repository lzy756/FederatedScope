# FedProto Implementation for FederatedScope

## Overview

FedProto (Federated Prototypical Learning) is a personalized federated learning method that learns class prototypes in the embedding space to handle data heterogeneity across clients.

This implementation has been integrated into the FederatedScope framework following its architecture patterns.

## Key Features

- **Prototype-based Learning**: Learns class prototypes (centroids) in embedding space
- **Global Prototype Aggregation**: Server aggregates local prototypes to form global prototypes
- **Prototype-guided Training**: Clients use global prototypes for regularization
- **Flexible Configuration**: Support for multiple distance metrics and aggregation methods

## Files Created

### Core Implementation

1. **Configuration** (`federatedscope/core/configs/cfg_fedproto.py`)
   - Defines FedProto-specific configuration options
   - Registered via `register_config("fedproto", extend_fedproto_cfg)`

2. **Trainer** (`federatedscope/core/trainers/trainer_FedProto.py`)
   - `FedProtoTrainer`: Main trainer class inheriting from `GeneralTorchTrainer`
   - Key methods:
     - `_extract_embeddings()`: Extract embeddings from model
     - `_compute_prototypes()`: Compute class prototypes from embeddings
     - `_compute_proto_loss()`: Compute prototype-based loss
     - `_hook_on_batch_forward()`: Override forward pass with prototype loss
     - `update_global_prototypes()`: Receive global prototypes from server
     - `get_local_prototypes()`: Send local prototypes to server

3. **Server Worker** (`federatedscope/core/workers/server_FedProto.py`)
   - `FedProtoServer`: Server-side implementation
   - Responsibilities:
     - Aggregate local prototypes from clients
     - Compute global prototypes (mean or weighted mean)
     - Broadcast global prototypes to clients
     - Standard FedAvg model aggregation

4. **Client Worker** (`federatedscope/core/workers/client_FedProto.py`)
   - `FedProtoClient`: Client-side implementation
   - Responsibilities:
     - Receive global prototypes from server
     - Update trainer with global prototypes
     - Send local prototypes and model to server

### Configuration Files

1. **CIFAR-10 Example** (`scripts/example_configs/fedproto_cifar10.yaml`)
   - Standard heterogeneous FL setup with LDA splitting
   - 10 clients, alpha=0.5

2. **Office-Caltech Example** (`scripts/example_configs/fedproto_office_caltech.yaml`)
   - Domain adaptation scenario
   - 4 domains: Amazon, Webcam, DSLR, Caltech

### Registration

Updated files to register FedProto components:
- `federatedscope/core/trainers/__init__.py`: Added FedProtoTrainer import
- Config auto-registered via glob pattern in `federatedscope/core/configs/__init__.py`

## Configuration Options

```yaml
fedproto:
  use: True                          # Enable FedProto method
  proto_weight: 1.0                  # Weight for prototype loss
  embedding_dim: 512                 # Embedding dimension
  use_projector: False               # Use separate projection layer
  projector_hidden_dim: 256          # Hidden dim for projector
  distance_metric: 'euclidean'       # 'euclidean' or 'cosine'
  temperature: 0.5                   # Temperature for cosine distance
  aggregation_method: 'mean'         # 'mean' or 'weighted_mean'
  local_proto_epochs: 1              # Local epochs for prototype update
  normalize_prototypes: False        # Normalize prototypes
  freeze_backbone: False             # Freeze backbone during training
```

## Usage

### Basic Usage

1. **Set trainer type in config:**
```yaml
trainer:
  type: 'fedproto_trainer'
```

2. **Enable FedProto:**
```yaml
fedproto:
  use: True
  proto_weight: 1.0
  embedding_dim: 512
```

3. **Run training:**
```bash
python federatedscope/main.py --cfg scripts/example_configs/fedproto_cifar10.yaml
```

### Advanced Usage

#### Using Cosine Distance

```yaml
fedproto:
  distance_metric: 'cosine'
  temperature: 0.1
  normalize_prototypes: True
```

#### Using Projection Layer

```yaml
fedproto:
  use_projector: True
  projector_hidden_dim: 256
```

#### Weighted Aggregation

```yaml
fedproto:
  aggregation_method: 'weighted_mean'  # Weight by client sample size
```

## Algorithm Flow

### Server Side (each round)
1. Receive model parameters and local prototypes from clients
2. Aggregate model parameters using FedAvg
3. Aggregate prototypes to form global prototypes:
   - `mean`: Simple average of all prototypes
   - `weighted_mean`: Weighted by number of samples
4. Broadcast updated model and global prototypes to clients

### Client Side (each round)
1. Receive global model and global prototypes from server
2. Update local trainer with global prototypes
3. Train local model with combined loss:
   - `L_total = L_CE + λ * L_proto`
   - `L_CE`: Standard cross-entropy loss
   - `L_proto`: Prototype-based loss
4. Compute local prototypes from embeddings
5. Send model parameters and local prototypes to server

## Implementation Details

### Embedding Extraction

The trainer attempts multiple strategies to extract embeddings:

1. **Explicit methods**: If model has `forward_embedding()` or separate `features`/`classifier`
2. **Hook-based**: Registers forward hook on second-to-last module
3. **Projector**: Optional projection layer for embedding transformation

### Prototype Computation

```python
# For each class c:
prototypes[c] = mean(embeddings[labels == c])

# Optional normalization:
if normalize_prototypes:
    prototypes[c] = prototypes[c] / ||prototypes[c]||_2
```

### Distance Metrics

**Euclidean Distance:**
```python
dist(e, p) = ||e - p||_2
loss = mean(dist(e, p_correct))
```

**Cosine Distance:**
```python
sim(e, p) = dot(e_norm, p_norm)
dist(e, p) = 1 - sim(e, p)
logits = -dist / temperature
loss = CrossEntropy(logits, labels)
```

## Comparison with FedLSA

| Feature | FedProto | FedLSA |
|---------|----------|--------|
| **Prototypes** | Averaged from client embeddings | Learned on server via network |
| **Server Learning** | None (only aggregation) | Trains anchor learner network |
| **Loss Components** | Classification + Prototype distance | Classification + Compactness + Separation |
| **Projector** | Optional | Required (hyperspherical) |
| **Complexity** | Lower | Higher |
| **Best for** | General heterogeneous FL | Domain-skewed data |

## Customization

### Custom Embedding Extraction

If your model has a specific structure, modify `_extract_embeddings()`:

```python
def _extract_embeddings(self, model, x):
    # Custom extraction logic
    embeddings = model.my_custom_feature_extractor(x)
    return embeddings
```

### Custom Prototype Loss

Modify `_compute_proto_loss()` for different loss formulations:

```python
def _compute_proto_loss(self, embeddings, labels, prototypes):
    # Custom loss computation
    # e.g., triplet loss, contrastive loss, etc.
    return custom_loss
```

## Testing

### Syntax Check

```bash
python -m py_compile federatedscope/core/configs/cfg_fedproto.py
python -m py_compile federatedscope/core/trainers/trainer_FedProto.py
python -m py_compile federatedscope/core/workers/server_FedProto.py
python -m py_compile federatedscope/core/workers/client_FedProto.py
```

### Import Test

```bash
python test_fedproto_import.py
```

### Full Training Test

```bash
# Quick test with small dataset
python federatedscope/main.py --cfg scripts/example_configs/fedproto_cifar10.yaml \
  data.subsample 0.1 \
  federate.total_round_num 5
```

## Troubleshooting

### Issue: Prototypes are None

**Cause**: No training data or embedding extraction failed

**Solution**:
- Check if training data is available
- Verify `_extract_embeddings()` works with your model
- Add debug prints to check embedding shapes

### Issue: NaN loss

**Cause**: Numerical instability in distance computation

**Solution**:
- Enable `normalize_prototypes: True`
- Adjust `temperature` parameter for cosine distance
- Check for zero-division in prototype computation

### Issue: Poor performance

**Solution**:
- Increase `proto_weight` (0.5 → 2.0)
- Try different `distance_metric`
- Enable `use_projector` for better embedding quality
- Use `weighted_mean` aggregation for imbalanced data

## References

FedProto paper:
```
FedProto: Federated Prototype Learning across Heterogeneous Clients
```

FederatedScope framework:
```
FederatedScope: A Flexible Federated Learning Platform for Heterogeneity
```

## Notes

- The implementation follows FederatedScope's trainer architecture patterns
- Compatible with existing FL configurations and data loaders
- Can be combined with other FL methods via wrapper trainers
- Server and client workers can be customized for specific deployment scenarios

## Future Enhancements

Possible improvements:
- [ ] Add support for multi-prototype per class
- [ ] Implement momentum-based prototype updates
- [ ] Add prototype visualization utilities
- [ ] Support for hierarchical prototypes
- [ ] Integration with differential privacy mechanisms
