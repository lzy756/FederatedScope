# GGEUR Implementation Differences: Paper Source Code vs FederatedScope

## Critical Differences Found

### 1. Sample Generation Algorithm (MOST CRITICAL)

#### Paper Source Code (`prototype_cov_matrix_generate_features.py`)
```python
def generate_new_samples(feature, cov_matrix, num_generated):
    cov_matrix = nearest_pos_def(cov_matrix)
    jitter = 1e-6
    while True:
        try:
            B = np.linalg.cholesky(cov_matrix + jitter * np.eye(cov_matrix.shape[0]))
            break
        except np.linalg.LinAlgError:
            jitter *= 10
    new_features = np.random.multivariate_normal(feature, B @ B.T, num_generated)
    return new_features
```

**Key points:**
- Uses **full covariance matrix** Σ
- Performs Cholesky decomposition: B = chol(Σ)
- Generates samples from **N(μ, Σ)** using multivariate normal
- Adaptive jitter for numerical stability

#### FederatedScope (`ggeur_augmentation.py`)
```python
epsilon = torch.randn(n_gen, len(eigenvalues), device=self.device)  # ε ~ N(0,1)
scaled_epsilon = epsilon * eigenvalues.unsqueeze(0)  # ε * λ
beta = torch.mm(scaled_epsilon, eigenvectors.t())  # β = (ε * λ) @ ξ^T
new_samples = x.unsqueeze(0) + beta  # X_new = X + β
```

**Key points:**
- Uses **diagonal approximation** of covariance: diag(λ)
- Generates perturbations β from N(0, diag(λ))
- **Does NOT use full covariance structure**
- No Cholesky decomposition

**Impact:** FederatedScope generates samples from a **much simpler distribution** that ignores correlations between dimensions!

---

### 2. Eigenvalue Scaling (CRITICAL)

#### Paper Source Code (`nearest_pos_def`)
```python
def nearest_pos_def(cov_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    scale_factors = np.ones_like(eigenvalues)
    scale_factors[:10] = np.linspace(5, 1, 10)  # Scale top 10 eigenvalues!
    eigenvalues = eigenvalues * scale_factors
    eigenvalues[eigenvalues < 0] = 0
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
```

**Key points:**
- Top 10 eigenvalues are **amplified** by factors [5.0, 4.5, 4.0, ..., 1.0]
- This emphasizes major variations in the data
- Makes augmentation more aggressive

#### FederatedScope
- **No eigenvalue scaling** at all
- Uses raw eigenvalues directly

**Impact:** FederatedScope generates much less diverse augmented samples!

---

### 3. Augmentation Parameters

#### Paper Source Code (Digits dataset)
```python
datasets_config = {
    'mnist': {'client_ids': [0], 'num_generated_per_sample': 12},
    'usps': {'client_ids': [1], 'num_generated_per_sample': 12},
    'svhn': {'client_ids': [2], 'num_generated_per_sample': 12},
    'syn': {'client_ids': [3], 'num_generated_per_sample': 12}
}
```

**Augmentation strategy:**
1. For each original sample, generate 12 new samples
2. Randomly select 500 completion samples from all generated
3. Generate 1500 samples from other domain prototypes (500 per domain × 3 domains)
4. Final: ~2000 samples per class (original + completion + cross-domain)

#### FederatedScope (PACS config)
Current parameters are unclear from configs, but likely different.

---

### 4. Training Hyperparameters

#### Paper Source Code
- **PACS FedAvg:** 50 rounds, 10 local epochs
- **Office-Home FedAvg+GGEUR:** 50 rounds, **1 local epoch** (not 10!)
- Batch size: 16
- Learning rate: 0.001 (Adam)

#### FederatedScope
Need to check current configs...

---

## Root Cause Analysis

The **primary issue** is that FederatedScope's augmentation uses:
```
X_new = X + ε * λ * ξ
```

While the paper uses:
```
X_new ~ N(X, Σ)  where Σ = ξ @ diag(λ) @ ξ^T
```

The difference is:
- FederatedScope: Samples are independent across components (diagonal covariance)
- Paper: Samples capture **full correlation structure** of the distribution

This is mathematically equivalent to:
- FederatedScope: Generates from **N(X, diag(λ))** in the eigenspace
- Paper: Generates from **N(X, Σ)** in the original space

The correlations between dimensions are **completely lost** in FederatedScope's implementation!

---

## Recommended Fixes

### Priority 1: Fix Sample Generation Algorithm
Implement proper multivariate normal sampling using full covariance matrix:
```python
def generate_samples_with_full_covariance(
    mean: torch.Tensor,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    n_samples: int,
    eigenvalue_scaling: bool = True
) -> torch.Tensor:
    # Step 1: Scale eigenvalues (paper's nearest_pos_def)
    if eigenvalue_scaling:
        scale_factors = torch.ones_like(eigenvalues)
        scale_factors[:10] = torch.linspace(5.0, 1.0, 10)
        eigenvalues = eigenvalues * scale_factors

    # Step 2: Reconstruct covariance matrix
    cov_matrix = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.t()

    # Step 3: Cholesky decomposition with adaptive jitter
    jitter = 1e-6
    identity = torch.eye(cov_matrix.shape[0], device=cov_matrix.device)

    for _ in range(10):  # Max retries
        try:
            L = torch.linalg.cholesky(cov_matrix + jitter * identity)
            break
        except RuntimeError:
            jitter *= 10
    else:
        # Fallback: use diagonal approximation
        L = eigenvectors @ torch.diag(torch.sqrt(eigenvalues))

    # Step 4: Generate samples
    # X_new = mean + L @ z, where z ~ N(0, I)
    z = torch.randn(n_samples, cov_matrix.shape[0], device=mean.device)
    perturbations = torch.mm(z, L.t())  # [n_samples, D]
    new_samples = mean.unsqueeze(0) + perturbations

    return new_samples
```

### Priority 2: Verify Augmentation Parameters
Check and align with paper's parameters for each dataset.

### Priority 3: Check Training Hyperparameters
Ensure local_epochs, learning_rate, etc. match paper settings.

---

## Expected Impact

After fixes:
- ✅ Augmented samples will capture **full correlation structure**
- ✅ Eigenvalue scaling will create **more diverse** samples
- ✅ Should significantly improve accuracy to match paper results
