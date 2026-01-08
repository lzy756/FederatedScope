# CRITICAL DIFFERENCE FOUND - Root Cause of 2% Accuracy

## 🔴 The Key Issue

### Original Code (prototype_cov_matrix_generate_features.py)
```python
# Line 38-39: expand_features_with_cov
# From ALL generated samples, randomly select ONLY 50 samples
selected_indices = np.random.choice(all_generated_samples.shape[0], 50, replace=False)
return all_generated_samples[selected_indices]

# Line 100: Final samples per class
final_samples = combine_samples(original_features, expanded_features,
                                additional_samples, target_size=50)
```

**Result**: Each client has **exactly 50 samples per class** (65 classes × 50 = 3,250 total samples)

### FederatedScope Implementation
```python
# augment_multi_domain
# Step 1: Generate up to step1_target=500 samples
step1_embeddings, step1_labels = self.augment_single_domain(...)

# Step 2: Generate M=500 samples per prototype (3 domains × 500 = 1500)
# ...

# Combine: Step 1 + Step 2 = 500 + 1500 = 2000 samples per class
all_embeddings = torch.cat([step1_embeddings, step2_embeddings], dim=0)
```

**Result**: Each client has **~2000 samples per class** (65 classes × 2000 = 130,000 total samples)

## 📊 Impact

| Aspect | Original | FederatedScope | Ratio |
|--------|----------|----------------|-------|
| Samples per class | 50 | 2000 | **40x** |
| Total samples per client | 3,250 | 130,000 | **40x** |
| Has sample selection | ✅ Yes | ❌ No | - |
| combine_samples logic | ✅ Yes | ❌ No | - |

## 🎯 The combine_samples Logic (Original Code)

```python
def combine_samples(original_features, expanded_features, other_generated_samples, target_size=50):
    num_original = original_features.shape[0]

    # Priority 1: If we have enough original samples, use only original
    if num_original >= target_size:
        selected_indices = np.random.choice(num_original, target_size, replace=False)
        return original_features[selected_indices]

    # Priority 2: Need to fill with generated samples
    num_needed = target_size - num_original

    # Combine all generated samples (expanded + other domains)
    combined_generated_samples = []
    if expanded_features.shape[0] > 0:
        combined_generated_samples.append(expanded_features)
    if other_generated_samples.shape[0] > 0:
        combined_generated_samples.append(other_generated_samples)

    if len(combined_generated_samples) > 0:
        combined_generated_samples = np.vstack(combined_generated_samples)
        # Randomly select num_needed samples from generated
        selected_indices = np.random.choice(combined_generated_samples.shape[0],
                                           num_needed, replace=False)
        # Return: ALL original + selected generated
        final_samples = np.vstack((original_features,
                                   combined_generated_samples[selected_indices]))
    else:
        final_samples = original_features

    return final_samples  # Always exactly target_size or num_original
```

**Key points**:
1. Always tries to keep ALL original samples
2. Only adds generated samples if needed to reach target_size=50
3. Randomly samples from generated pool to fill the gap
4. Final result: exactly 50 samples per class

## 🔧 Required Fix

We need to add `combine_samples` logic to `augment_multi_domain()`:

```python
def augment_multi_domain(self, ..., target_size=50):
    # Step 1: Generate expanded features (select 50 from all)
    step1_embeddings, step1_labels = ...
    # Should randomly select 50 from step1_embeddings here!

    # Step 2: Generate from other domains (50 per prototype)
    step2_embeddings = ...
    # Should generate exactly 50 per prototype, not 500!

    # Step 3: combine_samples logic
    # Return: original samples + selected generated samples = exactly 50 total
```

## 📋 Action Items

1. ✅ Add `combine_samples` function to `ggeur_augmentation.py`
2. ✅ Modify `augment_multi_domain` to use target_size=50
3. ✅ Add random selection after Step 1 (select 50 from all generated)
4. ✅ Change Step 2 to generate 50 samples per prototype (not 500)
5. ✅ Update config: `m_samples_per_prototype: 50` (currently 500)

## 🎓 Why This Causes 2% Accuracy

1. **Data volume mismatch**: Training on 40x more data than expected
2. **No selection strategy**: Original code carefully selects diverse samples
3. **Memory and computation**: Processing 130k samples vs 3,250 samples
4. **Overfitting**: Too many augmented samples may cause overfitting to augmentation artifacts
5. **Distribution mismatch**: The random selection in original code ensures better sample diversity
