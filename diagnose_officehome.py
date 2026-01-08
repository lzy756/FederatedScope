#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diagnose OfficeHome Dataset Loading and GGEUR_Clip Training

This script systematically checks each component of the GGEUR_Clip pipeline
to identify where the accuracy problem occurs.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_dataset_loading():
    """Test 1: Check if OfficeHome dataset loads correctly."""
    print("\n" + "="*80)
    print("TEST 1: OfficeHome Dataset Loading")
    print("="*80)

    try:
        from federatedscope.cv.dataset.office_home import OfficeHome

        dataset = OfficeHome(
            root='OfficeHomeDataset_10072016/',
            domain='Art',
            split='train',
            transform=None,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=12345
        )

        print(f"✓ Dataset loaded: {len(dataset)} samples")
        print(f"  Number of classes: {len(OfficeHome.CLASSES)}")
        print(f"  Classes: {OfficeHome.CLASSES[:5]}... (showing first 5)")

        # Check label distribution
        labels = [dataset.targets[i] for i in range(len(dataset))]
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  Unique labels: {len(unique)}")
        print(f"  Label range: {min(labels)} to {max(labels)}")
        print(f"  Samples per class (first 10): {dict(zip(unique[:10], counts[:10]))}")

        # Load a sample
        img, label = dataset[0]
        print(f"  Sample 0: image shape={img.shape if hasattr(img, 'shape') else 'N/A'}, label={label}")

        return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_clip_features():
    """Test 2: Check CLIP feature extraction."""
    print("\n" + "="*80)
    print("TEST 2: CLIP Feature Extraction")
    print("="*80)

    try:
        from federatedscope.cv.dataset.office_home import OfficeHome
        from federatedscope.contrib.utils.clip_extractor import CLIPExtractor
        from torchvision import transforms

        # Load a small subset
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        dataset = OfficeHome(
            root='OfficeHomeDataset_10072016/',
            domain='Art',
            split='train',
            transform=transform,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=12345
        )

        # Take first 10 samples
        subset_indices = list(range(min(10, len(dataset))))
        subset = torch.utils.data.Subset(dataset, subset_indices)

        print(f"Testing CLIP extraction on {len(subset)} samples...")

        clip_extractor = CLIPExtractor(
            model_name='ViT-B/16',
            device='cuda' if torch.cuda.is_available() else 'cpu',
            batch_size=4
        )

        result = clip_extractor.extract_dataset_features(subset, use_cache=False)

        embeddings = result['embeddings']
        labels = result['labels']

        print(f"✓ CLIP extraction successful")
        print(f"  Embeddings shape: {embeddings.shape}")
        print(f"  Embeddings dtype: {embeddings.dtype}")
        print(f"  Embeddings range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
        print(f"  Embeddings mean: {embeddings.mean():.3f}, std: {embeddings.std():.3f}")
        print(f"  Labels: {labels.tolist()}")

        # Check if embeddings are normalized
        norms = torch.norm(embeddings, dim=1)
        print(f"  L2 norms: mean={norms.mean():.3f}, std={norms.std():.3f}")
        print(f"  (Should be ~1.0 if normalized)")

        return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mlp_model():
    """Test 3: Check MLP model architecture and forward pass."""
    print("\n" + "="*80)
    print("TEST 3: MLP Model Architecture")
    print("="*80)

    try:
        # Build MLP manually
        input_dim = 512
        num_classes = 65

        mlp = nn.Sequential(
            nn.Linear(input_dim, num_classes)
        )

        print(f"✓ MLP model created")
        print(f"  Input dim: {input_dim}")
        print(f"  Output dim: {num_classes}")
        print(f"  Architecture: {mlp}")

        # Test forward pass
        batch_size = 4
        dummy_input = torch.randn(batch_size, input_dim)

        output = mlp(dummy_input)

        print(f"\n  Forward pass test:")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"  Output sample: {output[0, :5].tolist()}")

        # Test with CrossEntropyLoss
        dummy_labels = torch.randint(0, num_classes, (batch_size,))
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output, dummy_labels)

        print(f"\n  Loss test:")
        print(f"  Labels: {dummy_labels.tolist()}")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Expected loss for random: ~{np.log(num_classes):.4f}")

        # Test predictions
        preds = output.argmax(dim=1)
        accuracy = (preds == dummy_labels).float().mean()
        print(f"  Random accuracy: {accuracy.item():.4f}")
        print(f"  Expected random accuracy: ~{1/num_classes:.4f}")

        return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_loop():
    """Test 4: Check a simple training loop."""
    print("\n" + "="*80)
    print("TEST 4: Simple Training Loop")
    print("="*80)

    try:
        from federatedscope.cv.dataset.office_home import OfficeHome
        from federatedscope.contrib.utils.clip_extractor import CLIPExtractor
        from torchvision import transforms

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        # Load dataset
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        train_dataset = OfficeHome(
            root='OfficeHomeDataset_10072016/',
            domain='Art',
            split='train',
            transform=transform,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=12345
        )

        test_dataset = OfficeHome(
            root='OfficeHomeDataset_10072016/',
            domain='Art',
            split='test',
            transform=transform,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=12345
        )

        print(f"Train samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")

        # Extract CLIP features
        print("\nExtracting CLIP features...")
        clip_extractor = CLIPExtractor(
            model_name='ViT-B/16',
            device=device,
            batch_size=16
        )

        # Use subset for quick test
        train_subset = torch.utils.data.Subset(train_dataset, range(min(100, len(train_dataset))))
        test_subset = torch.utils.data.Subset(test_dataset, range(min(50, len(test_dataset))))

        train_result = clip_extractor.extract_dataset_features(train_subset, use_cache=False)
        test_result = clip_extractor.extract_dataset_features(test_subset, use_cache=False)

        train_embeddings = train_result['embeddings'].to(device)
        train_labels = train_result['labels'].to(device)
        test_embeddings = test_result['embeddings'].to(device)
        test_labels = test_result['labels'].to(device)

        print(f"Train embeddings: {train_embeddings.shape}")
        print(f"Test embeddings: {test_embeddings.shape}")

        # Create MLP model
        input_dim = 512
        num_classes = 65

        model = nn.Sequential(
            nn.Linear(input_dim, num_classes)
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        print("\nTraining for 10 epochs...")
        batch_size = 16

        for epoch in range(10):
            model.train()

            # Shuffle training data
            perm = torch.randperm(len(train_embeddings))
            train_embeddings_shuffled = train_embeddings[perm]
            train_labels_shuffled = train_labels[perm]

            total_loss = 0
            total_correct = 0
            num_batches = 0

            for i in range(0, len(train_embeddings), batch_size):
                batch_embeddings = train_embeddings_shuffled[i:i+batch_size]
                batch_labels = train_labels_shuffled[i:i+batch_size]

                optimizer.zero_grad()

                logits = model(batch_embeddings)
                loss = criterion(logits, batch_labels)

                loss.backward()
                optimizer.step()

                # Calculate accuracy
                preds = logits.argmax(dim=1)
                correct = (preds == batch_labels).sum().item()

                total_loss += loss.item()
                total_correct += correct
                num_batches += 1

            train_acc = total_correct / len(train_embeddings)
            avg_loss = total_loss / num_batches

            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                test_logits = model(test_embeddings)
                test_preds = test_logits.argmax(dim=1)
                test_correct = (test_preds == test_labels).sum().item()
                test_acc = test_correct / len(test_labels)
                test_loss = criterion(test_logits, test_labels).item()

            print(f"Epoch {epoch+1:2d}: "
                  f"Train Loss={avg_loss:.4f}, Train Acc={train_acc:.4f} | "
                  f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

        print(f"\n✓ Training completed")
        print(f"  Final train accuracy: {train_acc:.4f}")
        print(f"  Final test accuracy: {test_acc:.4f}")

        if test_acc < 0.05:
            print(f"  ⚠ WARNING: Test accuracy is very low ({test_acc:.4f})")
            print(f"  This suggests a problem with the model or data")

        return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all diagnostic tests."""
    print("\n" + "="*80)
    print("GGEUR_Clip DIAGNOSTIC SUITE")
    print("="*80)

    tests = [
        ("Dataset Loading", test_dataset_loading),
        ("CLIP Feature Extraction", test_clip_features),
        ("MLP Model", test_mlp_model),
        ("Training Loop", test_training_loop),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} {test_name}")

    print("="*80)

    if all(results.values()):
        print("\n✓ All tests passed! The pipeline seems to be working correctly.")
        print("  If accuracy is still low in federated training, check:")
        print("  1. Data distribution across clients")
        print("  2. Client sampling strategy")
        print("  3. GGEUR_Clip augmentation parameters")
    else:
        print("\n✗ Some tests failed. Please fix the issues above before running experiments.")


if __name__ == '__main__':
    main()
