"""
Test script to verify minimum samples per client guarantee
"""
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class MockDataset:
    """Mock dataset for testing"""
    def __init__(self, targets):
        self.targets = targets


def split_data_by_dirichlet(dataset, num_clients, num_classes, alpha, seed, min_samples_per_client=5):
    """
    Split dataset among clients using Dirichlet distribution for class heterogeneity.
    Ensures each client gets at least min_samples_per_client samples.
    """
    np.random.seed(seed)

    # Get labels from dataset
    labels = np.array(dataset.targets)

    # Group indices by class
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]

    # Initialize client indices
    client_indices = [[] for _ in range(num_clients)]

    # For each class, split samples among clients according to Dirichlet distribution
    for c_idx, c_indices in enumerate(class_indices):
        if len(c_indices) == 0:
            continue

        # Shuffle class indices
        np.random.shuffle(c_indices)

        # Sample from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

        # Balance proportions to match number of samples
        proportions = np.array([p * (len(c_idx) < len(c_indices)) for p, c_idx in zip(proportions, client_indices)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(c_indices)).astype(int)[:-1]

        # Split indices according to proportions
        split_indices = np.split(c_indices, proportions)

        # Assign to clients
        for client_id, indices in enumerate(split_indices):
            client_indices[client_id].extend(indices.tolist())

    # Shuffle each client's data
    for client_id in range(num_clients):
        np.random.shuffle(client_indices[client_id])

    # Post-processing: Ensure minimum samples per client
    if min_samples_per_client > 0:
        # Find clients with insufficient samples
        clients_need_samples = []
        clients_have_extra = []

        for client_id in range(num_clients):
            if len(client_indices[client_id]) < min_samples_per_client:
                clients_need_samples.append(client_id)
            elif len(client_indices[client_id]) > min_samples_per_client * 2:
                clients_have_extra.append(client_id)

        if clients_need_samples:
            logger.info(f"Found {len(clients_need_samples)} clients with < {min_samples_per_client} samples")

            # Redistribute samples from clients with extra to clients with insufficient
            for client_id in clients_need_samples:
                needed = min_samples_per_client - len(client_indices[client_id])

                if needed > 0:
                    # Try to take samples from clients with extra
                    samples_collected = []

                    for donor_id in clients_have_extra:
                        if len(samples_collected) >= needed:
                            break

                        donor_samples = client_indices[donor_id]
                        # Calculate how many we can take
                        can_donate = max(0, len(donor_samples) - min_samples_per_client)
                        to_take = min(can_donate, needed - len(samples_collected))

                        if to_take > 0:
                            # Take samples from the end of donor's list
                            taken = donor_samples[-to_take:]
                            client_indices[donor_id] = donor_samples[:-to_take]
                            samples_collected.extend(taken)

                    # Add collected samples to the client
                    client_indices[client_id].extend(samples_collected)

                    logger.info(f"  Client {client_id}: {len(client_indices[client_id]) - len(samples_collected)} -> {len(client_indices[client_id])} samples (added {len(samples_collected)})")

            # Final check
            still_insufficient = [cid for cid in range(num_clients)
                                 if len(client_indices[cid]) < min_samples_per_client]
            if still_insufficient:
                logger.warning(f"WARNING: {len(still_insufficient)} clients still have < {min_samples_per_client} samples")

    return client_indices


def test_min_samples_guarantee():
    """Test the minimum samples guarantee"""
    print("="*80)
    print("Testing Minimum Samples Per Client Guarantee")
    print("="*80)

    # Simulate PACS scenario: 2576 training samples, 7 classes, 25 clients per domain
    num_samples = 2576
    num_classes = 7
    num_clients = 25
    alpha = 0.1  # High heterogeneity
    min_samples = 5

    # Create mock dataset with balanced classes
    samples_per_class = num_samples // num_classes
    targets = []
    for class_id in range(num_classes):
        targets.extend([class_id] * samples_per_class)
    # Add remaining samples to last class
    remaining = num_samples - len(targets)
    targets.extend([num_classes - 1] * remaining)

    dataset = MockDataset(targets)

    print(f"\nTest Configuration:")
    print(f"  Total samples: {num_samples}")
    print(f"  Classes: {num_classes}")
    print(f"  Samples per class: ~{samples_per_class}")
    print(f"  Clients: {num_clients}")
    print(f"  Dirichlet alpha: {alpha}")
    print(f"  Minimum samples per client: {min_samples}")

    # Test WITHOUT minimum guarantee
    print(f"\n{'='*80}")
    print("WITHOUT Minimum Samples Guarantee:")
    print("="*80)

    client_indices = split_data_by_dirichlet(
        dataset=dataset,
        num_clients=num_clients,
        num_classes=num_classes,
        alpha=alpha,
        seed=123,
        min_samples_per_client=0  # Disabled
    )

    sample_counts = [len(indices) for indices in client_indices]
    empty_clients = sum(1 for c in sample_counts if c == 0)
    insufficient_clients = sum(1 for c in sample_counts if c < min_samples)

    print(f"\nResults:")
    print(f"  Total samples distributed: {sum(sample_counts)}")
    print(f"  Min samples: {min(sample_counts)}")
    print(f"  Max samples: {max(sample_counts)}")
    print(f"  Mean samples: {np.mean(sample_counts):.1f}")
    print(f"  Std samples: {np.std(sample_counts):.1f}")
    print(f"  Clients with 0 samples: {empty_clients} / {num_clients} ({100*empty_clients/num_clients:.1f}%)")
    print(f"  Clients with < {min_samples} samples: {insufficient_clients} / {num_clients} ({100*insufficient_clients/num_clients:.1f}%)")

    # Show distribution
    print(f"\n  Sample distribution: {sorted(sample_counts)}")

    # Test WITH minimum guarantee
    print(f"\n{'='*80}")
    print(f"WITH Minimum Samples Guarantee (min={min_samples}):")
    print("="*80)

    client_indices = split_data_by_dirichlet(
        dataset=dataset,
        num_clients=num_clients,
        num_classes=num_classes,
        alpha=alpha,
        seed=123,
        min_samples_per_client=min_samples  # Enabled
    )

    sample_counts = [len(indices) for indices in client_indices]
    empty_clients = sum(1 for c in sample_counts if c == 0)
    insufficient_clients = sum(1 for c in sample_counts if c < min_samples)

    print(f"\nResults:")
    print(f"  Total samples distributed: {sum(sample_counts)}")
    print(f"  Min samples: {min(sample_counts)}")
    print(f"  Max samples: {max(sample_counts)}")
    print(f"  Mean samples: {np.mean(sample_counts):.1f}")
    print(f"  Std samples: {np.std(sample_counts):.1f}")
    print(f"  Clients with 0 samples: {empty_clients} / {num_clients} ({100*empty_clients/num_clients:.1f}%)")
    print(f"  Clients with < {min_samples} samples: {insufficient_clients} / {num_clients} ({100*insufficient_clients/num_clients:.1f}%)")

    # Show distribution
    print(f"\n  Sample distribution: {sorted(sample_counts)}")

    # Verify all clients have at least min_samples
    print(f"\n{'='*80}")
    print("Verification:")
    print("="*80)

    if all(c >= min_samples for c in sample_counts):
        print(f"✓ SUCCESS: All {num_clients} clients have at least {min_samples} samples!")
    else:
        print(f"✗ FAILED: {insufficient_clients} clients still have < {min_samples} samples")
        for cid, count in enumerate(sample_counts):
            if count < min_samples:
                print(f"  Client {cid}: {count} samples")

    print("\n" + "="*80)


if __name__ == '__main__':
    test_min_samples_guarantee()
