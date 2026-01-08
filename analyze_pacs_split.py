"""
Analyze PACS data split logic to understand the empty client issue
"""
import numpy as np

def simulate_pacs_split():
    """
    Simulate PACS data splitting with current configuration
    """
    print("="*80)
    print("PACS Data Split Analysis")
    print("="*80)

    # Configuration
    num_domains = 4
    domains = ['photo', 'art_painting', 'cartoon', 'sketch']
    num_classes = 7
    samples_per_class = 500  # Approximate average
    total_samples_per_domain = num_classes * samples_per_class  # ~3500

    client_num = 100
    server_test_samples_per_class = 40
    splits = [0.8, 0.1, 0.1]  # train, val, test
    dirichlet_alpha = 0.1

    print(f"\nConfiguration:")
    print(f"  Total clients: {client_num}")
    print(f"  Domains: {num_domains}")
    print(f"  Classes per domain: {num_classes}")
    print(f"  Approx samples per domain: {total_samples_per_domain}")
    print(f"  Server held-out per class: {server_test_samples_per_class}")
    print(f"  Server held-out total per domain: {server_test_samples_per_class * num_classes}")
    print(f"  Splits (train/val/test): {splits}")
    print(f"  Dirichlet alpha: {dirichlet_alpha}")

    print(f"\n{'='*80}")
    print("Step 1: Server Held-out Test Set Creation")
    print("="*80)

    server_test_per_domain = server_test_samples_per_class * num_classes
    remaining_per_domain = total_samples_per_domain - server_test_per_domain

    print(f"  Original samples per domain: {total_samples_per_domain}")
    print(f"  Server held-out per domain: {server_test_per_domain}")
    print(f"  Remaining for clients per domain: {remaining_per_domain}")
    print(f"  Remaining percentage: {100 * remaining_per_domain / total_samples_per_domain:.1f}%")

    print(f"\n{'='*80}")
    print("Step 2: Client Data Distribution")
    print("="*80)

    # Calculate clients per domain
    clients_per_domain = [client_num // num_domains] * num_domains
    remainder = client_num % num_domains
    for i in range(remainder):
        clients_per_domain[i] += 1

    print(f"\n  Clients per domain: {clients_per_domain}")
    print(f"  Total: {sum(clients_per_domain)}")

    print(f"\n{'='*80}")
    print("Step 3: Data Split per Client")
    print("="*80)

    for domain_id, domain in enumerate(domains):
        num_clients_this_domain = clients_per_domain[domain_id]

        print(f"\n  Domain '{domain}':")
        print(f"    Clients: {num_clients_this_domain}")
        print(f"    Remaining samples: {remaining_per_domain}")

        # Train split
        train_samples = int(remaining_per_domain * splits[0])
        val_samples = int(remaining_per_domain * splits[1])
        test_samples = remaining_per_domain - train_samples - val_samples

        print(f"\n    After train/val/test split:")
        print(f"      Train: {train_samples} samples")
        print(f"      Val: {val_samples} samples")
        print(f"      Test: {test_samples} samples")

        # Dirichlet distribution simulation
        print(f"\n    With Dirichlet (alpha={dirichlet_alpha}) split:")

        # Simulate Dirichlet split
        np.random.seed(123)

        # For each class, calculate expected samples per client
        expected_train_per_client = train_samples / num_clients_this_domain
        expected_val_per_client = val_samples / num_clients_this_domain
        expected_test_per_client = test_samples / num_clients_this_domain

        print(f"      Expected train samples per client: {expected_train_per_client:.1f}")
        print(f"      Expected val samples per client: {expected_val_per_client:.1f}")
        print(f"      Expected test samples per client: {expected_test_per_client:.1f}")

        # Simulate Dirichlet distribution for one class
        print(f"\n    Simulating Dirichlet split for one class:")
        class_samples = train_samples // num_classes
        proportions = np.random.dirichlet(np.repeat(dirichlet_alpha, num_clients_this_domain))
        client_class_samples = (proportions * class_samples).astype(int)

        print(f"      Class samples: {class_samples}")
        print(f"      Min samples per client (this class): {client_class_samples.min()}")
        print(f"      Max samples per client (this class): {client_class_samples.max()}")
        print(f"      Mean samples per client (this class): {client_class_samples.mean():.1f}")
        print(f"      Clients with 0 samples (this class): {(client_class_samples == 0).sum()}")

        # Calculate expected total per client (across all classes)
        total_expected = client_class_samples.sum() * num_classes
        print(f"\n      Expected total train samples across {num_classes} classes: {total_expected}")
        print(f"      Min expected per client (all classes): ~{client_class_samples.min() * num_classes}")
        print(f"      Max expected per client (all classes): ~{client_class_samples.max() * num_classes}")

    print(f"\n{'='*80}")
    print("Analysis Summary")
    print("="*80)

    print(f"\nWith {client_num} clients and {num_domains} domains:")
    print(f"  - Each domain is split among {clients_per_domain[0]}-{clients_per_domain[-1]} clients")
    print(f"  - After server held-out: {remaining_per_domain} samples/domain")
    print(f"  - After train split (80%): {int(remaining_per_domain * 0.8)} samples/domain")
    print(f"  - Per client (uniform): ~{int(remaining_per_domain * 0.8) / clients_per_domain[0]:.1f} train samples")

    print(f"\n⚠️  ISSUE WITH DIRICHLET DISTRIBUTION:")
    print(f"  - With alpha={dirichlet_alpha} (high heterogeneity)")
    print(f"  - Some clients get very few samples per class")
    print(f"  - Across {num_classes} classes, some clients might get ZERO samples for ALL classes")
    print(f"  - This is especially likely when samples per client is already small")

    print(f"\n✓ RECOMMENDED FIXES:")
    print(f"  1. Reduce client_num to 4-30 (match or slightly exceed domain count)")
    print(f"  2. Increase dirichlet_alpha to 0.5-1.0 (less heterogeneity)")
    print(f"  3. Reduce server_test_samples_per_class to 20-30")
    print(f"  4. Add minimum samples per client validation in code")

if __name__ == '__main__':
    simulate_pacs_split()
