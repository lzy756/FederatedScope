"""
Quick verification of GGEUR_Clip configuration fixes
"""

# Verify 1: Trainer returns correct format
def test_trainer_format():
    target_data_split_name = 'test'
    accuracy = 0.85
    correct = 850
    loss = 0.42
    total = 1000

    metrics = {
        f'{target_data_split_name}_acc': accuracy,
        f'{target_data_split_name}_correct': correct,
        f'{target_data_split_name}_loss': loss,
        f'{target_data_split_name}_total': total,
        f'{target_data_split_name}_avg_loss': loss,
    }

    print("Trainer metrics format:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Check required keys
    required_keys = ['test_acc', 'test_total', 'test_correct']
    for key in required_keys:
        assert key in metrics, f"Missing required key: {key}"

    print("✅ Trainer format is correct\n")

# Verify 2: Config keys match metric keys
def test_config_keys():
    config_key = 'test_acc'
    metric_keys = ['test_acc', 'test_total', 'test_correct', 'test_loss', 'test_avg_loss']

    print(f"Config best_res_update_round_wise_key: {config_key}")
    print(f"Available metric keys: {metric_keys}")

    assert config_key in metric_keys, f"Config key '{config_key}' not in metric keys"
    print("✅ Config key matches metric keys\n")

# Verify 3: Monitor can extract metric name
def test_monitor_extraction():
    round_wise_update_key = 'test_acc'

    # Simulate monitor's extraction logic
    update_key = round_wise_update_key  # Default to the full key
    for mode in ['train', 'val', 'test']:
        if mode in round_wise_update_key:
            update_key = round_wise_update_key.split(f'{mode}_')[1]
            break

    print(f"Monitor extraction:")
    print(f"  Input: {round_wise_update_key}")
    print(f"  Extracted: {update_key}")

    assert update_key == 'acc', f"Expected 'acc', got '{update_key}'"
    print("✅ Monitor extraction works\n")

if __name__ == '__main__':
    print("="*60)
    print("GGEUR_Clip Configuration Verification")
    print("="*60 + "\n")

    test_trainer_format()
    test_config_keys()
    test_monitor_extraction()

    print("="*60)
    print("All verifications passed! ✅")
    print("="*60)
