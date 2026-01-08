"""
GGEUR_Clip Main Runner - Office-Home-LDS

This script runs all GGEUR_Clip steps in sequence:
1. Data Preparation - Split data and create indices
2. Client-Class Index Matching - Generate per-client per-class indices
3. CLIP Feature Extraction - Extract CLIP features for train and test
4. Covariance Aggregation - Compute global covariance matrices
5. Prototype Computation - Compute class prototypes for each client
6. Feature Generation - GGEUR_Clip core: augment features using Gaussian geometry
7. FedAvg Training - Train and evaluate federated model

Usage:
    python run_ggeur.py --data_path ./data/Office-Home

    # Run specific steps only
    python run_ggeur.py --steps 1 2 3

    # Skip preprocessing (if already done)
    python run_ggeur.py --steps 6 7
"""

import os
import sys
import argparse
import subprocess


def run_step(step_num, step_name, script_name, args_list, workspace):
    """Run a single step"""
    print("\n" + "="*70)
    print(f"STEP {step_num}: {step_name}")
    print("="*70)

    script_path = os.path.join(os.path.dirname(__file__), script_name)

    if not os.path.exists(script_path):
        print(f"Error: Script not found: {script_path}")
        return False

    cmd = [sys.executable, script_path] + args_list
    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error in step {step_num}: {e}")
        return False


def main(args):
    print("="*70)
    print("GGEUR_Clip - Gaussian Geometry-Embedded Universal Representation")
    print("Office-Home-LDS Implementation")
    print("="*70)

    workspace = args.workspace
    os.makedirs(workspace, exist_ok=True)

    steps_to_run = args.steps if args.steps else list(range(1, 8))

    # Step configurations
    step_configs = {
        1: {
            "name": "Data Preparation",
            "script": "step1_data_preparation.py",
            "args": [
                "--data_path", args.data_path,
                "--output_base", workspace,
                "--alpha", str(args.alpha),
                "--seed", str(args.seed)
            ]
        },
        2: {
            "name": "Client-Class Index Matching",
            "script": "step2_client_class_index.py",
            "args": ["--workspace", workspace]
        },
        3: {
            "name": "CLIP Feature Extraction",
            "script": "step3_clip_feature_extraction.py",
            "args": [
                "--data_path", args.data_path,
                "--workspace", workspace,
                "--clip_model_path", args.clip_model_path,
                "--backbone", args.backbone
            ]
        },
        4: {
            "name": "Covariance Matrix Aggregation",
            "script": "step4_covariance_aggregation.py",
            "args": ["--workspace", workspace]
        },
        5: {
            "name": "Prototype Computation",
            "script": "step5_prototype_computation.py",
            "args": ["--workspace", workspace]
        },
        6: {
            "name": "GGEUR_Clip Feature Generation",
            "script": "step6_feature_generation.py",
            "args": [
                "--workspace", workspace,
                "--target_size", str(args.target_size),
                "--seed", str(args.seed)
            ]
        },
        7: {
            "name": "FedAvg Training",
            "script": "step7_fedavg_training.py",
            "args": [
                "--workspace", workspace,
                "--communication_rounds", str(args.communication_rounds),
                "--local_epochs", str(args.local_epochs),
                "--batch_size", str(args.batch_size),
                "--learning_rate", str(args.learning_rate)
            ]
        }
    }

    # Run selected steps
    for step_num in steps_to_run:
        if step_num not in step_configs:
            print(f"Warning: Unknown step {step_num}, skipping...")
            continue

        config = step_configs[step_num]
        success = run_step(step_num, config["name"], config["script"], config["args"], workspace)

        if not success:
            print(f"\nStep {step_num} failed. Stopping execution.")
            sys.exit(1)

    print("\n" + "="*70)
    print("GGEUR_Clip PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nResults saved in: {workspace}")
    print(f"  - Model: {os.path.join(workspace, 'model', 'best_model.pth')}")
    print(f"  - Logs: {os.path.join(workspace, 'results', 'training_log.txt')}")
    print(f"  - Plot: {os.path.join(workspace, 'results', 'accuracy_plot.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='GGEUR_Clip Main Runner for Office-Home-LDS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python run_ggeur.py --data_path ./data/Office-Home

  # Run only training (assuming preprocessing is done)
  python run_ggeur.py --steps 7

  # Run preprocessing only
  python run_ggeur.py --steps 1 2 3 4 5

  # Run feature generation and training
  python run_ggeur.py --steps 6 7
        """
    )

    # Data paths
    parser.add_argument('--data_path', type=str, default='/root/CDA_new/OfficeHomeDataset_10072016',
                        help='Path to Office-Home dataset root')
    parser.add_argument('--workspace', type=str, default='./ggeur_standalone/workspace',
                        help='Workspace directory for intermediate outputs')
    parser.add_argument('--clip_model_path', type=str, default='/root/model/open_clip_vitb16.bin',
                        help='Path to CLIP model weights')
    parser.add_argument('--backbone', type=str, default='ViT-B-16',
                        help='CLIP backbone architecture (ViT-B-16 or ViT-B-32)')

    # Step selection
    parser.add_argument('--steps', type=int, nargs='+', default=None,
                        help='Steps to run (1-7). Default: run all steps')

    # Data preparation parameters
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet concentration parameter')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Feature generation parameters
    parser.add_argument('--target_size', type=int, default=50,
                        help='Target samples per class after augmentation')

    # Training parameters
    parser.add_argument('--communication_rounds', type=int, default=50,
                        help='Number of FedAvg communication rounds')
    parser.add_argument('--local_epochs', type=int, default=1,
                        help='Local training epochs per round')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')

    args = parser.parse_args()
    main(args)
