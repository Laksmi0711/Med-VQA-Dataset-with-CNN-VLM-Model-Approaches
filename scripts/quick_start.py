import subprocess
import argparse
from pathlib import Path
import sys

def run_command(cmd):
    """Run a command and handle errors"""
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    
    print(result.stdout)

def main():
    parser = argparse.ArgumentParser(description='Quick start test')
    parser.add_argument('--subset_size', type=int, default=100,
                        help='Number of samples for quick test')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs for quick test')
    args = parser.parse_args()
    
    print("="*70)
    print("QUICK START TEST - Testing pipeline on small subset")
    print("="*70)
    
    # Check if data exists
    if not Path('data/raw/slake_qa.json').exists():
        print("\n[1/4] Downloading dataset...")
        run_command(['python', 'scripts/download_dataset.py'])
    else:
        print("\n[1/4] Dataset already downloaded ✓")
    
    # Preprocess
    if not Path('data/processed/splits/train.json').exists():
        print("\n[2/4] Preprocessing data...")
        run_command(['python', 'scripts/preprocess_data.py'])
    else:
        print("\n[2/4] Data already preprocessed ✓")
    
    # Train baseline (quick)
    print(f"\n[3/4] Training baseline model ({args.epochs} epochs)...")
    run_command([
        'python', 'training/train_baseline.py',
        '--epochs', str(args.epochs),
        '--batch_size', '16',
        '--output_dir', 'results/quick_test_baseline'
    ])
    
    # Evaluate
    print("\n[4/4] Evaluating model...")
    run_command([
        'python', 'evaluation/evaluate.py',
        '--checkpoint', 'results/quick_test_baseline/best_model.pth',
        '--output_dir', 'results/quick_test_evaluation'
    ])
    
    print("\n" + "="*70)
    print("QUICK START TEST COMPLETE!")
    print("="*70)
    print("\nResults saved to:")
    print("  - Model: results/quick_test_baseline/")
    print("  - Metrics: results/quick_test_evaluation/")
    print("\nTo run full training, use: bash scripts/run_full_experiments.sh")

if __name__ == '__main__':
    main()
