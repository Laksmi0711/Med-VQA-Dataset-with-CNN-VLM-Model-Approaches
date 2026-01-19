import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from pathlib import Path
import numpy as np

def plot_training_curves(history_file, output_dir):
    """Plot training curves from history"""
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curve
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curve
    if 'train_acc' in history:
        axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    if 'val_acc' in history:
        axes[1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved training curves to {output_path / 'training_curves.png'}")

def plot_category_performance(metrics_file, output_dir):
    """Plot performance by category"""
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    categories = list(metrics['by_category'].keys())
    accuracies = [metrics['by_category'][cat]['accuracy'] for cat in categories]
    f1_scores = [metrics['by_category'][cat]['f1_score'] for cat in categories]
    counts = [metrics['by_category'][cat]['count'] for cat in categories]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy by category
    colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))
    bars = axes[0].bar(categories, accuracies, color=colors)
    axes[0].set_xlabel('Category', fontsize=12)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Accuracy by Question Category', fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                     f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # F1-Score by category
    bars = axes[1].bar(categories, f1_scores, color=colors)
    axes[1].set_xlabel('Category', fontsize=12)
    axes[1].set_ylabel('F1-Score', fontsize=12)
    axes[1].set_title('F1-Score by Question Category', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, f1 in zip(bars, f1_scores):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                     f'{f1:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / 'category_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved category performance to {output_path / 'category_performance.png'}")

def plot_question_type_comparison(metrics_file, output_dir):
    """Plot comparison between question types"""
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    question_types = list(metrics['by_question_type'].keys())
    accuracies = [metrics['by_question_type'][qt]['accuracy'] for qt in question_types]
    
    plt.figure(figsize=(10, 6))
    colors = ['#3498db', '#e74c3c']
    bars = plt.bar(question_types, accuracies, color=colors[:len(question_types)])
    
    plt.xlabel('Question Type', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Performance: Closed-ended vs Open-ended Questions', 
              fontsize=14, fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / 'question_type_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved question type comparison to {output_path / 'question_type_comparison.png'}")

def main():
    parser = argparse.ArgumentParser(description='Visualize results')
    parser.add_argument('--metrics_file', type=str, required=True)
    parser.add_argument('--history_file', type=str, default=None)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    print("Generating visualizations...")
    
    # Plot metrics
    plot_category_performance(args.metrics_file, args.output_dir)
    plot_question_type_comparison(args.metrics_file, args.output_dir)
    
    # Plot training curves if history provided
    if args.history_file and Path(args.history_file).exists():
        plot_training_curves(args.history_file, args.output_dir)
    
    print(f"\n✓ All visualizations saved to {args.output_dir}")

if __name__ == '__main__':
    main()