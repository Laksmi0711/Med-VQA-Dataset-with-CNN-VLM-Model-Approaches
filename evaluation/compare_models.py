import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_metrics(file_path):
    """Load metrics from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def compare_overall_performance(baseline_metrics, blip2_metrics, output_dir):
    """Compare overall performance"""
    models = ['CNN-LSTM', 'BLIP-2']
    
    # For baseline: use accuracy
    # For BLIP-2: use exact_match as comparable metric
    baseline_acc = baseline_metrics['overall']['accuracy']
    blip2_acc = blip2_metrics['overall'].get('exact_match', 
                                             blip2_metrics['overall'].get('accuracy', 0))
    
    accuracies = [baseline_acc, blip2_acc]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#3498db', '#2ecc71']
    bars = ax.bar(models, accuracies, color=colors, width=0.6)
    
    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_title('Overall Model Performance Comparison', fontsize=15, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom', 
                fontsize=12, fontweight='bold')
    
    # Add improvement annotation
    improvement = blip2_acc - baseline_acc
    ax.annotate(f'+{improvement:.1f}%\nimprovement',
                xy=(1, blip2_acc), xytext=(1.3, (baseline_acc + blip2_acc)/2),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, color='red', fontweight='bold')
    
    plt.tight_layout()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / 'overall_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved overall comparison to {output_path / 'overall_comparison.png'}")

def compare_by_category(baseline_metrics, blip2_metrics, output_dir):
    """Compare performance by category"""
    categories = list(baseline_metrics['by_category'].keys())
    
    baseline_accs = [baseline_metrics['by_category'][cat]['accuracy'] for cat in categories]
    
    # For BLIP-2, use exact_match
    blip2_accs = []
    for cat in categories:
        if cat in blip2_metrics['by_category']:
            blip2_accs.append(blip2_metrics['by_category'][cat].get('exact_match',
                             blip2_metrics['by_category'][cat].get('accuracy', 0)))
        else:
            blip2_accs.append(0)
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, baseline_accs, width, label='CNN-LSTM', color='#3498db')
    bars2 = ax.bar(x + width/2, blip2_accs, width, label='BLIP-2', color='#2ecc71')
    
    ax.set_xlabel('Question Category', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Performance Comparison by Question Category', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = Path(output_dir)
    plt.savefig(output_path / 'category_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved category comparison to {output_path / 'category_comparison.png'}")

def generate_comparison_report(baseline_metrics, blip2_metrics, output_dir):
    """Generate text comparison report"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_file = output_path / 'comparison_report.txt'
    
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MODEL COMPARISON REPORT: CNN-LSTM vs BLIP-2\n")
        f.write("="*70 + "\n\n")
        
        # Overall performance
        f.write("OVERALL PERFORMANCE\n")
        f.write("-"*70 + "\n")
        baseline_acc = baseline_metrics['overall']['accuracy']
        blip2_acc = blip2_metrics['overall'].get('exact_match', 
                                                 blip2_metrics['overall'].get('accuracy', 0))
        f.write(f"CNN-LSTM Baseline:  {baseline_acc:.2f}%\n")
        f.write(f"BLIP-2 VLM:         {blip2_acc:.2f}%\n")
        f.write(f"Improvement:        +{blip2_acc - baseline_acc:.2f}%\n\n")
        
        # By category
        f.write("PERFORMANCE BY CATEGORY\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Category':<15} {'CNN-LSTM':<12} {'BLIP-2':<12} {'Improvement'}\n")
        f.write("-"*70 + "\n")
        
        for cat in baseline_metrics['by_category'].keys():
            baseline_cat_acc = baseline_metrics['by_category'][cat]['accuracy']
            blip2_cat_acc = blip2_metrics['by_category'].get(cat, {}).get('exact_match',
                           blip2_metrics['by_category'].get(cat, {}).get('accuracy', 0))
            improvement = blip2_cat_acc - baseline_cat_acc
            f.write(f"{cat:<15} {baseline_cat_acc:>6.2f}%     {blip2_cat_acc:>6.2f}%     ")
            f.write(f"{'+' if improvement >= 0 else ''}{improvement:.2f}%\n")
        
        f.write("\n")
        
        # By question type
        f.write("PERFORMANCE BY QUESTION TYPE\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Type':<15} {'CNN-LSTM':<12} {'BLIP-2':<12} {'Improvement'}\n")
        f.write("-"*70 + "\n")
        
        for qtype in baseline_metrics['by_question_type'].keys():
            baseline_type_acc = baseline_metrics['by_question_type'][qtype]['accuracy']
            blip2_type_acc = blip2_metrics['by_question_type'].get(qtype, {}).get('exact_match',
                            blip2_metrics['by_question_type'].get(qtype, {}).get('accuracy', 0))
            improvement = blip2_type_acc - baseline_type_acc
            f.write(f"{qtype:<15} {baseline_type_acc:>6.2f}%     {blip2_type_acc:>6.2f}%     ")
            f.write(f"{'+' if improvement >= 0 else ''}{improvement:.2f}%\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"✓ Saved comparison report to {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Compare models')
    parser.add_argument('--baseline_metrics', type=str, required=True)
    parser.add_argument('--blip2_metrics', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    print("Loading metrics...")
    baseline_metrics = load_metrics(args.baseline_metrics)
    blip2_metrics = load_metrics(args.blip2_metrics)
    
    print("Generating comparisons...")
    compare_overall_performance(baseline_metrics, blip2_metrics, args.output_dir)
    compare_by_category(baseline_metrics, blip2_metrics, args.output_dir)
    generate_comparison_report(baseline_metrics, blip2_metrics, args.output_dir)
    
    print(f"\n✓ All comparisons saved to {args.output_dir}")

if __name__ == '__main__':
    main()