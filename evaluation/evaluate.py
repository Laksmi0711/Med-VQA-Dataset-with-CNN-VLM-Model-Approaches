import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import sys
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from preprocessing.dataset import SLAKEDataset, get_transforms
from models.baseline import CNNLSTMModel

def evaluate_model(model, dataloader, device, vocab):
    """Evaluate model on test set"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_question_types = []
    all_categories = []
    
    idx2answer = {v: k for k, v in vocab.items()}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            images = batch['image'].to(device)
            answer_idx = batch['answer_idx'].to(device)
            
            # Simple question tokenization
            questions = batch['question']
            max_len = 20
            vocab_simple = {'<PAD>': 0}
            question_indices = []
            
            for q in questions:
                tokens = q.lower().split()[:max_len]
                indices = []
                for token in tokens:
                    if token not in vocab_simple:
                        vocab_simple[token] = len(vocab_simple)
                    indices.append(vocab_simple[token])
                indices += [0] * (max_len - len(indices))
                question_indices.append(indices)
            
            question_indices = torch.tensor(question_indices, dtype=torch.long).to(device)
            
            # Forward pass
            logits = model(images, question_indices)
            _, predicted = torch.max(logits, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(answer_idx.cpu().numpy())
            all_question_types.extend(batch['question_type'])
            all_categories.extend(batch['category'])
    
    # Calculate overall metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    overall_acc = 100 * np.mean(all_predictions == all_targets)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='weighted', zero_division=0
    )
    
    # Calculate metrics by question type
    question_type_metrics = {}
    for q_type in set(all_question_types):
        mask = np.array(all_question_types) == q_type
        type_acc = 100 * np.mean(all_predictions[mask] == all_targets[mask])
        type_p, type_r, type_f1, _ = precision_recall_fscore_support(
            all_targets[mask], all_predictions[mask], 
            average='weighted', zero_division=0
        )
        question_type_metrics[q_type] = {
            'accuracy': float(type_acc),
            'precision': float(type_p),
            'recall': float(type_r),
            'f1_score': float(type_f1),
            'count': int(mask.sum())
        }
    
    # Calculate metrics by category
    category_metrics = {}
    for category in set(all_categories):
        mask = np.array(all_categories) == category
        cat_acc = 100 * np.mean(all_predictions[mask] == all_targets[mask])
        cat_p, cat_r, cat_f1, _ = precision_recall_fscore_support(
            all_targets[mask], all_predictions[mask], 
            average='weighted', zero_division=0
        )
        category_metrics[category] = {
            'accuracy': float(cat_acc),
            'precision': float(cat_p),
            'recall': float(cat_r),
            'f1_score': float(cat_f1),
            'count': int(mask.sum())
        }
    
    results = {
        'overall': {
            'accuracy': float(overall_acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        },
        'by_question_type': question_type_metrics,
        'by_category': category_metrics
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate baseline model')
    parser.add_argument('--model_type', type=str, default='baseline')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data/processed/splits')
    parser.add_argument('--images_dir', type=str, default='data/raw/images')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--output_dir', type=str, default='results/baseline_evaluation')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load vocabulary
    vocab_file = Path(args.data_dir).parent / 'vocab.json'
    with open(vocab_file, 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    
    # Load model
    print("Loading model...")
    model = CNNLSTMModel(vocab_size=vocab_size).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create dataset
    test_dataset = SLAKEDataset(
        Path(args.data_dir) / f'{args.split}.json',
        args.images_dir,
        vocab_file,
        transform=get_transforms('test')
    )
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4)
    
    print(f"Evaluating on {len(test_dataset)} samples...")
    
    # Evaluate
    results = evaluate_model(model, test_loader, device, vocab)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nOverall Performance:")
    print(f"  Accuracy: {results['overall']['accuracy']:.2f}%")
    print(f"  Precision: {results['overall']['precision']:.3f}")
    print(f"  Recall: {results['overall']['recall']:.3f}")
    print(f"  F1-Score: {results['overall']['f1_score']:.3f}")
    
    print(f"\nPerformance by Question Type:")
    for q_type, metrics in results['by_question_type'].items():
        print(f"  {q_type}:")
        print(f"    Accuracy: {metrics['accuracy']:.2f}% ({metrics['count']} samples)")
        print(f"    F1-Score: {metrics['f1_score']:.3f}")
    
    print(f"\nPerformance by Category:")
    for category, metrics in results['by_category'].items():
        print(f"  {category}:")
        print(f"    Accuracy: {metrics['accuracy']:.2f}% ({metrics['count']} samples)")
        print(f"    F1-Score: {metrics['f1_score']:.3f}")
    
    # Save results
    output_file = output_path / f'metrics_{args.split}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")

if __name__ == '__main__':
    main()