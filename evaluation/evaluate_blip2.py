import torch
from torch.utils.data import DataLoader
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import sys
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import PeftModel

sys.path.append(str(Path(__file__).parent.parent))
from preprocessing.dataset import SLAKEBLIP2Dataset

def collate_fn(batch, processor):
    """Custom collate function"""
    images = [item['image'] for item in batch]
    prompts = [item['prompt'] for item in batch]
    
    encoding = processor(images=images, text=prompts, 
                         return_tensors="pt", padding=True, truncation=True)
    
    encoding['question_type'] = [item['question_type'] for item in batch]
    encoding['category'] = [item['category'] for item in batch]
    encoding['answers'] = [item['answer'] for item in batch]
    encoding['questions'] = [item['question'] for item in batch]
    
    return encoding

def calculate_metrics(predictions, ground_truths):
    """Calculate evaluation metrics"""
    exact_match = 0
    fuzzy_match = 0
    bleu_scores = []
    
    for pred, gt in zip(predictions, ground_truths):
        pred = pred.strip().lower()
        gt = gt.strip().lower()
        
        # Exact match
        if pred == gt:
            exact_match += 1
            fuzzy_match += 1
        # Fuzzy match (contains)
        elif gt in pred or pred in gt:
            fuzzy_match += 1
        
        # BLEU score
        reference = [gt.split()]
        hypothesis = pred.split()
        bleu = sentence_bleu(reference, hypothesis, weights=(1, 0, 0, 0))
        bleu_scores.append(bleu)
    
    total = len(predictions)
    return {
        'exact_match': 100 * exact_match / total,
        'fuzzy_match': 100 * fuzzy_match / total,
        'bleu1': 100 * np.mean(bleu_scores)
    }

def evaluate_blip2(model, dataloader, processor, device):
    """Evaluate BLIP-2 model"""
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    all_question_types = []
    all_categories = []
    all_questions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                      for k, v in batch.items() 
                      if k not in ['question_type', 'category', 'answers', 'questions']}
            
            # Generate predictions
            generated_ids = model.generate(
                pixel_values=inputs['pixel_values'],
                input_ids=inputs['input_ids'],
                max_length=50,
                num_beams=3
            )
            
            generated_texts = processor.batch_decode(generated_ids, 
                                                     skip_special_tokens=True)
            
            all_predictions.extend(generated_texts)
            all_ground_truths.extend(batch['answers'])
            all_question_types.extend(batch['question_type'])
            all_categories.extend(batch['category'])
            all_questions.extend(batch['questions'])
    
    # Calculate overall metrics
    overall_metrics = calculate_metrics(all_predictions, all_ground_truths)
    
    # Calculate metrics by question type
    question_type_metrics = {}
    for q_type in set(all_question_types):
        mask = [i for i, t in enumerate(all_question_types) if t == q_type]
        type_preds = [all_predictions[i] for i in mask]
        type_gts = [all_ground_truths[i] for i in mask]
        
        metrics = calculate_metrics(type_preds, type_gts)
        metrics['count'] = len(mask)
        question_type_metrics[q_type] = metrics
    
    # Calculate metrics by category
    category_metrics = {}
    for category in set(all_categories):
        mask = [i for i, c in enumerate(all_categories) if c == category]
        cat_preds = [all_predictions[i] for i in mask]
        cat_gts = [all_ground_truths[i] for i in mask]
        
        metrics = calculate_metrics(cat_preds, cat_gts)
        metrics['count'] = len(mask)
        category_metrics[category] = metrics
    
    # Sample predictions
    sample_predictions = []
    for i in range(min(50, len(all_predictions))):
        sample_predictions.append({
            'question': all_questions[i],
            'prediction': all_predictions[i],
            'ground_truth': all_ground_truths[i],
            'question_type': all_question_types[i],
            'category': all_categories[i]
        })
    
    results = {
        'overall': overall_metrics,
        'by_question_type': question_type_metrics,
        'by_category': category_metrics,
        'sample_predictions': sample_predictions
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate BLIP-2 model')
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data/processed/splits')
    parser.add_argument('--images_dir', type=str, default='data/raw/images')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--output_dir', type=str, default='results/blip2_evaluation')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model and processor
    print("Loading BLIP-2 model...")
    processor = Blip2Processor.from_pretrained(args.model_dir)
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16
    ).to(device)
    
    # Create dataset
    test_dataset = SLAKEBLIP2Dataset(
        Path(args.data_dir) / f'{args.split}.json',
        args.images_dir,
        processor
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda batch: collate_fn(batch, processor)
    )
    
    print(f"Evaluating on {len(test_dataset)} samples...")
    
    # Evaluate
    results = evaluate_blip2(model, test_loader, processor, device)
    
    # Print results
    print("\n" + "="*60)
    print("BLIP-2 EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nOverall Performance:")
    print(f"  Exact Match: {results['overall']['exact_match']:.2f}%")
    print(f"  Fuzzy Match: {results['overall']['fuzzy_match']:.2f}%")
    print(f"  BLEU-1: {results['overall']['bleu1']:.2f}")
    
    print(f"\nPerformance by Question Type:")
    for q_type, metrics in results['by_question_type'].items():
        print(f"  {q_type}:")
        print(f"    Exact Match: {metrics['exact_match']:.2f}% ({metrics['count']} samples)")
        print(f"    Fuzzy Match: {metrics['fuzzy_match']:.2f}%")
        print(f"    BLEU-1: {metrics['bleu1']:.2f}")
    
    print(f"\nPerformance by Category:")
    for category, metrics in results['by_category'].items():
        print(f"  {category}:")
        print(f"    Exact Match: {metrics['exact_match']:.2f}% ({metrics['count']} samples)")
        print(f"    Fuzzy Match: {metrics['fuzzy_match']:.2f}%")
    
    # Save results
    output_file = output_path / f'metrics_{args.split}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")

if __name__ == '__main__':
    main()