import os
import json
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
from collections import Counter

def load_qa_data(qa_file):
    """Load QA pairs from JSON file"""
    with open(qa_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def categorize_question(question, answer, question_type, content_type):
    """Categorize question based on type and content"""
    question_lower = question.lower()
    
    # Determine if closed-ended or open-ended
    if question_type in ['CLOSED', 'CHOICE']:
        format_type = 'closed'
    else:
        format_type = 'open'
    
    # Categorize by content
    if content_type:
        category = content_type.lower()
    else:
        # Fallback categorization based on keywords
        if any(word in question_lower for word in ['modality', 'ct', 'mri', 'x-ray', 'imaging']):
            category = 'modality'
        elif any(word in question_lower for word in ['organ', 'liver', 'kidney', 'lung', 'brain']):
            category = 'organ'
        elif any(word in question_lower for word in ['abnormal', 'disease', 'lesion', 'pathology']):
            category = 'abnormality'
        elif any(word in question_lower for word in ['where', 'position', 'location', 'lobe']):
            category = 'position'
        else:
            category = 'other'
    
    return format_type, category

def create_vocabulary(qa_data, min_freq=2):
    """Create answer vocabulary from training data"""
    answer_counter = Counter()
    
    for entry in qa_data:
        answer = entry['answer'].strip().lower()
        answer_counter[answer] += 1
    
    # Keep answers with frequency >= min_freq
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    
    for answer, freq in answer_counter.most_common():
        if freq >= min_freq:
            vocab[answer] = idx
            idx += 1
    
    return vocab

def preprocess_dataset(raw_dir, output_dir):
    """Preprocess SLAKE dataset"""
    print("Starting data preprocessing...")
    
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load QA data
    qa_file = raw_path / "slake_qa.json"
    if not qa_file.exists():
        raise FileNotFoundError(f"QA file not found: {qa_file}")
    
    qa_data = load_qa_data(qa_file)
    print(f"Loaded {len(qa_data)} QA pairs")
    
    # Process each entry
    processed_data = []
    for entry in qa_data:
        format_type, category = categorize_question(
            entry['question'],
            entry['answer'],
            entry.get('question_type', 'OPEN'),
            entry.get('content_type', '')
        )
        
        processed_entry = {
            'img_id': entry['img_id'],
            'question': entry['question'].strip(),
            'answer': entry['answer'].strip().lower(),
            'question_type': format_type,
            'category': category
        }
        processed_data.append(processed_entry)
    
    # Convert to DataFrame
    df = pd.DataFrame(processed_data)
    
    # Create vocabulary
    vocab = create_vocabulary(processed_data)
    print(f"Created vocabulary with {len(vocab)} answers")
    
    # Add answer indices
    df['answer_idx'] = df['answer'].apply(lambda x: vocab.get(x, vocab['<UNK>']))
    
    # Split dataset (image-level to prevent leakage)
    unique_images = df['img_id'].unique()
    
    # Split: 70% train, 15% val, 15% test
    train_imgs, temp_imgs = train_test_split(unique_images, test_size=0.3, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
    
    # Create splits
    train_df = df[df['img_id'].isin(train_imgs)]
    val_df = df[df['img_id'].isin(val_imgs)]
    test_df = df[df['img_id'].isin(test_imgs)]
    
    # Save splits
    splits_dir = output_path / "splits"
    splits_dir.mkdir(exist_ok=True)
    
    train_df.to_json(splits_dir / "train.json", orient='records', indent=2)
    val_df.to_json(splits_dir / "val.json", orient='records', indent=2)
    test_df.to_json(splits_dir / "test.json", orient='records', indent=2)
    
    # Save vocabulary
    with open(output_path / "vocab.json", 'w') as f:
        json.dump(vocab, f, indent=2)
    
    # Save statistics
    stats = {
        'total_qa_pairs': len(df),
        'total_images': len(unique_images),
        'train_qa': len(train_df),
        'val_qa': len(val_df),
        'test_qa': len(test_df),
        'vocab_size': len(vocab),
        'question_types': df['question_type'].value_counts().to_dict(),
        'categories': df['category'].value_counts().to_dict()
    }
    
    with open(output_path / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n✓ Preprocessing complete!")
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_df)} QA pairs ({len(train_imgs)} images)")
    print(f"  Val: {len(val_df)} QA pairs ({len(val_imgs)} images)")
    print(f"  Test: {len(test_df)} QA pairs ({len(test_imgs)} images)")
    
    print(f"\nQuestion type distribution:")
    for q_type, count in stats['question_types'].items():
        print(f"  {q_type}: {count}")
    
    print(f"\nCategory distribution:")
    for cat, count in stats['categories'].items():
        print(f"  {cat}: {count}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess SLAKE dataset')
    parser.add_argument('--raw_dir', type=str, default='data/raw',
                        help='Directory containing raw data')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='Directory to save processed data')
    args = parser.parse_args()
    
    preprocess_dataset(args.raw_dir, args.output_dir)

if __name__ == '__main__':
    main()