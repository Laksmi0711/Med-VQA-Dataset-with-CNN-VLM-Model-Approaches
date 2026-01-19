import os
import argparse
from datasets import load_dataset
from pathlib import Path
import json
import shutil
from PIL import Image
from tqdm import tqdm

def download_slake_dataset(output_dir):
    """Download SLAKE dataset from Hugging Face"""
    print("Downloading SLAKE dataset from Hugging Face...")
    
    # Create output directories
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset from Hugging Face
    try:
        dataset = load_dataset("BoKelvin/SLAKE", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Attempting alternative download method...")
        dataset = load_dataset("BoKelvin/SLAKE", split="train")
    
    # Process and save data
    all_data = []
    
    for split_name, split_data in dataset.items():
        print(f"\nProcessing {split_name} split ({len(split_data)} examples)...")
        
        for idx, example in enumerate(tqdm(split_data)):
            # Save image
            if 'image' in example and example['image'] is not None:
                img = example['image']
                img_filename = f"img_{split_name}_{idx:05d}.jpg"
                img_path = images_dir / img_filename
                
                if isinstance(img, Image.Image):
                    img.save(img_path)
                else:
                    Image.fromarray(img).save(img_path)
                
                # Create QA entry
                qa_entry = {
                    'img_id': img_filename,
                    'question': example.get('question', ''),
                    'answer': example.get('answer', ''),
                    'q_lang': example.get('q_lang', 'en'),
                    'question_type': example.get('question_type', 'OPEN'),
                    'content_type': example.get('content_type', 'OTHER'),
                    'split': split_name
                }
                all_data.append(qa_entry)
    
    # Save QA pairs as JSON
    qa_file = output_path / "slake_qa.json"
    with open(qa_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Dataset downloaded successfully!")
    print(f"  Images: {images_dir} ({len(list(images_dir.glob('*.jpg')))} files)")
    print(f"  QA pairs: {qa_file} ({len(all_data)} entries)")
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Total QA pairs: {len(all_data)}")
    
    question_types = {}
    content_types = {}
    for entry in all_data:
        q_type = entry.get('question_type', 'UNKNOWN')
        c_type = entry.get('content_type', 'UNKNOWN')
        question_types[q_type] = question_types.get(q_type, 0) + 1
        content_types[c_type] = content_types.get(c_type, 0) + 1
    
    print("\n  Question Types:")
    for q_type, count in sorted(question_types.items()):
        print(f"    {q_type}: {count}")
    
    print("\n  Content Types:")
    for c_type, count in sorted(content_types.items()):
        print(f"    {c_type}: {count}")

def main():
    parser = argparse.ArgumentParser(description='Download SLAKE dataset')
    parser.add_argument('--output_dir', type=str, default='data/raw',
                        help='Directory to save downloaded data')
    args = parser.parse_args()
    
    download_slake_dataset(args.output_dir)

if __name__ == '__main__':
    main()