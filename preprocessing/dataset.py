import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from pathlib import Path
import torchvision.transforms as transforms

class SLAKEDataset(Dataset):
    """SLAKE Dataset for baseline CNN-LSTM model"""
    
    def __init__(self, json_file, images_dir, vocab_file, transform=None, max_seq_len=20):
        """
        Args:
            json_file: Path to split JSON file (train/val/test)
            images_dir: Directory containing images
            vocab_file: Path to vocabulary JSON file
            transform: Image transformations
            max_seq_len: Maximum question length
        """
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)
        
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.max_seq_len = max_seq_len
        
        # Create reverse vocabulary for answers
        self.idx2answer = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        
        # Load image
        img_path = self.images_dir / entry['img_id']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Process question (simple tokenization)
        question = entry['question'].lower()
        tokens = question.split()[:self.max_seq_len]
        
        # Pad or truncate
        question_tokens = tokens + ['<PAD>'] * (self.max_seq_len - len(tokens))
        
        # Get answer index
        answer_idx = entry.get('answer_idx', self.vocab.get('<UNK>', 1))
        
        return {
            'image': image,
            'question': question,
            'question_tokens': question_tokens,
            'answer': entry['answer'],
            'answer_idx': torch.tensor(answer_idx, dtype=torch.long),
            'question_type': entry['question_type'],
            'category': entry['category']
        }

class SLAKEBLIP2Dataset(Dataset):
    """SLAKE Dataset for BLIP-2 model"""
    
    def __init__(self, json_file, images_dir, processor):
        """
        Args:
            json_file: Path to split JSON file
            images_dir: Directory containing images
            processor: BLIP2 processor for image and text
        """
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        self.images_dir = Path(images_dir)
        self.processor = processor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        
        # Load image
        img_path = self.images_dir / entry['img_id']
        image = Image.open(img_path).convert('RGB')
        
        # Process with BLIP2 processor
        question = entry['question']
        answer = entry['answer']
        
        # Format as VQA prompt
        prompt = f"Question: {question} Answer:"
        
        return {
            'image': image,
            'question': question,
            'answer': answer,
            'prompt': prompt,
            'question_type': entry['question_type'],
            'category': entry['category']
        }

def get_transforms(split='train', image_size=224):
    """Get image transformations"""
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

