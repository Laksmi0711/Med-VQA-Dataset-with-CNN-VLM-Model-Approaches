import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.dataset import SLAKEDataset, get_transforms
from models.baseline import CNNLSTMModel

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        images = batch['image'].to(device)
        answer_idx = batch['answer_idx'].to(device)
        
        # Simple question tokenization (word indices)
        # For baseline, we'll use a simple approach
        questions = batch['question']
        
        # Create simple word-to-index mapping on the fly
        # In production, use a proper tokenizer
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
            # Pad
            indices += [0] * (max_len - len(indices))
            question_indices.append(indices)
        
        question_indices = torch.tensor(question_indices, dtype=torch.long).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(images, question_indices)
        loss = criterion(logits, answer_idx)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(logits, 1)
        total += answer_idx.size(0)
        correct += (predicted == answer_idx).sum().item()
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
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
            loss = criterion(logits, answer_idx)
            
            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            total += answer_idx.size(0)
            correct += (predicted == answer_idx).sum().item()
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description='Train CNN-LSTM baseline')
    parser.add_argument('--data_dir', type=str, default='data/processed/splits')
    parser.add_argument('--images_dir', type=str, default='data/raw/images')
    parser.add_argument('--output_dir', type=str, default='results/baseline')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
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
    
    print(f"Vocabulary size: {vocab_size}")
    
    # Create datasets
    train_dataset = SLAKEDataset(
        Path(args.data_dir) / 'train.json',
        args.images_dir,
        vocab_file,
        transform=get_transforms('train')
    )
    
    val_dataset = SLAKEDataset(
        Path(args.data_dir) / 'val.json',
        args.images_dir,
        vocab_file,
        transform=get_transforms('val')
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    model = CNNLSTMModel(vocab_size=vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       patience=5, factor=0.5)
    
    # TensorBoard
    writer = SummaryWriter(output_path / 'tensorboard')
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                             optimizer, device, epoch)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Log
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'vocab_size': vocab_size
            }, output_path / 'best_model.pth')
            print(f"✓ Saved best model (val_acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    # Save final model and history
    torch.save(model.state_dict(), output_path / 'final_model.pth')
    
    with open(output_path / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    writer.close()
    
    print(f"\n✓ Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Results saved to: {output_path}")

if __name__ == '__main__':
    main()