import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import sys

from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model

sys.path.append(str(Path(__file__).parent.parent))
from preprocessing.dataset import SLAKEBLIP2Dataset

def collate_fn(batch, processor):
    """Custom collate function for BLIP2"""
    images = [item['image'] for item in batch]
    prompts = [item['prompt'] for item in batch]
    answers = [item['answer'] for item in batch]
    
    # Process images and text
    encoding = processor(images=images, text=prompts, 
                         return_tensors="pt", padding=True, truncation=True)
    
    # Encode answers
    answer_encoding = processor.tokenizer(answers, return_tensors="pt",
                                          padding=True, truncation=True)
    
    encoding['labels'] = answer_encoding.input_ids
    
    # Store metadata
    encoding['question_type'] = [item['question_type'] for item in batch]
    encoding['category'] = [item['category'] for item in batch]
    encoding['answers'] = answers
    
    return encoding

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, 
                gradient_accumulation_steps=4):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    steps = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    optimizer.zero_grad()
    
    for i, batch in enumerate(pbar):
        # Move to device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                  for k, v in batch.items() if k not in ['question_type', 'category', 'answers']}
        
        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights
        if (i + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            steps += 1
        
        total_loss += loss.item() * gradient_accumulation_steps
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def validate(model, dataloader, processor, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                      for k, v in batch.items() if k not in ['question_type', 'category', 'answers']}
            
            # Calculate loss
            outputs = model(**inputs)
            total_loss += outputs.loss.item()
            
            # Generate predictions
            generated_ids = model.generate(
                pixel_values=inputs['pixel_values'],
                input_ids=inputs['input_ids'],
                max_length=50,
                num_beams=3
            )
            
            generated_texts = processor.batch_decode(generated_ids, 
                                                     skip_special_tokens=True)
            
            # Calculate accuracy
            for pred, true_answer in zip(generated_texts, batch['answers']):
                pred = pred.strip().lower()
                true_answer = true_answer.strip().lower()
                
                predictions.append({
                    'prediction': pred,
                    'ground_truth': true_answer,
                    'question_type': batch['question_type'][len(predictions) % len(batch['answers'])],
                    'category': batch['category'][len(predictions) % len(batch['answers'])]
                })
                
                if pred == true_answer or true_answer in pred:
                    correct += 1
                total += 1
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy, predictions

def main():
    parser = argparse.ArgumentParser(description='Train BLIP-2 with LoRA')
    parser.add_argument('--data_dir', type=str, default='data/processed/splits')
    parser.add_argument('--images_dir', type=str, default='data/raw/images')
    parser.add_argument('--output_dir', type=str, default='results/blip2')
    parser.add_argument('--model_name', type=str, default='Salesforce/blip2-opt-2.7b')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--use_lora', action='store_true', default=True)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--mixed_precision', action='store_true', default=True)
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load processor and model
    print("Loading BLIP-2 model and processor...")
    processor = Blip2Processor.from_pretrained(args.model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.mixed_precision else torch.float32
    )
    
    # Apply LoRA
    if args.use_lora:
        print("Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    model = model.to(device)
    
    # Create datasets
    train_dataset = SLAKEBLIP2Dataset(
        Path(args.data_dir) / 'train.json',
        args.images_dir,
        processor
    )
    
    val_dataset = SLAKEBLIP2Dataset(
        Path(args.data_dir) / 'val.json',
        args.images_dir,
        processor
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=4,
        collate_fn=lambda batch: collate_fn(batch, processor)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=4,
        collate_fn=lambda batch: collate_fn(batch, processor)
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    total_steps = len(train_loader) // args.gradient_accumulation_steps * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # TensorBoard
    writer = SummaryWriter(output_path / 'tensorboard')
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, 
                                 device, epoch, args.gradient_accumulation_steps)
        
        # Validate
        val_loss, val_acc, predictions = validate(model, val_loader, processor, device)
        
        # Log
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            # Save model
            save_dir = output_path / 'best_model'
            save_dir.mkdir(exist_ok=True)
            
            if args.use_lora:
                model.save_pretrained(save_dir)
            else:
                model.save_pretrained(save_dir)
            processor.save_pretrained(save_dir)
            
            # Save predictions
            with open(save_dir / 'val_predictions.json', 'w') as f:
                json.dump(predictions, f, indent=2)
            
            print(f"✓ Saved best model (val_acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    # Save history
    with open(output_path / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    writer.close()
    
    print(f"\n✓ Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Results saved to: {output_path}")

if __name__ == '__main__':
    main()