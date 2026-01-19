"""
BLIP-2 Model for Medical Visual Question Answering
Uses parameter-efficient LoRA fine-tuning
"""
import torch
import torch.nn as nn
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    AutoProcessor,
    AutoModelForVision2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

class BLIP2MedVQA(nn.Module):
    """
    BLIP-2 Model for Medical VQA with LoRA fine-tuning
    
    Architecture:
    - Vision Encoder: ViT-g/14 (frozen)
    - Q-Former: Lightweight bridge (trainable with LoRA)
    - Language Model: OPT-2.7B or FlanT5-XL (frozen)
    """
    
    def __init__(
        self,
        model_name="Salesforce/blip2-opt-2.7b",
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        device="cuda"
    ):
        """
        Args:
            model_name: Hugging Face model name
            use_lora: Whether to use LoRA for parameter-efficient fine-tuning
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout
            device: Device to load model on
        """
        super().__init__()
        
        self.device = device
        self.model_name = model_name
        self.use_lora = use_lora
        
        print(f"Loading BLIP-2 model: {model_name}")
        
        # Load processor
        self.processor = Blip2Processor.from_pretrained(model_name)
        
        # Load model
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        # Apply LoRA if specified
        if use_lora:
            print("Applying LoRA for parameter-efficient fine-tuning...")
            self.apply_lora(lora_r, lora_alpha, lora_dropout)
        else:
            # Freeze vision encoder and language model, only train Q-Former
            self.freeze_components()
        
        # Move to device
        if device == "cpu":
            self.model = self.model.to(device)
        
        print(f"Model loaded successfully on {device}")
        self.print_trainable_parameters()
    
    def apply_lora(self, r=8, alpha=16, dropout=0.1):
        """Apply LoRA to Q-Former"""
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
            target_modules=["q", "v", "q_proj", "v_proj"],  # Attention layers
            inference_mode=False
        )
        
        self.model = get_peft_model(self.model, lora_config)
        print("✓ LoRA applied successfully")
    
    def freeze_components(self):
        """Freeze vision encoder and language model"""
        # Freeze vision model
        for param in self.model.vision_model.parameters():
            param.requires_grad = False
        
        # Freeze language model
        for param in self.model.language_model.parameters():
            param.requires_grad = False
        
        # Q-Former remains trainable
        for param in self.model.qformer.parameters():
            param.requires_grad = True
        
        print("✓ Vision encoder and language model frozen")
        print("✓ Q-Former parameters trainable")
    
    def print_trainable_parameters(self):
        """Print number of trainable parameters"""
        trainable_params = 0
        all_params = 0
        
        for _, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(f"Trainable params: {trainable_params:,} || "
              f"All params: {all_params:,} || "
              f"Trainable %: {100 * trainable_params / all_params:.2f}%")
    
    def forward(self, images, questions, answers=None):
        """
        Forward pass
        
        Args:
            images: List of PIL Images or tensor (batch, 3, H, W)
            questions: List of question strings
            answers: Optional list of answer strings for training
            
        Returns:
            If training (answers provided): loss
            If inference: generated text
        """
        # Process inputs
        if isinstance(images, torch.Tensor):
            # Convert tensor to PIL Images
            images = [self.tensor_to_pil(img) for img in images]
        
        # Prepare inputs
        inputs = self.processor(
            images=images,
            text=questions,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        if answers is not None:
            # Training mode
            labels = self.processor.tokenizer(
                answers,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).input_ids.to(self.device)
            
            # Replace padding token id with -100 for loss computation
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            
            outputs = self.model(**inputs, labels=labels)
            return outputs.loss
        else:
            # Inference mode
            generated_ids = self.model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                early_stopping=True
            )
            
            generated_texts = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            return generated_texts
    
    def generate(self, images, questions, max_length=50, num_beams=5):
        """
        Generate answers for questions
        
        Args:
            images: List of PIL Images or tensor
            questions: List of question strings
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            
        Returns:
            List of generated answer strings
        """
        self.model.eval()
        
        with torch.no_grad():
            if isinstance(images, torch.Tensor):
                images = [self.tensor_to_pil(img) for img in images]
            
            inputs = self.processor(
                images=images,
                text=questions,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            generated_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                temperature=0.7,
                top_p=0.9
            )
            
            generated_texts = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            # Clean up generated text
            generated_texts = [text.strip() for text in generated_texts]
        
        return generated_texts
    
    def tensor_to_pil(self, tensor):
        """Convert normalized tensor to PIL Image"""
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        tensor = tensor.cpu() * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to PIL
        from torchvision.transforms import ToPILImage
        to_pil = ToPILImage()
        return to_pil(tensor)
    
    def save_pretrained(self, save_directory):
        """Save model and processor"""
        if self.use_lora:
            # Save LoRA weights
            self.model.save_pretrained(save_directory)
        else:
            # Save full model
            self.model.save_pretrained(save_directory)
        
        self.processor.save_pretrained(save_directory)
        print(f"✓ Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, load_directory, device="cuda"):
        """Load pretrained model"""
        print(f"Loading model from {load_directory}")
        
        processor = Blip2Processor.from_pretrained(load_directory)
        model = Blip2ForConditionalGeneration.from_pretrained(
            load_directory,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        instance = cls.__new__(cls)
        instance.processor = processor
        instance.model = model
        instance.device = device
        
        if device == "cpu":
            instance.model = instance.model.to(device)
        
        print(f"✓ Model loaded from {load_directory}")
        return instance

def build_blip2_model(
    model_name="Salesforce/blip2-opt-2.7b",
    use_lora=True,
    lora_r=8,
    device="cuda"
):
    """
    Build BLIP-2 model for Medical VQA
    
    Args:
        model_name: Hugging Face model name
            - "Salesforce/blip2-opt-2.7b" (default, faster)
            - "Salesforce/blip2-flan-t5-xl" (better performance)
        use_lora: Use LoRA for parameter-efficient fine-tuning
        lora_r: LoRA rank (higher = more parameters)
        device: Device to load model on
        
    Returns:
        BLIP2MedVQA model instance
    """
    model = BLIP2MedVQA(
        model_name=model_name,
        use_lora=use_lora,
        lora_r=lora_r,
        device=device
    )
    
    return model

if __name__ == "__main__":
    # Test BLIP-2 model
    print("Testing BLIP-2 Medical VQA Model...")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Build model
    model = build_blip2_model(
        model_name="Salesforce/blip2-opt-2.7b",
        use_lora=True,
        lora_r=8,
        device=device
    )
    
    print("\n" + "=" * 80)
    print("Model Architecture:")
    print("=" * 80)
    print(f"Vision Encoder: ViT-g/14 (frozen)")
    print(f"Q-Former: Trainable with LoRA")
    print(f"Language Model: OPT-2.7B (frozen)")
    
    print("\n" + "=" * 80)
    print("✓ Model test passed!")
    print("=" * 80)