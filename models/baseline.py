import torch
import torch.nn as nn
import torchvision.models as models

class CNNLSTMModel(nn.Module):
    """CNN-LSTM Multimodal Fusion for Med-VQA"""
    
    def __init__(self, vocab_size, embedding_dim=768, hidden_dim=512, 
                 num_layers=2, dropout=0.5):
        super(CNNLSTMModel, self).__init__()
        
        # Visual encoder: ResNet-50
        resnet = models.resnet50(pretrained=True)
        # Remove final FC layer
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.visual_dim = 2048
        
        # Freeze early layers (optional)
        for param in list(self.visual_encoder.parameters())[:-20]:
            param.requires_grad = False
        
        # Text encoder: Embedding + BiLSTM
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.text_dim = hidden_dim * 2  # Bidirectional
        
        # Attention mechanism for text
        self.attention = nn.Linear(self.text_dim, 1)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.visual_dim + self.text_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, vocab_size)
        )
    
    def forward(self, images, question_indices):
        """
        Args:
            images: (batch_size, 3, H, W)
            question_indices: (batch_size, seq_len)
        Returns:
            logits: (batch_size, vocab_size)
        """
        # Extract visual features
        visual_features = self.visual_encoder(images)  # (B, 2048, 1, 1)
        visual_features = visual_features.view(visual_features.size(0), -1)  # (B, 2048)
        
        # Extract text features
        embedded = self.embedding(question_indices)  # (B, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embedded)  # (B, seq_len, hidden_dim*2)
        
        # Apply attention
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (B, seq_len, 1)
        text_features = torch.sum(attn_weights * lstm_out, dim=1)  # (B, hidden_dim*2)
        
        # Fusion
        combined = torch.cat([visual_features, text_features], dim=1)
        logits = self.fusion(combined)
        
        return logits