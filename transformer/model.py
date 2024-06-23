import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class TransformerModel(nn.Module):
    def __init__(self, feature_dim, num_classes, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.feature_dim = feature_dim
        
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, feature_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, src):
        src = src + self.positional_encoding[:, :src.size(1), :]
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)
        output = self.fc(output)
        return output