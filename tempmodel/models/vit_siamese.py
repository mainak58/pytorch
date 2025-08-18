# models/vit_siamese.py
import torch.nn as nn
from .vit_encoder import ViTEncoder


class SiameseViT(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.encoder = ViTEncoder(emb_dim=emb_dim)
        self.fc = nn.Linear(emb_dim, emb_dim)
    
    def forward_once(self, x):
        return self.encoder(x)
    
    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2
