# models/vit_encoder.py
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, emb_dim=128):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches + 1, emb_dim))

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)  # [B, C, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, C]
        x = x + self.pos_embedding
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim=128, n_heads=4, depth=4, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.emb_dim = emb_dim

    def forward(self, x):
        return self.transformer(x)

class ViTEncoder(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.patch_embed = PatchEmbedding(224, 16, emb_dim)
        self.encoder = TransformerEncoder(emb_dim=emb_dim)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        return self.mlp_head(x[:, 0])  # use [CLS] token only
