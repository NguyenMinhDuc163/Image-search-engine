import torch.nn as nn
import torch
from pydantic import BaseModel


class EmbeddingLayer(nn.Module):
    """
    prepare the image input for transformer encoder
    input: (b, c, h, w)
    output: (b, num_patch, hidden_dim)
    """
    def __init__(self, hidden_dim, num_patch, patch_size):
        super().__init__()
        # create a cls token, though it is NOT necessary
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.position_embedding = nn.Parameter(torch.randn(1, num_patch + 1, hidden_dim))

        self.dropout = nn.Dropout(p=0.1)
        self.patch_embedding = PatchEmbedding(hidden_dim, patch_size)

    def forward(self, x):
        b = x.shape[0]
        x = self.patch_embedding(x) # (b, num_patch, hidden_dim)
        batched_cls_token = self.cls_token.expand(b, 1, -1)
        x = torch.cat([x, batched_cls_token], dim=1) # (b, num_patch + 1, hidden_dim)
        x = x + self.position_embedding
        return x


class PatchEmbedding(nn.Module):
    """
    Convert an image into patches and then project them into a vector space
    input: (b, c, h, w)
    output: (b, num_patch, hidden_dim)
    """
    def __init__(self, hidden_dim, patch_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv_layer = nn.Conv2d(in_channels=3,
                                    out_channels=hidden_dim,
                                    kernel_size=patch_size,
                                    stride=patch_size)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.conv_layer(x).view(b, self.hidden_dim, -1) # (b, hidden_dim, h * w // patch_size)
        return x.permute(0, 2, 1)


class TransformerEncoderLayer(nn.Module):
    """
    simple implementation of transformer encoder, w/o masking mechanism
    input: embedded patches of shape (b, num_patch + 1, hidden_dim)
    out: (b, num_patch + 1, hidden_dim)
    """
    def __init__(self, num_heads, hidden_dim, intermediate_dim):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim,
                                         num_heads=num_heads,
                                         dropout=0.1,
                                         batch_first=True)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, intermediate_dim),
                                  nn.ReLU(),
                                  nn.Linear(intermediate_dim, hidden_dim),
                                  nn.Dropout(p=0.1))
    def forward(self, x):
        x1 = self.layer_norm1(x)
        x1, _ = self.mha(x1, x1, x1)
        # skip connection
        x1 += x
        x2 = self.layer_norm2(x1)
        x2 = self.mlp(x1)

        # another skip connection
        x2 += x1

        return x2

class TransformerEncoder(nn.Module):
    """
    just a stack of TransformerEncoderLayer
    """
    def __init__(self, num_layers, num_heads, hidden_dim, intermediate_dim):
        super().__init__()
        self.encoder = nn.ModuleList([TransformerEncoderLayer(num_heads, hidden_dim, intermediate_dim) \
                                     for i in range(num_layers)])

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        return x


class ViT(nn.Module):
    def __init__(self, num_patch, patch_size, num_layers, num_heads, hidden_dim, intermediate_dim):
        super().__init__()
        self.embedding_layer = EmbeddingLayer(hidden_dim, num_patch, patch_size)
        self.transformer_encoder = TransformerEncoder(num_layers, num_heads, hidden_dim, intermediate_dim)

    def forward(self, x):

        patches = self.embedding_layer(x)
        out = self.transformer_encoder(patches)

        return out # return the cls token representation


class ViTForImageClassification(nn.Module):
    def __init__(self, num_classes, num_patch, patch_size, num_layers, num_heads, hidden_dim, intermediate_dim):
        super().__init__()
        self.ViT_model = ViT(num_patch, patch_size, num_layers, num_heads, hidden_dim, intermediate_dim)
        self.mlp = nn.Sequential(nn.Dropout(p=0.1),
                                 nn.Linear(hidden_dim, num_classes)
                                )
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, labels=None):
        first_token_embedding = self.ViT_model(x)[:, 0] # (b_size, hidden_dim)
        logits = self.mlp(first_token_embedding)
        if labels is not None:
            loss = self.loss_func(logits, labels)
            return {'loss': loss, 'logits': logits, 'embedding': first_token_embedding}
        return {'logits': logits, 'embedding': first_token_embedding}


class PostData(BaseModel):
    image: str
