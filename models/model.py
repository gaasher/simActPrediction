import torch
import torch.nn as nn
import math
from einops import rearrange, repeat
from x_transformers import Encoder

'''
Future embedding strategies:
1) patchify each chanel into several tokens, concat tokens
2) make spectrogram, then pass to vit or CNN
'''


# start for initial embedding of model, start with simple linear layer, but obviously this can be explored further

class Embedding(nn.Module):
    def __init__(self, token_size, embed_dim):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(token_size, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x):
        x = x.float()
        return self.embed(x)

class Model(nn.Module):
    def __init__(self, seq_len, num_classes, num_channels, embed_dim, heads, depth, dropout=0.0, num_tokens=192):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.num_tokens = num_tokens

        token_size = (num_channels * seq_len) // num_tokens
        self.embedding = Embedding(token_size, embed_dim)

        self.encoder = Encoder(
            dim = embed_dim,
            depth = depth,
            heads = heads,
            layer_dropout = dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, num_classes),
            nn.Softmax(dim=-1)
        )

        self.cls = nn.Parameter(torch.randn(1, 1, embed_dim))
    
    def forward(self, x):
        # Using einops to rearrange the tensor
        x = rearrange(x, 'b c s -> b (c s)') # Flatten channel and sequence length dimensions
        x = rearrange(x, 'b (t token) -> b t token', token=(self.num_channels * self.seq_len) // self.num_tokens)
        
        # Embed tokens
        x = self.embedding(x)

        # Add cls token and pass data through the encoder
        cls_token = repeat(self.cls, '() n d -> b n d', b=x.shape[0])
        x = torch.cat((cls_token, x), dim=1)
        x = self.encoder(x)

        # Classifier
        cls = x[:,0,:]
        y_hat = self.classifier(cls)

        return y_hat, x