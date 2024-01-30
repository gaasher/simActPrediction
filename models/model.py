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
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x):
        return self.embed(x)
    
class Model(nn.Module):
    def __init__(self, input_dim, num_classes, num_channels, embed_dim, heads, depth, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.depth = depth
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.input_dim = input_dim

        # for each channel we will have a separate embedding
        self.embedding = nn.ModuleList([
            # in dim is sequence length of raw channel data, we will embed each channel separately
            Embedding(self.input_dim, self.embed_dim) for i in range(num_channels)
        ])

        # encoder, we have a sequence length of num_channels
        self.encoder = Encoder(
            dim = embed_dim,
            depth = depth,
            heads = heads,
            layer_dropout = dropout
        )

        #simple classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, num_classes),
            nn.Softmax(dim=-1)
        )

        self.cls = nn.Parameter(torch.randn(1, 1, embed_dim))
    
    def forward(self, x):
        # x shape is (batch_size, num_channels, seq_len)

        # embed data
        x = torch.stack([self.embedding[i](x[:,i,:]) for i in range(self.num_channels)], dim=1)

        # add cls token
        cls_token = repeat(self.cls, '() n d -> b n d', b = x.shape[0])
        x = torch.cat((cls_token, x), dim=1)

        # pass data through encoder, for now each channel is treated as a separate token
        x = self.encoder(x)
        
        # pass cls token through decoder
        cls = x[:,0,:]
        y_hat = self.classifier(cls)

        return y_hat, x