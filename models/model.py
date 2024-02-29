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


class Embedding(nn.Module):
    def __init__(self, input_size, embed_dim):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_size, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x):
        x = x.float()
        return self.embed(x)


class Model(nn.Module):
    def __init__(self, seq_len, num_classes, num_channels, embed_dim, heads, depth, token_strat='channel', dropout=0.0, 
                 num_tokens=192, ssl=False, mask_pct = 0.25, task='pretrain'):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.num_tokens = num_tokens
        self.token_strat = token_strat
        self.ssl = ssl
        self.mask_pct = mask_pct
        self.task = task

        #if we're doing ssl we need flattened token strategy
        assert not (self.ssl and self.token_strat != 'flattened')

        if token_strat == 'seq':
            token_size = (num_channels * seq_len) // num_tokens
            self.embedding = Embedding(token_size, embed_dim)
        elif token_strat == 'channel':
            # Initialize separate embedding layers for each channel
            self.embedding = nn.ModuleList([Embedding(seq_len, embed_dim) for _ in range(num_channels)])
        elif token_strat == 'flattened':
            # initialize a codebook of size 1001, and embed it to the desired dimension
            self.embedding = nn.Embedding(1001, embed_dim)

        self.batch_norm = nn.BatchNorm1d(embed_dim)

        self.encoder = Encoder(
            dim = embed_dim,
            depth = depth,
            heads = heads,
            layer_dropout = dropout,
            alibi_pos_bias = True, # turns on ALiBi positional embedding
            alibi_num_heads = 4    # only use ALiBi for 4 out of the n heads, 

        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, num_classes),
            nn.Softmax(dim=-1)
        )

        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.cls = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.recon_classifier = nn.Sequential(
            nn.Linear(embed_dim, 1)
            )
    
    def forward(self, x):
        # if ssl we do everything separately
        if self.ssl:
            # first we flatten the input
            x = rearrange(x, 'b c s -> b (c s)')

            #multiply by 1000 to get the discrete values, then embed
            x = (x * 1000).long()
            x = self.embedding(x)

            #mask the input first by randomly choosing tokens to mask (dim=1)
            mask = torch.rand(x.shape[0], x.shape[1]) < self.mask_pct

            #replace the masked tokens with the mask token
            x[mask] = self.mask_token

            # add cls token
            cls_token = repeat(self.cls, '() n d -> b n d', b=x.shape[0])
            x = torch.cat((cls_token, x), dim=1)

            #pass through the encoder
            x = self.encoder(x)

            if self.task == 'pretrain':
                #reconstruct the input
                x = self.recon_classifier(x)
                x = x.squeeze(-1)
                x = x[:,1:]
                return None, x
            
            else:
                #classify
                cls = x[:,0,:]
                y_hat = self.classifier(cls)
                return y_hat, None

        # Using einops to rearrange the tensor
        if self.token_strat == 'seq':
            x = rearrange(x, 'b c s -> b (c s)') # Flatten channel and sequence length dimensions
            x = rearrange(x, 'b (t token) -> b t token', token=(self.num_channels * self.seq_len) // self.num_tokens)
        
            # Embed tokens
            x = self.embedding(x)

        elif self.token_strat == 'channel':
            # Embed each channel separately
            x = torch.stack([self.embedding[i](x[:,i,:]) for i in range(self.num_channels)], dim=1)          

        elif self.token_strat == 'flattened':
            x = rearrange(x, 'b c s -> b (c s)')
            #multiply by 1000 to get the discrete values
            x = (x * 1000).long()
            x = self.embedding(x)

        cls_token = repeat(self.cls, '() n d -> b n d', b=x.shape[0])
        x = torch.cat((cls_token, x), dim=1)

        x = self.encoder(x)

        # Classifier
        cls = x[:,0,:]
        y_hat = self.classifier(cls)

        return y_hat, x