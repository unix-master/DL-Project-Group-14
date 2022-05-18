
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Self-Attention layer for calculating attention scores
'''

class SelfAttention(nn.Module):
  def __init__(self, embed_size, heads):
    super(SelfAttention, self).__init__()
    self.embed_size = embed_size
    self.heads = heads
    self.head_dim = embed_size // heads

    assert (self.head_dim * heads == embed_size), 'embed_size {} should be a multiple of heads {}'.format(self.embed_size, self.heads)

    self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
    self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
    self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

    self.feed_forward = nn.Linear(self.heads * self.head_dim, self.embed_size)

  def forward(self, values, keys, queries):
    N = queries.shape[0]
    value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

    # splitting into multiple heads
    values = values.reshape(N, value_len, self.heads, self.head_dim)
    keys = keys.reshape(N, key_len, self.heads, self.head_dim)
    queries = queries.reshape(N, query_len, self.heads, self.head_dim)

    values = self.values(values)
    keys = self.keys(keys)
    queries = self.queries(queries)

    match_score = torch.einsum('nqhd,nkhd->nhqk', [queries, keys])

    attention_score = torch.softmax(match_score / (self.embed_size ** 0.5), dim=3)

    out = torch.einsum('nhql,nlhd->nqhd', [attention_score, values]).reshape(N, query_len, self.heads*self.head_dim)
    out = self.feed_forward(out)

    return out

'''
A single Transformer block (standard architecture used in Attention paper)
'''

class TransformerBlock(nn.Module):
  def __init__(self, embed_size, heads, dropout, forward_expansion):
    super(TransformerBlock, self).__init__()
    self.mhattention = SelfAttention(embed_size, heads)
    self.norm1 = nn.LayerNorm(embed_size)
    self.norm2 = nn.LayerNorm(embed_size)

    self.dropout = nn.Dropout(dropout)
    
    self.feed_forward = nn.Sequential(
        nn.Linear(embed_size, forward_expansion * embed_size),
        nn.ReLU(),
        nn.Linear(forward_expansion * embed_size, embed_size)
    )
  
  def forward(self, value, key, query):
    attention = self.mhattention(value, key, query)
    x = self.dropout(self.norm1(attention + query))
    forward = self.feed_forward(x)
    out = self.dropout(self.norm2(forward + x))
    return out

'''
Encoder block in the Transformer
'''

class TransformerEncoder(nn.Module):
  def __init__(self, num_layers, embed_size, heads, forward_expansion, dropout, device):
    super(TransformerEncoder, self).__init__()
    self.embed_size = embed_size
    self.device = device
    self.layers = nn.ModuleList(
        [
         TransformerBlock(embed_size, heads, dropout, forward_expansion)
         for _ in range(num_layers)
        ]
    )
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    # N, num_features = x.shape
    out = self.dropout(x)

    for layer in self.layers:
      out = layer(out, out, out)
    
    # print('shape encoder output: {}'.format(out.shape))

    return out

'''
Transformer Encoder + MLP classifier
We have used the self-attention pooling outputs for classifying each window in the time-series data.
''' 

class TransformerClassifier(nn.Module):
  def __init__(self, num_labels, embed_size, hidden_size, device):
    super(TransformerClassifier, self).__init__()
    self.encoder = TransformerEncoder(
        num_layers = 2,
        embed_size = embed_size,
        heads = 2,
        forward_expansion = 2,
        dropout = 0,
        device = device
    )
    self.flatten = nn.Flatten()
    self.classifier = nn.Sequential(
        nn.LazyLinear(hidden_size),
        nn.ReLU(True),
        nn.Linear(hidden_size, num_labels),
    )

  def forward(self, x):
    x = self.encoder(x)
    # take average across 150 timesteps in each window (one example in a batch)
    # x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
    x = self.flatten(x)
    out = self.classifier(x)
    # print('shape classifier output: {}'.format(out.shape))
    return out

'''
Deep Convnet used for classifying each window in the time-series data.
Our convnet architecture consists of three blocks followed by classifier layer which is also a convolution layer.
'''

class ConvNet(nn.Module):
    def __init__(self, window_size=150, num_features=8, num_classes=4):
        super(ConvNet, self).__init__()
        
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1,8), stride=1, padding=0, bias=True),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(num_features, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=25),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        )
        
        self.cnn2 = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(1,8), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=50),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),
        )
        
        self.cnn3 = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(1,8), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=50),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        )
        
#         self.classifier = nn.Sequential(
#             nn.Linear(in_features=1825, out_features=500, bias=True),
#             nn.ReLU(),
#             nn.Linear(in_features=500, out_features=num_classes, bias=True)
#         )

        self.classifier = nn.Conv2d(in_channels=50, out_channels=num_classes, kernel_size=(1,12), bias=True)
        
    def forward(self, x):
        x = self.cnn1(x.unsqueeze(dim=1).transpose(2, 3))
        x = self.cnn2(x)
        x = self.cnn3(x)
#         print(x.shape)
#         x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        x = self.classifier(x)
        x = x.squeeze(dim=2).squeeze(dim=2)
        return F.log_softmax(x, dim=1)