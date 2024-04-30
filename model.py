import torch
import torch.nn as nn
from torch.nn import functional as F

class AttentionHead(nn.Module):

    def __init__(self, embed_dim, head_dim, context_length, dropout_rate = 0):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.dropout_rate = dropout_rate
        self.context_length = context_length
        
        self.query = nn.Linear(embed_dim, head_dim)
        self.key = nn.Linear(embed_dim, head_dim)
        self.value = nn.Linear(embed_dim, head_dim)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attention_pattern = q @ k.transpose(-2, -1) * self.embed_dim**0.5
        attention_pattern = attention_pattern.masked_fill(self.tril[:self.context_length,
                                                                    :self.context_length] == 0,
                                                                      float('-inf'))
        attention_pattern = F.softmax(attention_pattern, dim=-1)
        # Optimization
        attention_pattern = self.dropout(attention_pattern)

        return attention_pattern @ v
    


        
