import torch
import torch.nn as nn
import math
from torch import Tensor

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x)

class PositionEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super(PositionEmbedding, self).__init__()
        #todo
        encoding = torch.zeros(max_len, d_model)
        encoding.requires_grad = False
    
        position = torch.arange(0, max_len).float().unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        
        self.encoding = encoding.unsqueeze(0)
    
    def forward(self, x: Tensor) -> Tensor:
        #todo one line!
        return self.encoding[:, :x.size(1), :]