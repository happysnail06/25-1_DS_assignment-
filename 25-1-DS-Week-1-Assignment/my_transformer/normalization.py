import torch.nn as nn
from torch import Tensor

class LayerNormalization(nn.Module):
    def __init__(self, d_model: int) -> None:
        super(LayerNormalization, self).__init__()
        self.layerNorm = nn.LayerNorm(d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.layerNorm(x)