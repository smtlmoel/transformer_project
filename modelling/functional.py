from modelling.attention import MultiHeadAttention
from collections import OrderedDict
from torch import Tensor
import torch.nn as nn

class BaseTransformerLayer(nn.Module):
    def __init__(self, input_dim, num_heads=2, feature_dim=6, dropout=0.0) -> None:
        super().__init__()

        self.self_attention = MultiHeadAttention(input_dim, num_heads, False)

        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)

        self.layer_dropout_1 = nn.Dropout(dropout)
        self.layer_dropout_2 = nn.Dropout(dropout)

        self.feature_transformation = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(input_dim, feature_dim)),
            ('relu', nn.ReLU()),
            ('linear2', nn.Linear(feature_dim, input_dim))
        ]))

    def forward(self, x: Tensor, mask: Tensor):
        _x = self.self_attention(x, x, x, mask)
        x = self.layer_norm_1(x + self.layer_dropout_1(_x))
        _x = self.feature_transformation(x)
        return self.layer_norm_2(x + self.layer_dropout_2(_x))