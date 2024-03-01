from modelling.attention import MultiHeadAttention
from collections import OrderedDict
from torch import Tensor
import torch.nn as nn

class BaseTransformerLayer(nn.Module):
    """
    Base transformer layer module.

    Args:
        input_dim (int): Dimensionality of the input.
        num_heads (int): Number of attention heads.
        feature_dim (int): Dimensionality of the feature transformation.
        dropout (float): Dropout probability.
    """
    def __init__(self, input_dim, num_heads=2, feature_dim=6, dropout=0.0) -> None:
        super().__init__()

        self.self_attention = MultiHeadAttention(input_dim, num_heads, False)

        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)

        self.layer_dropout = nn.Dropout(dropout)

        self.feature_transformation = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(input_dim, feature_dim)),
            ('relu', nn.ReLU()),
            ('linear2', nn.Linear(feature_dim, input_dim))
        ]))

    def forward(self, src: Tensor, src_mask: Tensor):
        """
        Forward pass of the BaseTransformerLayer module.

        Args:
            src (torch.Tensor): Input tensor.
            src_mask (torch.Tensor): Mask tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        src1 = self.self_attention(src, src, src, src_mask)
        src2 = self.layer_norm_1(src + self.layer_dropout(src1))
        src3 = self.feature_transformation(src2)
        return self.layer_norm_2(src2 + self.layer_dropout(src3))
    
class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer module.

    Args:
        input_dim (int): Dimensionality of the input.
        num_heads (int): Number of attention heads.
        feature_dim (int): Dimensionality of the feature transformation.
        dropout (float): Dropout probability.
    """
    def __init__(self, input_dim, num_heads=2, feature_dim=6, dropout=0.0) -> None:
        super().__init__()

        self.self_attention = MultiHeadAttention(input_dim, num_heads, True)
        self.encoder_attention = MultiHeadAttention(input_dim, num_heads, False)

        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)
        self.layer_norm_3 = nn.LayerNorm(input_dim)

        self.layer_dropout = nn.Dropout(dropout)

        self.feature_transformation = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(input_dim, feature_dim)),
            ('relu', nn.ReLU()),
            ('linear2', nn.Linear(feature_dim, input_dim))
        ]))

    def forward(self, input: Tensor, encoder: Tensor, encoder_mask:Tensor, input_mask: Tensor):
        """
        Forward pass of the TransformerDecoderLayer module.

        Args:
            input (torch.Tensor): Input tensor.
            encoder (torch.Tensor): Encoder tensor.
            encoder_mask (torch.Tensor): Encoder mask tensor.
            input_mask (torch.Tensor): Mask tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        input1 = self.self_attention(input, input, input, input_mask)
        trg = self.layer_norm_1(input + self.layer_dropout(input1))
        trg1 = self.encoder_attention(trg, encoder, encoder, encoder_mask)
        trg2 = self.layer_norm_2(trg + self.layer_dropout(trg1))
        trg3 = self.feature_transformation(trg2)
        return self.layer_norm_3(trg2 + self.layer_dropout(trg3))