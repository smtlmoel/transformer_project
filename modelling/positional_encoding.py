import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Positional Encoding module to add positional information to input sequences.

    Args:
        d_model (int): Dimensionality of the model.
        seq_len (int): Length of the input sequences.
    """
    def __init__(self, d_model, seq_len) -> None:
        super().__init__()
        pe = torch.zeros(seq_len, d_model)

        k = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.) / d_model))

        pe[:, 0::2] = torch.sin(k * div_term)
        pe[:, 1::2] = torch.cos(k * div_term)
        
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the PositionalEncoding module.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor with positional encoding added to the input.
        """
        return input + self.pe[:, : input.size(1)].requires_grad_(False)