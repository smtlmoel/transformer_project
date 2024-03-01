import torch.nn as nn

from modelling.positional_encoding import PositionalEncoding
from modelling.word_embedding import WordEmbedding
from modelling.functional import BaseTransformerLayer, TransformerDecoderLayer

class Encoder(nn.Module):
    """
    Encoder module of the transformer architecture.

    Args:
        num_encoder_layers (int): Number of encoder layers.
        num_heads (int): Number of attention heads.
        d_model (int): Dimensionality of the model.
        dim_feedforward (int): Dimensionality of the feedforward layers.
        dropout (float): Dropout probability.
    """
    def __init__(self, num_encoder_layers, num_heads, d_model, dim_feedforward, dropout) -> None:
        super().__init__()
        self.encoder_blocks = nn.ModuleList(
            [BaseTransformerLayer(d_model, num_heads, dim_feedforward, dropout) for _ in range(num_encoder_layers)]
        )

    def forward(self, input, input_mask):
        """
        Forward pass of the Encoder module.

        Args:
            input (torch.Tensor): Input tensor.
            input_mask (torch.Tensor): Mask tensor for input.

        Returns:
            torch.Tensor: Encoded tensor.
        """
        for encoder_block in self.encoder_blocks:
            input = encoder_block(input, input_mask)
        return input
    

class Decoder(nn.Module):
    """
    Decoder module of the transformer architecture.

    Args:
        num_decoder_layers (int): Number of decoder layers.
        num_heads (int): Number of attention heads.
        d_model (int): Dimensionality of the model.
        dim_feedforward (int): Dimensionality of the feedforward layers.
        dropout (float): Dropout probability.
    """
    def __init__(self, num_decoder_layers, num_heads, d_model, dim_feedforward, dropout) -> None:
        super().__init__()
        self.decoder_blocks = nn.ModuleList(
            [TransformerDecoderLayer(d_model, num_heads, dim_feedforward, dropout) for _ in range(num_decoder_layers)]
        )

    def forward(self, src, src_mask, trg, trg_mask):
        """
        Forward pass of the Decoder module.

        Args:
            src (torch.Tensor): Source tensor.
            trg (torch.Tensor): Target tensor.
            src_mask (torch.Tensor): Mask tensor for source.
            trg_mask (torch.Tensor): Mask tensor for target.

        Returns:
            torch.Tensor: Decoded tensor.
        """
        for decoder_block in self.decoder_blocks:
            trg = decoder_block(trg, src, src_mask, trg_mask)
        return trg


class Transformer(nn.Module):
    """
    Transformer model.

    Args:
        vocab_size (int): Size of the vocabulary.
        d_model (int): Dimensionality of the model.
        num_heads (int): Number of attention heads.
        num_encoder_layers (int): Number of encoder layers.
        num_decoder_layers (int): Number of decoder layers.
        dim_feedforward (int): Dimensionality of the feedforward layers.
        dropout (float): Dropout probability.
        max_len (int): Maximum sequence length.
    """
    def __init__(self, vocab_size, d_model, num_heads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, max_len):
        super().__init__()

        self.embedding = WordEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, seq_len=max_len)

        self.encoder = Encoder(num_encoder_layers, num_heads, d_model, dim_feedforward, dropout)
        self.decoder = Decoder(num_decoder_layers, num_heads, d_model, dim_feedforward, dropout)

        self.projection = nn.Linear(d_model, vocab_size)

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, src, src_mask, trg, trg_mask):
        """
        Forward pass of the Transformer model.

        Args:
            src (torch.Tensor): Source tensor.
            src_mask (torch.Tensor): Mask tensor for source.
            trg (torch.Tensor): Target tensor.
            trg_mask (torch.Tensor): Mask tensor for target.

        Returns:
            torch.Tensor: Output tensor.
        """
        src_emb = self.embedding(src)
        trg_emb = self.embedding(trg)
        src_pos = self.positional_encoding(src_emb)
        trg_pos = self.positional_encoding(trg_emb)

        enc = self.encoder(src_pos, src_mask)
        dec = self.decoder(trg_pos, enc, src_mask, trg_mask)
        return self.projection(dec)
    
    def encode(self, src, src_mask):
        """
        Encoder forward pass.

        Args:
            src (torch.Tensor): Input tensor.
            src_mask (torch.Tensor): Mask tensor for input.

        Returns:
            torch.Tensor: Encoded tensor.
        """
        src1 = self.embedding(src)
        src2 = self.positional_encoding(src1)
        src3 = self.encoder(src2, src_mask)
        return src3
    
    def decode(self, src, src_mask, trg, trg_mask):
        """
        Decoder forward pass.

        Args:
            src (torch.Tensor): Input tensor.
            src_mask (torch.Tensor): Mask tensor for input.
            trg (torch.Tensor): Target tensor.
            trg_mask (torch.Tensor): Mask tensor for target.

        Returns:
            torch.Tensor: Decoded tensor.
        """
        trg1 = self.embedding(trg)
        trg2 = self.positional_encoding(trg1)
        trg3 = self.decoder(src, src_mask, trg2, trg_mask)
        return trg3
