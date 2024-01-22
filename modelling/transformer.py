from positional_encoding import PositionalEncoding
from word_embedding import WordEmbedding
from functional import BaseTransformerLayer, TransformerDecoderLayer

import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, num_encoder_layers, num_heads, d_model, dim_feedforward, dropout) -> None:
        super().__init__()
        self.encoder_blocks = nn.ModuleList(
            [BaseTransformerLayer(d_model, num_heads, dim_feedforward, dropout) for _ in range(num_encoder_layers)]
        )

    def forward(self, x, x_mask):
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, x_mask)
        return x
    

class Decoder(nn.Module):
    def __init__(self, num_decoder_layers, num_heads, d_model, dim_feedforward, dropout) -> None:
        super().__init__()
        self.decoder_blocks = nn.ModuleList(
            [TransformerDecoderLayer(d_model, num_heads, dim_feedforward, dropout) for _ in range(num_decoder_layers)]
        )

    def forward(self, src, trg, src_mask, trg_mask):
        for decoder_block in self.decoder_blocks:
            trg = decoder_block(trg, src, src_mask, trg_mask)
        return trg


class Transformer(nn.Module):
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
        src_emb = self.embedding(src)
        trg_emb = self.embedding(trg)
        src_pos = self.positional_encoding(src_emb)
        trg_pos = self.positional_encoding(trg_emb)

        enc = self.encoder(src_pos, src_mask)
        dec = self.decoder(trg_pos, enc, src_mask, trg_mask)
        return self.projection(dec)
