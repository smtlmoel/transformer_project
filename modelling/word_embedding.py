import torch

import torch.nn as nn

class WordEmbedding(nn.Module):
    def __init__(self, num_embeddings, d_model) -> None:
        super().__init__()

        self.embedding_layer = nn.Embedding(num_embeddings, d_model)

    def forward(self, x: torch.Tensor):
        return self.embedding_layer(x)