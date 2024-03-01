import torch
import torch.nn as nn
import math

class WordEmbedding(nn.Module):
    """
    Word Embedding module to map tokens to their corresponding embedding vectors.

    Args:
        num_embeddings (int): Number of unique tokens in the vocabulary.
        d_model (int): Dimensionality of the embedding vectors.
    """
    def __init__(self, num_embeddings, d_model) -> None:
        super().__init__()

        self.d_model = d_model

        self.embedding_layer = nn.Embedding(num_embeddings, self.d_model)

    def forward(self, token_id_tensor: torch.Tensor):
        """
        Forward pass of the WordEmbedding module.

        Args:
            token_id_tensor (torch.Tensor): Input tensor containing token indices.

        Returns:
            torch.Tensor: Embedding vectors for the input tokens.
        """
        return self.embedding_layer(token_id_tensor) * math.sqrt(self.d_model)