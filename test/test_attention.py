import numpy as np
import torch

from modelling.attention import Attention

def test_linear_layer():
    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)

    # Define linear layer
    attention_layer = Attention(mask_future=False)

    # Generate random query, key and value with shape (batch_size, sequence_length, embedding_size)
    query = torch.randn((16, 10, 8))
    key = torch.randn((16, 10, 8))
    value = torch.randn((16, 10, 8))
    mask = torch.ones((16, 10))

    # Compute the expected and actual outputs
    expected = torch.nn.functional.softmax((query @ key.transpose(1,2) / np.sqrt(query.shape[-1]) + torch.zeros((16, 10, 10))), dim=2) @ value
    actual = attention_layer(query, key, value, mask)

    assert torch.allclose(expected, actual)