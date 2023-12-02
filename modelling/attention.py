import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, mask_future=False) -> None:
        super().__init__()
        self.mask_future = mask_future

    def _attention(self, q, k, v, mask):        
        return nn.functional.softmax((q @ k.transpose(1,2) / np.sqrt(q.shape[-1]) + mask), dim=-1) @ v
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor=None):
        B_Q, N_Q, C_Q = query.shape
        B_K, N_K, C_K = key.shape

        future_mask = torch.zeros((N_Q, N_K))
        
        if self.mask_future:
            future_mask = torch.full((N_Q, N_K), -np.Infinity)
            future_mask = torch.triu(future_mask, diagonal=1).unsqueeze(0)

        combined_mask = future_mask.masked_fill(~mask.unsqueeze(1).to(torch.bool), -np.Infinity)

        # attn = self._attention(query, key, value)
        return self._attention(query, key, value, combined_mask)
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads=1, mask_future=False) -> None:
        super().__init__()
        self.mask_future = mask_future
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_key = int(self.d_model/self.num_heads)

        self.query_transform = nn.Linear(d_model, d_model, bias=False)
        self.key_transform = nn.Linear(d_model, d_model, bias=False)
        self.value_transform = nn.Linear(d_model, d_model, bias=False)
        self.output_transform = nn.Linear(d_model, d_model, bias=False)

    def _mask_attention(self, Q: Tensor, K: Tensor, V: Tensor, mask: Tensor):
        H = Q.size(1)
        N_Q = Q.size(2)
        N_K = K.size(2)

        future_mask = torch.zeros((H, N_Q, N_K))
        
        if self.mask_future:
            future_mask = torch.full((N_Q, N_K), -np.Infinity)
            future_mask = torch.triu(future_mask, diagonal=1).unsqueeze(0).unsqueeze(1)
            
        combined_mask = future_mask.masked_fill(~mask.unsqueeze(2).unsqueeze(1).to(torch.bool), -np.Infinity)

        scaled_dot = (Q @ K.permute(0, 1, 3, 2) / np.sqrt(self.d_model))
        scaled_dot.masked_fill(mask.unsqueeze(2).unsqueeze(1)==0, -np.inf)

        # nn.functional.softmax(((Q @ K.permute(0, 1, 3, 2) / np.sqrt(self.d_model)) + combined_mask), dim=-1) @ V

        return nn.functional.softmax(scaled_dot, dim=-1) @ V
    
    def _linear_projection(self, query: Tensor, key: Tensor, value: Tensor, B: int):
        Q = self.query_transform(query).view(B, -1, self.num_heads, self.d_key).permute(0, 2, 1, 3)
        K = self.key_transform(key).view(B, -1, self.num_heads, self.d_key).permute(0, 2, 1, 3)
        V = self.value_transform(value).view(B, -1, self.num_heads, self.d_key).permute(0, 2, 1, 3)
        return Q, K, V
    
    def _linear_output(self, attn: Tensor, B: int):
        out = attn.permute(0, 2, 1, 3).contiguous().view(B, -1, self.d_model)
        return self.output_transform(out)
    
    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor=None):
        B = query.size(0)
        Q, K, V = self._linear_projection(query, key, value, B)
        attn = self._mask_attention(Q, K, V, mask)
        return self._linear_output(attn, B)
