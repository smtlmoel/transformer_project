import numpy as np
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, d_model, num_heads=1) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)

    def _attention(self, q, k, v):        
        return nn.functional.softmax((q @ k.transpose(2,3) / np.sqrt(self.d_model)), dim=2) @ v
    
    def _linear_projection(self, x: torch.Tensor, B: int):
        q = self.q_linear(x).view(B, -1, self.num_heads, int(self.d_model/self.num_heads)).transpose(1,2)
        k = self.k_linear(x).view(B, -1, self.num_heads, int(self.d_model/self.num_heads)).transpose(1,2)
        v = self.v_linear(x).view(B, -1, self.num_heads, int(self.d_model/self.num_heads)).transpose(1,2)
        return q, k, v
    
    def _linear_output(self, attn: torch.Tensor, B: int):
        out = attn.transpose(1,2).contiguous().view(B, -1, self.d_model)
        return self.output_linear(out)
    
    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        q, k, v = self._linear_projection(x, B)
        attn = self._attention(q, k, v)
        return self._linear_output(attn, B)
    

class MaskAttention(nn.Module):
    def __init__(self, d_model, num_heads=1) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)

    def _mask_attention(self, q, k, v):
        B, H, N, C = q.shape
        mask = torch.full((N, N), -np.Infinity)
        mask = torch.triu(mask, diagonal=1)
        return nn.functional.softmax((q @ k.transpose(2,3) / np.sqrt(self.d_model)) + mask, dim=2) @ v
    
    def _linear_projection(self, x: torch.Tensor, B: int):
        q = self.q_linear(x).view(B, -1, self.num_heads, int(self.d_model/self.num_heads)).transpose(1,2)
        k = self.k_linear(x).view(B, -1, self.num_heads, int(self.d_model/self.num_heads)).transpose(1,2)
        v = self.v_linear(x).view(B, -1, self.num_heads, int(self.d_model/self.num_heads)).transpose(1,2)
        return q, k, v
    
    def _linear_output(self, attn: torch.Tensor, B: int):
        out = attn.transpose(1,2).contiguous().view(B, -1, self.d_model)
        return self.output_linear(out)
    
    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        q, k, v = self._linear_projection(x, B)
        attn = self._mask_attention(q, k, v)
        return self._linear_output(attn, B)
