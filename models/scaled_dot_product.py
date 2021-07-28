import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

class ScaledDotProduct(pl.LightningModule):
  def __init__(self, d_head):
    super(ScaledDotProduct, self).__init__()

    self.d_head = d_head

  def forward(self, q, k, v, mask):
    # |q| : (batch_size, n_heads, q_len, d_model), |k| : (batch_size, n_heads, k_len, d_model), |v| : (batch_size, n_heads, v_len, d_model)
    # |mask| : (batch_size, n_heads, seq_len(=q_len), seq_len(=k_len))

    # Calculate attention
    attn_score = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_head)

    attn_score.masked_fill_(mask, 1e-9)
    # |attn_score| : (batch_size, n_heads, q_len, k_len)

    attn_weights = nn.Softmax(dim=-1)(attn_score)
    # |attn_weights| : (batch_size, n_heads, q_len, k_len)

    output = torch.matmul(attn_weights, v)
    # |output| : (batch_size, n_heads, q_len, d_head)

    return output, attn_weights