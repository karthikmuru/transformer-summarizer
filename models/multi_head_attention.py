import torch
import torch.nn as nn
import pytorch_lightning as pl
from .scaled_dot_product import ScaledDotProduct

class MultiHeadAttention(pl.LightningModule):
  def __init__(self, d_model, n_heads):
    super(MultiHeadAttention, self).__init__()

    self.d_head = d_model // n_heads
    self.n_heads  = n_heads

    self.WQ = nn.Linear(d_model, d_model)
    self.WK = nn.Linear(d_model, d_model)
    self.WV = nn.Linear(d_model, d_model)

    self.scaled_dot_product = ScaledDotProduct(self.d_head)

    self.linear = nn.Linear(self.d_head * self.n_heads, d_model)

  def forward(self, Q, K, V, mask):
    # |Q| : (batch_size, q_len, d_model) : |K| -> (batch_size, k_len, d_model), |V| : (batch_size, v_len, d_model)
    # |mask| : (batch_size, seq_len(=q_len), seq_len(=k_len))

    self.batch_size = Q.shape[0]

    # Split attention heads
    # Input shape : (batch_size, q_len, d_model)
    q_heads = self.WQ(Q).view(self.batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
    k_heads = self.WK(K).view(self.batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
    v_heads = self.WV(V).view(self.batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
    # |q_heads| : (batch_size, n_heads, q_len, d_model), |k_heads| : (batch_size, n_heads, k_len, d_model), |v_heads| : (batch_size, n_heads, v_len, d_model)

    # Scalar dot product
    attn_mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
    # |attn_mask| : (batch_size, n_heads, seq_len(=q_len), seq_len(=k_len))
    
    attn, attn_weights = self.scaled_dot_product(q_heads, k_heads, v_heads, attn_mask)
    # |attn| : (batch_size, n_heads, q_len, v_len)
    # |attn_weights| : (batch_size, n_heads, q_len, k_len)

    # Combining all the heads in attention output
    # Input shape : (batch_size, n_heads, q_len, v_len)
    # Output shape : (batch_size, seq_len, d_model)
    attn = attn.transpose(1, 2).contiguous().view(self.batch_size, -1, self.n_heads * self.d_head)
    # |attn| : (batch_size, seq_len, d_head * n_heads)

    output = self.linear(attn)
    # |output| : (batch_size, seq_len, d_model)

    return output, attn_weights 