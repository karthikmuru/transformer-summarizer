import torch.nn as nn
import pytorch_lightning as pl

from .multi_head_attention import MultiHeadAttention
from .feed_forward_network import FeedForwardNetwork
from .positional_encoding import PositionalEncoding

class EncoderLayer(nn.Module):
  def __init__(self, n_heads, d_model, p_drop, d_ff):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, n_heads)
    self.dropout1 = nn.Dropout(p_drop)
    self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)

    self.ffn = FeedForwardNetwork(d_model, d_ff)
    self.dropout2 = nn.Dropout(p_drop)
    self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

  def forward(self, input, mask):
    # |input| : (batch_size, seq_len, d_model)

    attn, attn_weights = self.mha(input, input, input, mask)
    attn = self.dropout1(attn)
    attn = self.layernorm1(input + attn)
    # |attn| : (batch_size, seq_len, d_model)
    # |attn_weights| : (batch_size, n_heads, seq_len(=q_len), seq_len(=k_len))

    output = self.ffn(attn)
    output = self.dropout2(output)
    output = self.layernorm2(attn + output)
    # |output| : (batch_size, seq_len, d_model)

    return output, attn_weights



class TransformerEncoder(pl.LightningModule):
  def __init__(self, vocab_size, n_layers, n_heads, d_model, p_drop, d_ff):
    super(TransformerEncoder, self).__init__()

    self.pad_id = 0

    self.embedding = nn.Embedding(vocab_size, d_model)
    self.positional_encoding = PositionalEncoding(d_model)

    self.layers = nn.ModuleList([EncoderLayer(n_heads, d_model, p_drop, d_ff) for _ in range(n_layers)])
  
  def forward(self, input):
    # |input| : (batch_size, seq_len)
    
    output = self.embedding(input)
    output = self.positional_encoding(output)
    # |output| : (batch_size, seq_len, d_model)
    mask = self.get_attention_padding_mask(input, input, self.pad_id)

    attention_weights = []
    for layer in self.layers:
      output, attn_weights = layer(output, mask)
      # |output| : (batch_size, seq_len, d_model)
      # |attn_weights| : (batch_size, n_heads, seq_len, seq_len)
      attention_weights.append(attn_weights)

    return output, attention_weights

  def get_attention_padding_mask(self, q, k, pad_id):
    attn_pad_mask = k.eq(pad_id).unsqueeze(1).repeat(1, q.size(1), 1)
    # |attn_pad_mask| : (batch_size, q_len, k_len)

    return attn_pad_mask
