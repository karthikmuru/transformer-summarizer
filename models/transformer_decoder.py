import torch
import torch.nn as nn
import pytorch_lightning as pl

from .multi_head_attention import MultiHeadAttention
from .feed_forward_network import FeedForwardNetwork
from .positional_encoding import PositionalEncoding

class DecoderLayer(pl.LightningModule):

  def __init__(self, n_heads, d_model, p_drop, d_ff):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, n_heads)
    self.dropout1 = nn.Dropout(p_drop)
    self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)

    self.mha2 = MultiHeadAttention(d_model, n_heads)
    self.dropout2 = nn.Dropout(p_drop)
    self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

    self.ffn = FeedForwardNetwork(d_model, d_ff)
    self.dropout3 = nn.Dropout(p_drop)
    self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)

  def forward(self, input, enc_output, attn_mask, enc_dec_attn_mask):
    # |input| : (batch_size, seq_len, d_model)
    # |enc_output| : (batch_size, enc_output_len, d_model)
    # |attn_mask| : (batch_size, seq_len, seq_len)
    # |enc_output| : (batch_size, enc_output_len, d_model)
    # |enc_dec_attn_mask| : (batch_size, seq_len, enc_output_len)
    
    mha1_output, attn_weights = self.mha1(input, input, input, attn_mask)
    mha1_output = self.dropout1(mha1_output)
    mha1_output = self.layernorm1(mha1_output)
    # |mha1_output| : (batch_size, seq_len, d_model)
    # |attn_weights| : (batch_size, n_heads, q_len(=seq_len), k_len(=seq_len))

    mha2_output, end_dec_attn_weights = self.mha2(mha1_output, enc_output, enc_output, enc_dec_attn_mask)
    mha2_output = self.dropout2(mha2_output)
    mha2_output = self.layernorm2(mha2_output)
    # |mha2_output| : (batch_size, seq_len, d_model)
    # |end_dec_attn_weights| : (batch_size, n_heads, q_len(=seq_len), k_len(=enc_output_len))

    ffn_output = self.ffn(mha2_output)
    ffn_output = self.dropout3(ffn_output)
    ffn_output = self.layernorm3(ffn_output)
    # |ffn_output| : (batch_size, seq_len, d_model)

    return ffn_output, attn_weights, end_dec_attn_weights


class TransformerDecoder(pl.LightningModule):

  def __init__(self, vocab_size, n_layers, n_heads, d_model, p_drop, d_ff):
    super(TransformerDecoder, self).__init__()

    self.pad_id=0

    self.embedding = nn.Embedding(vocab_size, d_model)
    self.positional_encoding = PositionalEncoding(d_model)

    self.layers = nn.ModuleList([DecoderLayer(n_heads, d_model, p_drop, d_ff) for _ in range(n_layers)])

  def forward(self, inputs, enc_inputs, enc_outputs):
    # |input| : (batch_size, seq_len)
    # |enc_input| : (batch_size, enc_input_len)
    # |enc_output| : (batch_size, seq_len, d_model)

    outputs = self.embedding(inputs)
    outputs = self.positional_encoding(outputs)
    # |output| : (batch_size, seq_len, d_model)

    # Masking the padding
    attn_pad_mask = self.get_attention_padding_mask(inputs, inputs, self.pad_id)
    # |attn_pad_mask| : (batch_size, seq_len, seq_len)
    # Masking future lookup 
    attn_subsequent_mask = self.get_attention_subsequent_mask(inputs).to(device=attn_pad_mask.device)
    # |attn_subsequent_mask| : (batch_size, seq_len, seq_len)
    attn_mask = torch.gt((attn_pad_mask.to(dtype=attn_subsequent_mask.dtype) + attn_subsequent_mask), 0)
    # |attn_mask| : (batch_size, seq_len, seq_len)
    enc_dec_attn_mask = self.get_attention_padding_mask(inputs, enc_inputs, self.pad_id)
    # |enc_dec_attn_mask| : (batch_size, seq_len, encoder_inputs_len)
    
    attention_weights = []
    end_dec_attention_weights = []
    for layer in self.layers:
      output, attn_weights, end_dec_attn_weights = layer(outputs, enc_outputs, attn_mask, enc_dec_attn_mask)
      # |outputs| : (batch_size, seq_len, d_model)
      # |attn_weights| : (batch_size, n_heads, seq_len, seq_len)
      # |end_dec_attn_weights| : (batch_size, n_heads, seq_len, enc_outputs_len)
      attention_weights.append(attn_weights)
      end_dec_attention_weights.append(end_dec_attn_weights)

    return outputs, attention_weights, end_dec_attention_weights

  def get_attention_padding_mask(self, q, k, pad_id):
    attn_pad_mask = k.eq(pad_id).unsqueeze(1).repeat(1, q.size(1), 1)
    # |attn_pad_mask| : (batch_size, q_len, k_len)

    return attn_pad_mask
    
  def get_attention_subsequent_mask(self, q):
    bs, q_len = q.size()
    subsequent_mask = torch.ones(bs, q_len, q_len).triu(diagonal=1)
    # |subsequent_mask| : (batch_size, q_len, q_len)
    
    return subsequent_mask