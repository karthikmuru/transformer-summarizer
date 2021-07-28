import torch
import torch.nn as nn
import pytorch_lightning as pl

from .transformer_encoder import TransformerEncoder
from .transformer_decoder import TransformerDecoder

class Transformer(pl.LightningModule):

  def __init__(self, 
               src_vocab_size,
               tgt_vocab_size, 
               n_layers = 2, 
               n_heads = 4, 
               d_model = 256, 
               p_drop = 0.1, 
               d_ff = 1024):
    
    super(Transformer, self).__init__()

    self.encoder = TransformerEncoder(src_vocab_size, n_layers, n_heads, d_model, p_drop, d_ff)
    self.decoder = TransformerDecoder(tgt_vocab_size, n_layers, n_heads, d_model, p_drop, d_ff)
    self.linear = nn.Linear(d_model, tgt_vocab_size)

    self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    self.tgt_vocab_size = tgt_vocab_size
    self.src_vocab_size = src_vocab_size

  def forward(self, encoder_inputs, decoder_inputs):
    # |encoder_inputs| : (batch_size, encoder_inputs_len)
    # |decoder_inputs| : (batch_size, decoder_inputs_len)

    encoder_outputs, encoder_attn = self.encoder(encoder_inputs)
    # |encoder_outputs| : (batch_size, encoder_inputs_len, d_model)
    # |encoder_attn| : [(batch_size, n_heads, encoder_inputs_len, encoder_inputs_len)] * n_layers

    decoder_outputs, decoder_attn, enc_dec_attn = self.decoder(decoder_inputs, encoder_inputs, encoder_outputs)
    # |decoder_outputs| : (batch_size, decoder_inputs_len, d_model)
    # |decoder_attn| : [(batch_size, n_heads, decoder_inputs_len, decoder_inputs_len)] * n_layers
    # |enc_dec_attn| : [(batch_size, n_heads, decoder_inputs_len, encoder_inputs_len)] * n_layers

    outputs = self.linear(decoder_outputs)
    # |outputs| : (batch_size, decoder_inputs_len, tgt_vocab_size)

    return outputs, encoder_attn, decoder_attn, enc_dec_attn
  
  def training_step(self, batch, batch_idx):

    outputs, _, _, _ = self.forward(batch['article'], batch['highlights'])
    loss = self.criterion(outputs.view(-1, self.tgt_vocab_size), batch['highlights'].view(-1))
    
    self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

    return {'loss': loss}
  
  def validation_step(self, batch, batch_idx):

    outputs, _, _, _ = self.forward(batch['article'], batch['highlights'])
    loss = self.criterion(outputs.view(-1, self.tgt_vocab_size), batch['highlights'].view(-1))

    self.log('val_loss', loss, prog_bar=True, logger=True, on_epoch=True)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(
        [
          {"params": self.encoder.parameters()},
          {"params": self.decoder.parameters()},
          {"params": self.linear.parameters()}
        ], lr=1e-3)
    return optimizer