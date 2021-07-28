import torch
import torch.nn as nn
import pytorch_lightning as pl


class FeedForwardNetwork(pl.LightningModule):
  def __init__(self, d_model, d_ff):
    super(FeedForwardNetwork, self).__init__()

    self.linear1 = nn.Linear(d_model, d_ff)
    self.linear2 = nn.Linear(d_ff, d_model)

    self.relu = nn.ReLU()

  def forward(self, x):
    # |x| : (batch_size, seq_len, d_model)

    output = self.linear1(x)
    # |output| : (batch_size, seq_len, d_ff)

    output = self.linear2(output)
    # |output| : (batch_size, seq_len, d_model)

    return output