# Code by Karthik Murugesan (Github: @karthikmuru)

import argparse
import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from data import Data
from models import Transformer

def _setup_parser():
  parser = argparse.ArgumentParser(add_help=False)

  trainer_parser = pl.Trainer.add_argparse_args(parser)
  trainer_parser._action_groups[1].title = "Trainer Args"
  parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

  data_group = parser.add_argument_group("Data Args")
  Data.add_to_argparse(data_group)

  parser.add_argument("--help", "-h", action="help")

  return parser

def main():
  parser = _setup_parser()
  args = parser.parse_args()

  data = Data(args)
  vocab_size = data.get_vocab_size()
  model = Transformer(vocab_size, vocab_size)

  logger = TensorBoardLogger('training/logs')
  model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor="val_loss", 
    dirpath = 'training/logs',
    filename="{epoch:03d}-{val_loss:.3f}", 
    save_top_k = 3,
    mode="min"
  )

  print(data)
  
  args.weights_summary = "full"
  trainer = pl.Trainer.from_argparse_args(args, callbacks=[model_checkpoint_callback], logger=logger)
  trainer.fit(model, data)

if __name__ == "__main__":
  main()