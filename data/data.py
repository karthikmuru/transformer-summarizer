import pytorch_lightning as pl
import argparse
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from .base_data_module import BaseDataModule
from datasets import load_from_disk

class Data(BaseDataModule):

  def __init__(self, args: argparse.Namespace) -> None:
    super().__init__()
    self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    self.vocab_size = self.tokenizer.vocab_size

  def setup(self, stage = None):
    self.dataset = load_from_disk(self.data_dir)
    self.dataset.set_format(type='torch', columns=['article', 'highlights'])
    self.data_train = self.dataset['train']
    self.data_val = self.dataset['test']