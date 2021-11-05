import numpy as np

class CharDataset:
  def __init__(self, data, block_size):
    chars = sorted(list(set(data)))
    data_size, vocab_size = len(data), len(chars)
    print(f'Data has {data_size} characters, {vocab_size} unique.')
    self.stoi = { ch:i for i, ch in enumerate(chars) }
    self.itos = { i:ch for i, ch in enumerate(chars) }
    self.block_size = block_size
    self.vocab_size = vocab_size
    self.data = data

  def __len__(self):
    return len(self.data) - self.block_size - 1

  def __getitem__(self, idx):
    # grab a chunk of (block_size + 1) characters from the data
    chunk = self.data[idx:idx + self.block_size + 1]
    # encode every character to an integer
    dix = [self.stoi[s] for s in chunk]
    x = np.asarray(dix[:-1], dtype=np.float32)
    y = np.asarray(dix[1:], dtype=np.float32)
    return x, y