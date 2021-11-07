from typing import Tuple

import jax
import numpy as np

from jaxdl.utils.commons import PRNGKey

class CharDataset:
  def __init__(self, data: str, block_size: int = 100):
    chars = sorted(list(set(data)))
    data_size, vocab_size = len(data), len(chars)
    print(f'Data has {data_size} characters, {vocab_size} unique.')
    self.stoi = { ch:i for i, ch in enumerate(chars) }
    self.itos = { i:ch for i, ch in enumerate(chars) }
    self.block_size = block_size
    self.vocab_size = vocab_size
    self.data = data

  def __len__(self) -> int:
    return len(self.data) - self.block_size - 1

  def __getitem__(self, idx: int):
    # grab a chunk of (block_size + 1) characters from the data
    chunk = self.data[idx:idx + self.block_size + 1]
    # encode every character to an integer
    dix = [self.stoi[s] for s in chunk]
    x = np.asarray(dix[:-1], dtype=np.float32)
    y = np.asarray(dix[1:], dtype=np.float32)
    return x, y

  def get_sampled_batch(self, rng: PRNGKey, batch_size: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
    source_batch = []
    target_batch = []
    for _ in range(0, batch_size):
      rng, key = jax.random.split(rng)
      idx = jax.random.randint(
        key, shape=(1,), minval=0, maxval=len(self))
      source, target = self[idx[0]]
      source_batch.append(source)
      target_batch.append(target)
    return np.array(source_batch, dtype=np.float32), \
      np.array(target_batch, dtype=np.float32)