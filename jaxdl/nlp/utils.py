from typing import Sequence, Tuple
import jax
import numpy as np

from jaxdl.utils.commons import PRNGKey


def char_to_embedding(c: str):
  return ord(c)

def sequence_to_embeddings(squence: str) -> np.ndarray:
  text = []
  for c in squence:
    text.append(char_to_embedding(c))
  return np.array(text, dtype=np.int32)

def embeddings_to_sequence(embeddings: np.ndarray):
  str = ""
  for idx in embeddings:
    str += chr(idx)
  return str

def sample_data(key: PRNGKey, data: np.array, seq_len: int,
  batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
  """Generate inputs and labels"""
  start_ids = jax.random.randint(
    key, shape=(batch_size,), minval=0, maxval=len(data)-seq_len-1)
  inputs = []
  targets = []
  for start_id in start_ids:
    inp = [ord(c) for c in data[start_id:start_id+seq_len]]
    target = [ord(c) for c in data[start_id+1:start_id+seq_len+1]]
    inputs.append(inp)
    targets.append(target)
  return np.array(inputs, dtype=np.int32), np.array(targets, dtype=np.int32)
