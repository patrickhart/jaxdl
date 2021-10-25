import unittest
import jax
import optax
import numpy as np
from jax import random

from jaxdl.nn.dnn.mlp import MLP
from jaxdl.nn.transformer.transformer import Transformer, TransformerConfig, \
  TransformerBlock
from jaxdl.utils.commons import create_train_state
from jaxdl.nn.transformer.transformer_fns import sample_embedding, \
  update_transformer
from jaxdl.nlp.utils import sample_data, transform_data

class TestTransformer(unittest.TestCase):
  def test_transformer_block(self):
    key = jax.random.PRNGKey(0)
    # transformer block
    config = TransformerConfig()
    transformer_module = TransformerBlock(config)
    inp = random.uniform(key,
      (config.batch_size, config.seq_len, config.emb_dim))
    transformer = create_train_state(
      transformer_module, [key, inp], optax.adam(learning_rate=0.0003))
    out = transformer.apply_fn(transformer.params, inp)
    # batch_size: int = 10
    # seq_len: int = 50
    # emb_dim: int = 256
    self.assertEqual(
      out.shape, (config.batch_size, config.seq_len, config.emb_dim))

  def test_transformer(self):
    key = jax.random.PRNGKey(0)
    # transformer network
    config = TransformerConfig()
    transformer_module = Transformer(config)
    embedding = random.uniform(key,
      (config.batch_size, config.seq_len, config.emb_dim))
    transformer = create_train_state(
      transformer_module, [key, embedding], optax.adam(learning_rate=0.0003))
    # batch_size: int = 10
    # seq_len: int = 50
    # emb_dim: int = 256
    _, out = sample_embedding(
      key, transformer_net=transformer, embedding=embedding)
    self.assertEqual(
      out.shape, (config.batch_size, config.seq_len, config.emb_dim))

    text = "Hello there. How are you?! You are truly marvellous today! Have a good day."
    key = random.PRNGKey(0)
    emb_data = transform_data(text)
    source, target = sample_data(
      key, emb_data, seq_len=50, batch_size=10, embed_dim=256)
    rng, new_transformer, info = update_transformer(key, transformer, source, target)


if __name__ == '__main__':
  unittest.main()