import unittest
import jax
import numpy as np
from jax import random
from jaxdl.nn.transformers.attention import dot_product_attention, \
  MultiHeadDotProductAttention, SelfAttention, make_causal_mask
from jaxdl.nn.transformers.transformer import Transformer, TransformerConfig


class TestTransformers(unittest.TestCase):
  def test_attention_fn(self):
    rng = random.PRNGKey(0)
    rng, key = jax.random.split(rng)

    # [batch, seq. len, heads, feature_dim]
    input = jax.random.uniform(key, shape=(5, 10, 7, 20))
    result = dot_product_attention(input, input, input)
    self.assertEqual(input.shape, result.shape)

  def test_multi_head_attentin(self):
    attention_net = MultiHeadDotProductAttention(5)
    rng = random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    # [batch, sequence length, feature_dim]
    input = jax.random.uniform(key, shape=(5, 10, 20))
    # call network
    attention_net_params = attention_net.init(rng, input, input)
    result = attention_net.apply(attention_net_params, input, input)
    self.assertEqual(input.shape, result.shape)

  def test_self_attention(self):
    self_attention_net = SelfAttention(5)
    rng = random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    # [batch, sequence length, feature_dim]
    input = jax.random.uniform(key, shape=(5, 10, 20))
    # call network
    attention_net_params = self_attention_net.init(rng, input)
    result = self_attention_net.apply(attention_net_params, input)
    self.assertEqual(input.shape, result.shape)

  def test_attention_mask(self):
    rng = random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    # [batch, sequence length, feature_dim]
    input = jax.random.uniform(key, shape=(5, 5))
    mask = make_causal_mask(input)
    # batch, head, dim, dim
    self.assertEqual(mask.shape, (5, 1, 5, 5))

  def test_self_attention_with_mask(self):
    rng = random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    self_attention_net = SelfAttention(5)

    # [batch, sequence length, feature_dim]
    input = jax.random.uniform(key, shape=(5, 10, 20))
    mask_inputs = jax.random.uniform(key, shape=(5, 10))
    mask = make_causal_mask(mask_inputs)

    # call network
    attention_net_params = self_attention_net.init(rng, input)
    result = self_attention_net.apply(attention_net_params, input, mask=mask)
    self.assertEqual(input.shape, result.shape)

  def test_transformer(self):
    transformer_config = TransformerConfig()
    transformer_net = Transformer(transformer_config)
    rng = random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    # [batch, sequence length]
    input = jax.random.randint(key, minval=0, maxval=1, shape=(5, 10))
    transformer_net_params = transformer_net.init(rng, input, rng)
    result = transformer_net.apply(transformer_net_params, input, key)

if __name__ == '__main__':
  unittest.main()