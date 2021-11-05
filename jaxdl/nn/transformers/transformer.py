
from typing import Tuple, Callable
from flax.linen.attention import PRNGKey

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct

from jaxdl.nn.transformers.attention import SelfAttention

@struct.dataclass
class TransformerConfig:
  """GPT-1 like network"""
  num_layer: int = 8
  num_head: int = 8
  emb_dim: int = 512
  dropout_rate: float = 0.1
  block_size: int = 100
  vocab_size: int = 256
  dropout_rate: float = 0.1


class TransformerBlock(nn.Module):
  """Transformer block"""
  config: TransformerConfig
  attention_fn: Callable = SelfAttention

  @nn.compact
  def __call__(self, x: jnp.ndarray, rng: PRNGKey) -> jnp.ndarray:
    """Forward transformer block function
    Args:
      x (jnp.ndarray): Input of size (batch, len, token_dim)

    Returns:
      jnp.ndarray: Output of size (batch, len, token_dim)
    """
    # attention network
    x = nn.LayerNorm()(x)
    attn_x = self.attention_fn(num_heads=self.config.num_head)(x)
    x = x + attn_x

    # forward network
    x_ln = nn.LayerNorm()(x)
    x = nn.Dense(features=4 * self.config.emb_dim)(x_ln)
    x = nn.relu(x)
    x = nn.Dense(self.config.emb_dim)(x)
    x = nn.Dropout(self.config.dropout_rate, deterministic=False)(x, rng=rng)

    # combine
    x = x_ln + x
    return x


class Transformer(nn.Module):
  config: TransformerConfig

  @nn.compact
  def __call__(self, x: jnp.ndarray, rng: PRNGKey,
    deterministic: bool = False) -> Tuple[PRNGKey, jnp.ndarray]:

    # encoding
    seq_length = x.shape[1]
    token_embedding = nn.Embed(
      num_embeddings=self.config.vocab_size, features=self.config.emb_dim)(x)
    # learnable positional encoding
    pos_encoding = self.param('pos_encoding',
      init_fn=lambda _: jax.random.uniform(
        rng, (1, self.config.block_size, self.config.emb_dim)))
    position_embeddings = pos_encoding[:, :seq_length, :]

    # networks
    rng, key = jax.random.split(rng)
    x = nn.Dropout(self.config.dropout_rate)(
      token_embedding + position_embeddings,
      deterministic=deterministic, rng=key)
    for _ in range(0, self.config.num_layer):
      rng, key = jax.random.split(rng)
      x = TransformerBlock(config=self.config)(x, rng=key)
    x = nn.LayerNorm()(x)
    logits = nn.Dense(features=self.config.vocab_size, use_bias=False)(x)

    return rng, logits
