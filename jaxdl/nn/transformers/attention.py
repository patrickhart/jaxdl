
from typing import Optional, Callable

import functools
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.linear import DenseGeneral
from flax.linen.linear import default_kernel_init
from jax.nn.initializers import zeros


def dot_product_attention(
  query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray,
  mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:

  assert query.ndim == key.ndim, 'q, k must have same rank.'
  assert query.shape[:-3] == key.shape[:-3], (
    'q, k batch dims must match.')
  assert query.shape[-2] == key.shape[-2], (
    'q, k num_heads must match.')
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  # calculate attention matrix
  depth = query.shape[-1]
  query = query / jnp.sqrt(depth).astype(jnp.float32)

  # qhd = query_dim, num_heads, dim
  # khd = key_dim, num_heads, dim
  # -> = heads, query_dim, key_dim
  # for q:
  #   for h:
  #     for k:
  #       sum = 0
  #       for d:
  #         sum += query[q, h, d]*key[k, h, d]
  #       output[h, q, k] = sum
  attn_weights = jnp.einsum('...qhd,...khd->...hqk', query, key)
  # attn_weights shape is (batch..., num_heads, query_dim, key_dim)

  if mask is not None:
    big_neg = jnp.finfo(jnp.float32).min
    attn_weights = jnp.where(mask, attn_weights, big_neg)

  # normalize the attention weights
  attn_weights = jax.nn.softmax(attn_weights).astype(jnp.float32)

  # hqk = head_dim, query_dim, key_dim
  # khd = key_dim, head_dim, value_dim
  # -> = q_dim, head_dim, value_dim
  # for h:
  #   for q:
  #     for d:
  #       sum = 0
  #       for k:
  #         sum += attn_weights[h,q,k]*value[k, h, d]
  #       output[q,h,d] = sum
  return jnp.einsum('...hqk,...khd->...qhd', attn_weights, value)


class MultiHeadDotProductAttention(nn.Module):
  """Multi-head dot-product attention."""
  num_heads: int
  kernel_init: Callable = default_kernel_init
  bias_init: Callable = zeros
  use_bias: bool = True
  attention_fn: Callable = dot_product_attention

  @nn.compact
  def __call__(self, queries: jnp.ndarray, key_values: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Applies multi-head dot product attention on the input data."""

    # dimensions
    features = queries.shape[-1]
    qkv_features = key_values.shape[-1]
    assert qkv_features % self.num_heads == 0, (
      f'Memory dimension {qkv_features} must be divisible'
      f'by number of heads {self.num_heads}.')
    head_dim = qkv_features // self.num_heads

    dense = functools.partial(DenseGeneral, axis=-1,
      features=(self.num_heads, head_dim), kernel_init=self.kernel_init,
      bias_init=self.bias_init, use_bias=self.use_bias)

    # project queries to multi-headed q/k/v
    # dimensions are then [batch..., length, n_heads, n_features_per_head]
    query, key, value = (dense(name='query')(queries),
      dense(name='key')(key_values), dense(name='value')(key_values))

    # apply attention
    x = self.attention_fn(query, key, value, mask=mask)

    # back to the original inputs dimensions
    out = DenseGeneral(features=features, axis=(-2, -1),
      kernel_init=self.kernel_init, bias_init=self.bias_init,
      use_bias=self.use_bias, name='out')(x)
    return out


class SelfAttention(MultiHeadDotProductAttention):
  """Self-attention special case of multi-head dot-product attention."""
  # def __init__(self, num_heads=1):
  #   super().__init__(num_heads=num_heads)

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
    return super().__call__(inputs, inputs, mask)


def make_attention_mask(query_input: jnp.ndarray, key_input: jnp.ndarray,
  pairwise_fn: Callable= jnp.multiply) -> jnp.ndarray:
  mask = pairwise_fn(jnp.expand_dims(query_input, axis=-1),
    jnp.expand_dims(key_input, axis=-2))
  mask = jnp.expand_dims(mask, axis=-3)
  return mask

def make_causal_mask(x: jnp.ndarray) -> jnp.ndarray:
  idxs = jnp.broadcast_to(jnp.arange(x.shape[-1], dtype=jnp.int32), x.shape)
  return make_attention_mask(idxs, idxs, jnp.greater_equal)