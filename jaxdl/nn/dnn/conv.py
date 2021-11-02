from typing import Any, Callable, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp
from jaxdl.nn.dnn.mlp import forward_mlp_fn


def default_init(scale: Optional[float] = jnp.sqrt(2)):
  return nn.initializers.orthogonal(scale)

def forward_conv_fn(
  hidden_dims: Sequence[int], dropout_rate: Optional[float] = None,
  activations=nn.relu, activate_final=False):

  def fn(observations: jnp.ndarray, training: bool = False):
    return ConvNet(hidden_dims, activate_final=activate_final,
      dropout_rate=dropout_rate, activations=activations)(
      observations, training)

  return fn

class ConvNet(nn.Module):
  activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
  activate_final: int = False
  dropout_rate: Optional[float] = None
  forward_fn: Callable = forward_mlp_fn

  @nn.compact
  def __call__(
    self, inputs: jnp.ndarray, training: bool = False) -> jnp.ndarray:
    # convolutional layers
    x = nn.Conv(32, kernel_size=(8, 8), strides=4, padding='same')(inputs)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(64, kernel_size=(4, 4), strides=2, padding='same')(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(64, kernel_size=(3, 3), strides=1, padding='same')(x)
    x = nn.relu(x)
    x = x.reshape((x.shape[0], -1))

    return x