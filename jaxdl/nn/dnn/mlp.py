from typing import Any, Callable, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp

def default_init(scale: Optional[float] = jnp.sqrt(2)):
  return nn.initializers.orthogonal(scale)

def forward_mlp_fn(
  hidden_dims: Sequence[int], dropout_rate: Optional[float] = None,
  activations=nn.relu, activate_final=False):

  def fn(observations: jnp.ndarray, training: bool = False):
    return MLP(hidden_dims, activate_final=activate_final,
      dropout_rate=dropout_rate, activations=activations)(
      observations, training)

  return fn


class MLP(nn.Module):
  hidden_dims: Sequence[int]
  activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
  activate_final: int = False
  dropout_rate: Optional[float] = None

  @nn.compact
  def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
    for i, size in enumerate(self.hidden_dims):
      x = nn.Dense(size, kernel_init=default_init())(x)
      if i + 1 < len(self.hidden_dims) or self.activate_final:
        x = self.activations(x)
        if self.dropout_rate is not None:
          x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
    return x