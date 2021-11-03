"""Actor network implementations"""
import functools
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import flax.linen as nn
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
tfb = tfp.bijectors
tfd = tfp.distributions

from jaxdl.utils.commons import PRNGKey, Module, TrainState
from jaxdl.nn.dnn.mlp import default_init, forward_mlp_fn
from jaxdl.nn.dnn.conv import ConvNet


def create_conv_normal_dist_policy_fn(
  hidden_dims : Sequence[int] = [256, 256],
  forward_fn: Callable = forward_mlp_fn) -> Callable:
  """Return a normal distribution actor policy

  Args:
    hidden_dims (Sequence[int], optional): Hidden dimension of network.
      Defaults to [256, 256].
    action_dim (int, optional): Action dimensions of environment.
      Defaults to 2.

  Returns:
    Module: Returns a NormalDistPolicy
  """
  def network_fn(action_dim: int):
    return NormalConvDistPolicy(hidden_dims=hidden_dims,
      action_dim=action_dim, forward_fn=forward_fn)

  return network_fn


class NormalConvDistPolicy(nn.Module):
  """Normal distribution actor policy."""
  hidden_dims: Sequence[int]
  action_dim: int
  log_std_scale: float = 1.0
  log_std_min: float = -10.0
  log_std_max: float = 2.0
  tanh_squash_distribution: bool = True
  forward_fn: Callable = forward_mlp_fn
  dropout_rate: Optional[float] = None

  @nn.compact
  def __call__(self,
    observations: jnp.ndarray,
    temperature: float = 1.0,
    training: bool = False) -> tfd.Distribution:
    """Calls the network

    Args:
      observations (jnp.ndarray): Observation from environment.
      temperature (float, optional): Temperature parameter. Defaults to 1.0.
      training (bool, optional): Mode. Defaults to False.

    Returns:
      tfd.Distribution: Tensorflow probability distribution
    """

    observations = ConvNet()(observations)

    # call networks
    out = self.forward_fn(
      hidden_dims=self.hidden_dims, dropout_rate=self.dropout_rate,
      activate_final=True)(observations, training=training)

    # means
    means = nn.Dense(self.action_dim, kernel_init=default_init())(out)

    # log standard deviations
    log_stds = nn.Dense(self.action_dim,
      kernel_init=default_init(self.log_std_scale))(out)

    # clip log standard deviations
    log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

    # create distrubutions
    dist = tfd.MultivariateNormalDiag(
      loc=means, scale_diag=jnp.exp(log_stds) * temperature)

    # return distribution
    if self.tanh_squash_distribution:
      # will produce actions in [-1, 1]
      return tfd.TransformedDistribution(distribution=dist, bijector=tfb.Tanh())
    return dist