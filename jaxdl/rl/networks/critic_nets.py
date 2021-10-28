"""Critic network implementations"""
from typing import Callable, Optional, Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn

from jaxdl.utils.commons import Module
from jaxdl.nn.dnn.mlp import forward_mlp_fn


def create_double_critic_network_fn(
  hidden_dims : Sequence[int] = [256, 256],
  forward_fn: Callable = forward_mlp_fn) -> Callable:
  """Returns a double critic network

  Args:
    hidden_dims (Sequence[int], optional): Hidden layers dimensions.
      Defaults to [256, 256].

  Returns:
    Module: Double critic network
  """
  def network_fn():
    return DoubleCriticNetwork(
      hidden_dims=hidden_dims, forward_fn=forward_fn)
  return network_fn


class CriticNetwork(nn.Module):
  """Critic network implementation"""
  hidden_dims: Sequence[int]
  activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
  forward_fn: Callable = forward_mlp_fn

  @nn.compact
  def __call__(self,
    observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
    """Returns scalar state-value

    Args:
      observations (jnp.ndarray): Environment observation
      actions (jnp.ndarray): Action for environment

    Returns:
      jnp.ndarray: Value for given observations
    """
    inputs = jnp.concatenate([observations, actions], -1)

    # call networks
    out = self.forward_fn(
      hidden_dims=(*self.hidden_dims, 1), activate_final=False,
      activations=self.activations)(inputs)

    return jnp.squeeze(out, -1)


class DoubleCriticNetwork(nn.Module):
  """Returns two state-values"""
  hidden_dims: Sequence[int]
  activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
  forward_fn: Callable = forward_mlp_fn

  @nn.compact
  def __call__(self, observations: jnp.ndarray,
    actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns scalar state-values

    Args:
      observations (jnp.ndarray): Environment observation
      actions (jnp.ndarray): Action for environment

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Q1 and Q2 state-values
    """
    q1 = CriticNetwork(
      self.hidden_dims, activations=self.activations,
      forward_fn=self.forward_fn)(observations, actions)
    q2 = CriticNetwork(
      self.hidden_dims, activations=self.activations,
      forward_fn=self.forward_fn)(observations, actions)
    return q1, q2