"""Critic network implementations"""
from typing import Callable, Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn

from jaxdl.utils.commons import Module
from jaxdl.nn.dnn.mlp import MLP


def create_double_critic_network_fn(
  hidden_dims : Sequence[int] = [256, 256]) -> Module:
  """Returns a double critic network

  Args:
    hidden_dims (Sequence[int], optional): Hidden layers dimensions.
      Defaults to [256, 256].

  Returns:
    Module: Double critic network
  """
  return DoubleCriticNetwork(hidden_dims=hidden_dims)

class CriticNetwork(nn.Module):
  """Critic network implementation"""
  hidden_dims: Sequence[int]
  activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

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
    critic = MLP((*self.hidden_dims, 1), activations=self.activations)(inputs)
    return jnp.squeeze(critic, -1)

class DoubleCriticNetwork(nn.Module):
  """Returns two state-values"""
  hidden_dims: Sequence[int]
  activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

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
      self.hidden_dims, activations=self.activations)(observations, actions)
    q2 = CriticNetwork(
      self.hidden_dims, activations=self.activations)(observations, actions)
    return q1, q2