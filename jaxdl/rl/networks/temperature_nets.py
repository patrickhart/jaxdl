"""Temperature network implementations"""
import jax.numpy as jnp
from flax import linen as nn

from jaxdl.utils.commons import PRNGKey, Module


def create_temperature_network_fn(temperature: float = 1.0) -> Module:
  """Returns a temperature network

  Args:
      temperature (float, optional): Initial temperature. Defaults to 1.0.

  Returns:
      Module: Temperature network
  """
  return Temperature(temperature=temperature)

class Temperature(nn.Module):
  """Temperature network."""
  temperature: float = 1.0
  @nn.compact
  def __call__(self) -> jnp.ndarray:
    """Returns the temperature (alpha)"""
    log_temperature = self.param('log_temp',
      init_fn=lambda _: jnp.full((), jnp.log(self.temperature)))
    return jnp.exp(log_temperature)