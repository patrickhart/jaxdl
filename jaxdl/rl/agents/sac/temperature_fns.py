"""Temperature functions"""
from typing import Tuple, Any

import functools
import jax
import numpy as np

from jaxdl.utils.commons import InfoDict, TrainState


@functools.partial(jax.jit)
def update_temperature(temperature_net: TrainState, entropy: float,
   target_entropy: float) -> Tuple[TrainState, InfoDict]:
  """Updates the temperature (alpha) value

  Args:
    temperature_net (TrainState): Temperature network
    entropy (float): Externally passed entropy value
    target_entropy (float): Target entropy value

  Returns:
    Tuple[TrainState, InfoDict]: Updated network
  """

  # temperature loss
  def temperature_loss_fn(temperature_params):
    temperature = temperature_net.apply_fn(temperature_params)
    temperature_loss = temperature * (entropy - target_entropy).mean()
    return temperature_loss, {
      'temperature': temperature,
      'temperature_loss': temperature_loss
    }

  loss_info, grads = jax.value_and_grad(temperature_loss_fn, has_aux=True)(
    temperature_net.params)
  new_temperature_net = temperature_net.apply_gradients(grads=grads)
  return new_temperature_net, loss_info[1]