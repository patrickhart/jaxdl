from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


from jaxdl.rl.utils.replay_buffer import Batch
from jaxdl.utils.commons import InfoDict


TimeStep = Tuple[np.ndarray, float, bool, dict]


class RLAgent:
  """Base class definiton of an agent in JAXDL"""
  def __init__(self):
    pass

  def sample(self, observations: np.ndarray,
    temperature: float = 1.0) -> jnp.ndarray:
    pass

  def update(self, batch: Batch) -> InfoDict:
    return {}