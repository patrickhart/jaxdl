"""GPT-like implementation"""

from typing import Callable
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxdl.nlp.transformer_fns import update_transformer
from jaxdl.nn.transformers import transformer

from jaxdl.utils.commons import InfoDict, save_train_state, restore_train_state
from jaxdl.utils.commons import create_train_state
from jaxdl.rl.utils.replay_buffer import Batch
from jaxdl.nn.transformers.transformer import TransformerConfig, create_transformer_fn


class GPTLike:
  """GPT-like implementation

    Original PyTorch implementation: https://github.com/karpathy/minGPT

    Usage:
      agent = GPTLike(0)
      agent.restore('./tmp/')
      agent.save('./tmp/')
  """
  def __init__(self,
    seed: int,
    transformer_config: TransformerConfig,
    transformer_net_fn: Callable = create_transformer_fn,
    transformer_lr: float = 3e-4):
    # split rng and generate keys
    rng = jax.random.PRNGKey(seed)
    rng, transformer_key = jax.random.split(rng)

    # transformer network
    inp = jax.random.uniform(
      rng, shape=(1, transformer_config.block_size), dtype=np.float32)
    transformer_net = create_train_state(
      transformer_net_fn(transformer_config), [rng, inp, transformer_key],
      optax.adam(learning_rate=transformer_lr))

    # networks
    self.transformer_net = transformer_net

    # parameters
    self.rng = rng
    self.step_num = 1

  def restore(self, path):
    """Loads the network."""
    self.actor_net = restore_train_state(
      self.actor_net, path, prefix="transformer")

  def save(self, path):
    """Saves the network."""
    save_train_state(self.actor_net, path, prefix="transformer")

  def sample(self, observations: np.ndarray,
    temperature: float = 1.0, evaluate: bool = False) -> np.ndarray:
    """Returns sampled characters."""
    pass

  def update(self, source: np.ndarray, target: np.ndarray) -> InfoDict:
    """Updates the transformer network."""
    self.step_num += 1

    # update actor
    self.rng, key = jax.random.split(self.rng)
    self.rng, self.actor_net, actor_info = update_transformer(
      self.transformer_net, source, target, key)

    # increase step count
    return actor_info