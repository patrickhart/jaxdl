"""Transformer NLP functions"""
from typing import Tuple, Any

import functools
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

from flax.training import common_utils
from optax import softmax_cross_entropy, sigmoid_binary_cross_entropy
from jaxdl.utils.commons import InfoDict, PRNGKey, TrainState


def compute_cross_entropy(logits: jnp.ndarray, targets: jnp.ndarray):
  """Compute cross entropy"""
  vocab_size = logits.shape[-1]
  soft_targets = common_utils.onehot(
    targets, vocab_size, on_value=1., off_value=0.)
  loss = -jnp.sum(soft_targets * nn.log_softmax(logits), axis=-1)
  return loss.sum()


@functools.partial(jax.jit)
def update_transformer(transformer_net: TrainState, source: jnp.ndarray,
  target: jnp.ndarray, rng: PRNGKey) -> Tuple[PRNGKey, TrainState, InfoDict]:
  rng, key = jax.random.split(rng)

  # crossentropy loss
  def crossentropy_loss_fn(transformer_params):
    rng, predictions = transformer_net.apply_fn(transformer_params, source, key)
    crossentropy_loss = compute_cross_entropy(predictions, target)
    return crossentropy_loss, {
      'crossentropy_loss': crossentropy_loss,
      'rng': rng
    }
  loss_info, grads = jax.value_and_grad(crossentropy_loss_fn, has_aux=True)(
    transformer_net.params)
  new_transformer_net = transformer_net.apply_gradients(grads=grads)
  return loss_info[1]['rng'], new_transformer_net, loss_info[1]