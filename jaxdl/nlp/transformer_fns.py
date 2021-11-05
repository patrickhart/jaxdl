"""Transformer NLP functions"""
from typing import Tuple, Any

import functools
import jax
from jax._src.random import randint
import jax.numpy as jnp
from optax import softmax_cross_entropy, sigmoid_binary_cross_entropy

from jaxdl.utils.commons import InfoDict, PRNGKey, TrainState


@functools.partial(jax.jit)
def update_transformer(transformer_net: TrainState, source: jnp.ndarray,
  target: jnp.ndarray, key: PRNGKey) -> Tuple[TrainState, InfoDict]:

  # temperature loss
  def crossentropy_loss_fn(transformer_params):
    predictions = transformer_net.apply_fn(transformer_params, source, key)
    crossentropy_loss = softmax_cross_entropy(predictions, target).mean()
    return crossentropy_loss, {
      'crossentropy_loss': crossentropy_loss
    }

  loss_info, grads = jax.value_and_grad(crossentropy_loss_fn, has_aux=True)(
    transformer_net.params)
  new_transformer_net = transformer_net.apply_gradients(grads=grads)
  return new_transformer_net, loss_info[1]