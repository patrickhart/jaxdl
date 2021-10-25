"""Actor functions"""
from typing import Any, Tuple

import functools
import jax
import jax.numpy as jnp

from jaxdl.utils.commons import InfoDict, TrainState, Params, PRNGKey
from jaxdl.rl.utils.replay_buffer import Batch


@functools.partial(jax.jit)
def update_actor(rng: PRNGKey, actor_net: TrainState, critic_net: TrainState,
  temperature_net: TrainState, batch: Batch) -> Tuple[PRNGKey, TrainState, InfoDict]:
  """Update actor network

  Args:
    rng (PRNGKey): RNG
    actor_net (TrainState): Actor network
    critic_net (TrainState): Critic network
    temperature_net (TrainState): Temperature network
    batch (Batch):
      Batch (observations, actions, rewards, masks, next_observations)

  Returns:
    Tuple[PRNGKey, TrainState, InfoDict]: Updated actor network
  """

  rng, key = jax.random.split(rng)
  # actor loss
  def actor_loss_fn(actor_params: Params,
    batch: jnp.ndarray) -> Tuple[jnp.ndarray, InfoDict]:
    # policy
    policy_dist = actor_net.apply_fn(actor_params, batch.observations)
    actions = policy_dist.sample(seed=key)
    log_probs = policy_dist.log_prob(actions)
    # critic
    q1, q2 = critic_net.apply_fn(critic_net.params, batch.observations, actions)
    q = jnp.minimum(q1, q2)
    # temperature
    alpha = temperature_net.apply_fn(temperature_net.params)
    # actor loss
    actor_loss = (log_probs * alpha - q).mean()
    return actor_loss, {
      'actor_loss': actor_loss,
      'entropy': -log_probs.mean()
    }

  info, grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
    actor_net.params, batch)
  new_actor_net = actor_net.apply_gradients(grads=grads)
  return rng, new_actor_net, info[1]