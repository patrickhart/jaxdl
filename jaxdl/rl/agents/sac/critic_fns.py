"""Critic functions"""
import functools
from typing import Tuple

import jax
import jax.numpy as jnp

from jaxdl.utils.commons import InfoDict, TrainState, Params, PRNGKey
from jaxdl.rl.utils.replay_buffer import Batch


def update_target(critic: TrainState, target_critic_net: TrainState,
  tau: float) -> TrainState:
  """Function to update the target critic network

  Args:
    critic (TrainState): Critic network
    target_critic_net (TrainState): Target critic network
    tau (float): Linear interpolation parameter

  Returns:
    TrainState: Target critic network
  """
  new_target_net_params = jax.tree_multimap(
    lambda params, target_params: tau*params + (1 - tau) * target_params,
    critic.params, target_critic_net.params)
  return target_critic_net.replace(params=new_target_net_params)

@functools.partial(jax.jit, static_argnames=('soft_critic'))
def update_critic(
  rng: PRNGKey, actor_net: TrainState, critic_net: TrainState,
  target_critic_net: TrainState, temperature_net: TrainState, batch: Batch,
  discount: float, soft_critic: bool) -> Tuple[PRNGKey, TrainState, InfoDict]:
  """Update critic network

  Args:
      rng (PRNGKey): RNG
      actor_net (TrainState): Actor network
      critic_net (TrainState): Critic network
      target_critic_net (TrainState): Target critic network
      temperature_net (TrainState): Temperature network
      batch (Batch):
        Batch (observations, actions, rewards, masks, next_observations)
      discount (float): Discount factor lambda
      soft_critic (bool): Boolean flag for soft-update

  Returns:
      Tuple[PRNGKey, TrainState, InfoDict]: Updated critic network
  """

  rng, key = jax.random.split(rng)
  # policy
  policy_dist = actor_net.apply_fn(actor_net.params, batch.next_observations)
  next_actions = policy_dist.sample(seed=key)
  next_log_probs = policy_dist.log_prob(next_actions)
  # critics
  next_q1, next_q2 = target_critic_net.apply_fn(
    target_critic_net.params, batch.next_observations, next_actions)
  next_q = jnp.minimum(next_q1, next_q2)

  # create (soft) target
  target_q = batch.rewards + discount * batch.masks * next_q
  if soft_critic:
    alpha = temperature_net.apply_fn(temperature_net.params)
    target_q -= discount * batch.masks * alpha * next_log_probs

  def critic_loss_fn(critic_params: Params,
    batch: jnp.ndarray) -> Tuple[jnp.ndarray, InfoDict]:
    q1, q2 = critic_net.apply_fn(
      critic_params, batch.observations, batch.actions)
    # loss function
    critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
    return critic_loss, {
      'critic_loss': critic_loss,
      'q1': q1.mean(),
      'q2': q2.mean()
    }

  loss_info, grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(
    critic_net.params, batch)
  new_critic_net = critic_net.apply_gradients(grads=grads)
  return rng, new_critic_net, loss_info[1]