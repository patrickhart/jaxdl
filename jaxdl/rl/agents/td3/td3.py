"""SAC-Agent implementation"""
from typing import Optional, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxdl.rl.networks.actor_nets import create_normal_dist_policy_fn, sample_actions
from jaxdl.rl.networks.critic_nets import create_double_critic_network_fn
from jaxdl.rl.networks.temperature_nets import create_temperature_network_fn
from jaxdl.rl.agents.td3.critic_fns import update_critic, update_target
from jaxdl.rl.agents.td3.actor_fns import update_actor
from jaxdl.utils.commons import InfoDict, Module, save_train_state, restore_train_state
from jaxdl.utils.commons import create_train_state
from jaxdl.rl.utils.replay_buffer import Batch
from jaxdl.rl.utils.commons import RLAgent


class TD3Agent(RLAgent):
  """An Twin-Delayed Actor Critic implementation

    Original paper: https://arxiv.org/abs/1802.09477

    Usage:
      agent = TD3Agent(0, env.observation_space, env.action_space)
      agent.restore('./tmp/')
      agent.sample(observation)
      agent.save('./tmp/')
  """
  def __init__(self,
    seed: int,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    critic_net_fn: Callable = create_double_critic_network_fn,
    actor_net_fn: Callable = create_normal_dist_policy_fn,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    discount: float = 0.99,
    tau: float = 0.005,
    target_update_period: int = 2,
    target_noise: float = 0.2,
    target_noise_clip: float = 0.5,
    explore_noise: float = 0.1):

    # split rng and generate keys
    rng = jax.random.PRNGKey(seed)
    rng, actor_key, critic_key = jax.random.split(rng, 3)

    # set target entropy
    action_dim = actions.shape[-1]

    # actor network
    actor_net = create_train_state(
      actor_net_fn(action_dim=action_dim), [actor_key, observations],
      optax.adam(learning_rate=actor_lr))

    # critic networks
    critic_net = create_train_state(
      critic_net_fn(), [critic_key, observations, actions],
      optax.adam(learning_rate=critic_lr))
    target_critic_net = create_train_state(
      critic_net_fn(), [critic_key, observations, actions],
      optax.adam(learning_rate=critic_lr))

    # networks
    self.actor_net = actor_net
    self.critic_net = critic_net
    self.target_critic_net = target_critic_net

    # parameters
    self.rng = rng
    self.step_num = 1
    self.target_update_period = target_update_period
    self.discount = discount
    self.tau = tau
    self.target_noise = target_noise
    self.target_noise_clip = target_noise_clip
    self.explore_noise = explore_noise

  def restore(self, path):
    """Loads the networks of the agents."""
    self.actor_net = restore_train_state(self.actor_net, path, prefix="actor")
    self.critic_net = restore_train_state(self.critic_net, path, prefix="critic")
    self.target_critic_net = restore_train_state(
      self.target_critic_net, path, prefix="target_critic")

  def save(self, path):
    """Saves the networks of the agents."""
    save_train_state(self.actor_net, path, prefix="actor")
    save_train_state(self.critic_net, path, prefix="critic")
    save_train_state(self.target_critic_net, path, prefix="target_critic")

  def sample(self, observations: np.ndarray,
    temperature: float = 1.0, evaluate: bool = False) -> np.ndarray:
    """Samples (clipped) actions given an observation"""
    self.rng, actions = sample_actions(
      self.rng, self.actor_net, observations, temperature)
    self.rng, key = jax.random.split(self.rng)
    if not evaluate:
      actions += self.explore_noise * jax.random.normal(key, actions.shape)
    actions = np.asarray(actions)
    # Rescaling of actions is done by gym.RescaleAction
    return np.clip(actions, -1, 1)

  def update(self, batch: Batch) -> InfoDict:
    """Updates all networks of the TD3-Agent."""
    self.step_num += 1
    # update critic
    self.rng, self.critic_net, critic_info = update_critic(
      self.rng, self.actor_net, self.critic_net, self.target_critic_net,
      batch, self.discount, target_noise=self.target_noise,
      target_noise_clip=self.target_noise_clip)

    # update target net
    if self.step_num % self.target_update_period == 0:
      self.target_critic_net = update_target(
        self.critic_net, self.target_critic_net, self.tau)

    # update actor
    self.rng, self.actor_net, actor_info = update_actor(
      self.rng, self.actor_net, self.critic_net, batch)

    # increase step count
    return {**critic_info, **actor_info}