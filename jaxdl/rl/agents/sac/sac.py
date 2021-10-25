"""SAC-Agent implementation"""
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxdl.rl.networks.actor_nets import create_normal_dist_policy_fn, sample_actions
from jaxdl.rl.networks.critic_nets import create_double_critic_network_fn
from jaxdl.rl.networks.temperature_nets import create_temperature_network_fn
from jaxdl.rl.agents.sac.critic_fns import update_critic, update_target
from jaxdl.rl.agents.sac.actor_fns import update_actor
from jaxdl.rl.agents.sac.temperature_fns import update_temperature
from jaxdl.utils.commons import InfoDict, Module, save_train_state, restore_train_state
from jaxdl.utils.commons import create_train_state
from jaxdl.rl.utils.replay_buffer import Batch
from jaxdl.rl.utils.commons import RLAgent



class SACAgent(RLAgent):
  """An JAX implementation of the Soft-Actor-Critic (SAC)

    Original paper: https://arxiv.org/abs/1812.05905

    Usage:
      agent = SACAgent(0, env.observation_space, env.action_space)
      agent.restore('./tmp/')
      agent.sample(observation)
      agent.save('./tmp/')
  """
  def __init__(self,
    seed: int,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    critic_net_fn: Module = create_double_critic_network_fn,
    actor_net_fn: Module = create_normal_dist_policy_fn,
    temperature_net_fn: Module = create_temperature_network_fn,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    temperature_lr: float = 3e-4,
    discount: float = 0.99,
    tau: float = 0.005,
    target_update_period: int = 1,
    target_entropy: Optional[float] = None):

    # split rng and generate keys
    rng = jax.random.PRNGKey(seed)
    rng, actor_key, critic_key, temperature_key = jax.random.split(rng, 4)

    # set target entropy
    action_dim = actions.shape[-1]
    self.target_entropy = target_entropy or - action_dim / 2

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

    # temperature network
    temperature_net = create_train_state(
      temperature_net_fn(), [temperature_key],
      tx=optax.adam(learning_rate=temperature_lr))

    # networks
    self.actor_net = actor_net
    self.critic_net = critic_net
    self.target_critic_net = target_critic_net
    self.temperature_net = temperature_net

    # parameters
    self.rng = rng
    self.step_num = 1
    self.target_update_period = target_update_period
    self.discount = discount
    self.tau = tau

  def restore(self, path):
    """Loads the networks of the agents."""
    self.actor_net = restore_train_state(self.actor_net, path, prefix="actor")
    self.critic_net = restore_train_state(self.critic_net, path, prefix="critic")
    self.target_critic_net = restore_train_state(
      self.target_critic_net, path, prefix="target_critic")
    self.temperature_net = restore_train_state(self.temperature_net, path, prefix="temperature")

  def save(self, path):
    """Saves the networks of the agents."""
    save_train_state(self.actor_net, path, prefix="actor")
    save_train_state(self.critic_net, path, prefix="critic")
    save_train_state(self.target_critic_net, path, prefix="target_critic")
    save_train_state(self.temperature_net, path, prefix="temperature")

  def sample(self, observations: np.ndarray,
    temperature: float = 1.0) -> jnp.ndarray:
    """Samples (clipped) actions given an observation"""
    self.rng, actions = sample_actions(
      self.rng, self.actor_net, observations, temperature)
    actions = np.asarray(actions)
    # Rescaling of actions is done by gym.RescaleAction
    return np.clip(actions, -1, 1)

  def update(self, batch: Batch) -> InfoDict:
    """Updates all networks of the SAC-Agent."""
    self.step_num += 1
    # update critic
    self.rng, self.critic_net, critic_info = update_critic(
      self.rng, self.actor_net, self.critic_net, self.target_critic_net,
      self.temperature_net, batch, self.discount, soft_critic=True)

    # update target net
    if self.step_num % self.target_update_period == 0:
      self.target_critic_net = update_target(
        self.critic_net, self.target_critic_net, self.tau)

    # update actor
    self.rng, self.actor_net, actor_info = update_actor(
      self.rng, self.actor_net, self.critic_net, self.temperature_net, batch)

    # update temperature
    self.temperature_net, alpha_info = update_temperature(
      self.temperature_net, actor_info["entropy"], self.target_entropy)

    # increase step count
    return {**critic_info, **actor_info, **alpha_info}