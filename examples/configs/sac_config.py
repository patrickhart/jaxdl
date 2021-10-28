import ml_collections

from jaxdl.rl.networks.actor_nets import create_normal_dist_policy_fn
from jaxdl.rl.networks.critic_nets import create_double_critic_network_fn
from jaxdl.rl.networks.temperature_nets import create_temperature_network_fn


def get_config():
  config = ml_collections.ConfigDict()
  config.algorithm = 'SAC'
  config.critic_net_fn = create_double_critic_network_fn(
    hidden_dims=[256, 256])
  config.actor_net_fn = create_normal_dist_policy_fn(
    hidden_dims=[256, 256])
  config.temperature_net_fn = create_temperature_network_fn()
  config.actor_lr = 3e-4
  config.critic_lr = 3e-4
  config.temperature_lr = 3e-4
  config.tau = 0.005
  config.discount = 0.99
  config.target_update_period = 1
  config.target_entropy = None
  config.replay_buffer_size = 10000
  return config