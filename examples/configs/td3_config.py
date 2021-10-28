import ml_collections

def get_config():
  config = ml_collections.ConfigDict()
  config.algorithm = 'TD3'
  config.actor_lr = 3e-4
  config.critic_lr = 3e-4
  config.tau = 0.005
  config.discount = 0.99
  config.target_update_period = 1
  config.replay_buffer_size = 10000
  return config