import ml_collections

def get_config():
  config = ml_collections.ConfigDict()
  config.algorithm = 'SAC'
  config.actor_lr = 3e-4
  config.critic_lr = 3e-4
  config.temperature_lr = 3e-4
  config.tau = 0.005
  config.discount = 0.99
  # config.hidden_dims = (256, 256)
  config.target_update_period = 1
  # config.init_temperature = 1.0
  config.target_entropy = None
  config.replay_buffer_size = 10000
  return config