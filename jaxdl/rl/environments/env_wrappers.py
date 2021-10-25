import gym
from gym.wrappers import RescaleAction

from jaxdl.rl.environments.episode_monitor import EpisodeMonitor


def make_env(
  env_name: str,
  seed: int,
  episode_monitor: bool = True,
  flatten: bool = True) -> gym.Env:

  # check and create Gym environment
  all_envs = gym.envs.registry.all()
  env_ids = [env_spec.id for env_spec in all_envs]
  if env_name not in env_ids:
    raise NotImplementedError(f"{env_name} is not available in Gym.")
  env = gym.make(env_name)

  # flatten observation
  if flatten and isinstance(env.observation_space, gym.spaces.Dict):
    env = gym.wrappers.FlattenObservation(env)

  # inject additional information
  if episode_monitor:
    env = EpisodeMonitor(env)

  # rescale actions to -1, 1 of Gym Environment
  env = RescaleAction(env, -1.0, 1.0)

  # set random seed
  env.seed(seed)
  env.action_space.seed(seed)
  env.observation_space.seed(seed)
  return env