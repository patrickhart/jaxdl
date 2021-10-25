from typing import Tuple

import time
import gym
import numpy as np

from jaxdl.rl.utils.commons import TimeStep


class EpisodeMonitor(gym.ActionWrapper):
  """Environment wrapper that calculates episode returns and steps."""
  def __init__(self, env: gym.Env):
    super().__init__(env)
    self._reset_stats()
    self.timesteps = 0

  def _reset_stats(self):
    self.reward = 0.0
    self.steps = 0
    self.start_time = time.time()

  def step(self, action: np.ndarray) -> TimeStep:
    observation, reward, done, info = self.env.step(action)

    self.reward += reward
    self.steps += 1
    self.timesteps += 1
    info['timesteps'] = self.timesteps

    if done:
      info['episode'] = {}
      info['episode']['duration'] = time.time() - self.start_time
      info['episode']['return'] = self.reward
      info['episode']['length'] = self.steps
    return observation, reward, done, info

  def reset(self) -> np.ndarray:
    self._reset_stats()
    return self.env.reset()