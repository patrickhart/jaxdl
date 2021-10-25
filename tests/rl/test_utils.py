import gym
import numpy as np
import unittest
from jaxdl.rl.utils.replay_buffer import ReplayBuffer

class TestRLUtils(unittest.TestCase):
  def test_replay_buffer(self):
    env = gym.make("Pendulum-v1")
    state = env.reset()

    replay_buffer = ReplayBuffer(
      observation_space=env.observation_space,
      action_space=env.action_space,
      capacity=100)

    terminal = False
    while not terminal:
      action = np.array([0.], dtype=np.float32)
      next_state, reward, terminal, info = env.step(action)
      replay_buffer.insert(
        state, action, reward, int(not terminal), terminal, next_state)
      state = next_state

    batch = replay_buffer.sample(20)
    self.assertEqual(len(batch), 5)
    self.assertEqual(batch[0].shape[0], 20)

if __name__ == '__main__':
  unittest.main()