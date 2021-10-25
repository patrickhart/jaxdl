import unittest
import gym
import numpy as np
from jaxdl.rl.agents.sac.sac import SACAgent
from jaxdl.rl.utils.replay_buffer import ReplayBuffer


class TestAgents(unittest.TestCase):
  def test_sac_agent(self):
    observations = np.array([[5., 5., 5.]], dtype=np.float32)
    actions = np.array([[1., 1.]], dtype=np.float32)

    agent = SACAgent(0, observations, actions)
    output = agent.sample(observations)
    self.assertEqual(output.shape[1], 2)

  def test_sac_agent_tain(self):
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

    agent = SACAgent(0, env.observation_space.sample()[np.newaxis],
      env.action_space.sample()[np.newaxis])
    batch = replay_buffer.sample(20)
    update = agent.update(batch)
    actions = agent.sample(batch.observations)


if __name__ == '__main__':
  unittest.main()