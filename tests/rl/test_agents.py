import unittest
import gym
import numpy as np
from jaxdl.rl.agents.sac.sac import SACAgent
from jaxdl.rl.agents.td3.td3 import TD3Agent
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

    sac_agent = SACAgent(0, env.observation_space.sample()[np.newaxis],
      env.action_space.sample()[np.newaxis])
    td3_agent = TD3Agent(0, env.observation_space.sample()[np.newaxis],
      env.action_space.sample()[np.newaxis])

    # sanity check
    batch = replay_buffer.sample(20)
    sac_update = sac_agent.update(batch)
    sac_actions = sac_agent.sample(batch.observations)
    td3_update = td3_agent.update(batch)
    td3_actions = td3_agent.sample(batch.observations)


if __name__ == '__main__':
  unittest.main()