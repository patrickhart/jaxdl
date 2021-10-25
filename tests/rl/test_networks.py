import unittest
import numpy as np
import jax
from jax import random
from jaxdl.rl.networks.actor_nets import NormalDistPolicy
from jaxdl.rl.networks.critic_nets import DoubleCriticNetwork
from jaxdl.rl.networks.temperature_nets import Temperature

class TestNetworks(unittest.TestCase):
  def test_actor_net(self):
    observations = np.array([[5., 5., 5.]], dtype=np.float32)
    rng = random.PRNGKey(0)
    actor_net = NormalDistPolicy([24, 24], 2)

    actor_params = actor_net.init(
      rng, random.uniform(rng, (1, 3)))
    out = actor_net.apply(actor_params, observations)
    rng, key = jax.random.split(rng)
    sample = out.sample(seed=key)
    self.assertEqual(sample.shape[1], 2)

  def test_critic_net(self):
    observations = np.array([[5., 5., 5.]], dtype=np.float32)
    actions = np.array([[5., 5., 5.]], dtype=np.float32)
    rng = random.PRNGKey(0)
    critic_net = DoubleCriticNetwork([24, 24])

    actor_params = critic_net.init(
      rng, random.uniform(rng, (1, 3)), random.uniform(rng, (1, 3)))
    out = critic_net.apply(actor_params, observations, actions)
    self.assertEqual(len(out), 2)

  def test_temperature_net(self):
    rng = random.PRNGKey(0)
    temperature_net = Temperature()
    temperature_params = temperature_net.init(rng)
    out = temperature_net.apply(temperature_params)
    self.assertEqual(out, 1)

if __name__ == '__main__':
  unittest.main()