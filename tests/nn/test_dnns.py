import unittest
import numpy as np
from jax import random
from jaxdl.nn.dnn.mlp import MLP
from jaxdl.nn.dnn.conv import ConvMLPNet


class TestMLP(unittest.TestCase):
  def test_mlp_fwd(self):
    observations = np.array([[5., 5., 5.]], dtype=np.float32)
    rng = random.PRNGKey(0)
    mlp = MLP([24, 24], activate_final=True,
      dropout_rate=0.05)
    actor_params = mlp.init(
      rng, random.uniform(rng, (1, 3)))
    out = mlp.apply(actor_params, observations)
    self.assertEqual(out.shape[1], 24)

  def test_conv_fwd(self):
    N = 256
    observations = np.random.random((10, N, N, 3))
    rng = random.PRNGKey(0)
    mlp = ConvMLPNet([24, 24], activate_final=True,
      dropout_rate=0.05)
    actor_params = mlp.init(
      rng, random.uniform(rng, (1, N, N, 3)))
    out = mlp.apply(actor_params, observations)
    self.assertEqual(out.shape[0], 10)
    self.assertEqual(out.shape[1], 24)

if __name__ == '__main__':
  unittest.main()