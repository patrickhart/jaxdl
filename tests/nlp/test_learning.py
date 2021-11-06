from jax._src.dtypes import dtype
import numpy as np
import unittest
import jax
import optax

from jaxdl.nlp.char_dataset import CharDataset
from jaxdl.nn.transformers.transformer import Transformer, TransformerConfig
from jaxdl.nlp.transformer_fns import update_transformer
from jaxdl.utils.commons import create_train_state


class TestNLPUtils(unittest.TestCase):
  def test_text_processing(self):
    text = "Hello there. How are you?! good, very good?"
    dataset = CharDataset(data=text, block_size=10)
    # for _ in range(0, 10):
    #   idx = np.random.randint(0, len(dataset))
    #   label, target = dataset[idx]
    #   self.assertTrue(np.array_equal(label[1:], target[0:-1]))

    transformer_config = TransformerConfig(
      vocab_size=dataset.vocab_size, block_size=dataset.block_size)
    transformer_net_fn = Transformer(transformer_config)
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)

    # init
    source = jax.random.uniform(
      key, shape=(1, dataset.block_size), dtype=np.float32)
    transformer_net = create_train_state(
      transformer_net_fn, [rng, source, key], optax.adam(learning_rate=0.001))
    rng, target = transformer_net.apply_fn(transformer_net.params, source, key)

    # loss fn
    rng, new_transformer_net, info = update_transformer(
      transformer_net, source, target, key)
    print(info)

if __name__ == '__main__':
  unittest.main()